import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import contextlib
import os
from PIL import Image
from dust3r.cloud_opt.base_opt import BasePCOptimizer, edge_str
from dust3r.cloud_opt.pair_viewer import PairViewer
from dust3r.utils.geometry import xy_grid, geotrf, depthmap_to_pts3d
from dust3r.utils.device import to_cpu, to_numpy
from dust3r.utils.goem_opt import DepthBasedWarping, OccMask, WarpImage, depth_regularization_si_weighted, tum_to_pose_matrix
from third_party.raft import load_RAFT
from dust3r.utils.file_utils import *
from dust3r.utils.general_utils import *
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from object_tracker import DinoTracker



from sam2.build_sam import build_sam2_video_predictor
sam2_checkpoint = "third_party/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"



def smooth_L1_loss_fn(estimate, gt, mask, beta=1.0, per_pixel_thre=50.):
    loss_raw_shape = F.smooth_l1_loss(estimate*mask, gt*mask, beta=beta, reduction='none')
    if per_pixel_thre > 0:
        per_pixel_mask = (loss_raw_shape < per_pixel_thre) * mask
    else:
        per_pixel_mask = mask
    return torch.sum(loss_raw_shape * per_pixel_mask) / torch.sum(per_pixel_mask)

def mse_loss_fn(estimate, gt, mask):
    v = torch.sum((estimate*mask-gt*mask)**2) / torch.sum(mask)
    return v  # , v.item()

class PointCloudOptimizer(BasePCOptimizer):
    """ Optimize a global scene, given a list of pairwise observations.
    Graph node: images
    Graph edges: observations = (pred1, pred2)
    """

    # DAVIDE changed motion_mask_thre, it's default value is 0.35
    def __init__(self, *args, optimize_pp=False, focal_break=20, shared_focal=False, filelist=None, intrinsic_params=None, dist_coeffs=None, robot_poses=None, masks=None, flow_loss_fn='smooth_l1', flow_loss_weight=0.0, 
                 depth_regularize_weight=0.1, num_total_iter=300, temporal_smoothing_weight=0, translation_weight=0.1, flow_loss_start_epoch=0.15, flow_loss_thre=50,
                 sintel_ckpt=False, use_self_mask=False, pxl_thre=50, sam2_mask_refine=True, motion_mask_thre=0.35, batchify=True, **kwargs):
        super().__init__(*args, **kwargs)

        self.has_im_poses = True  # by definition of this class
        self.focal_break = focal_break
        self.num_total_iter = num_total_iter
        self.temporal_smoothing_weight = temporal_smoothing_weight
        self.translation_weight = translation_weight
        self.flow_loss_flag = False
        self.flow_loss_start_epoch = flow_loss_start_epoch
        self.flow_loss_thre = flow_loss_thre
        self.optimize_pp = optimize_pp
        self.pxl_thre = pxl_thre
        self.motion_mask_thre = motion_mask_thre
        self.batchify = batchify
        self.robot_poses = robot_poses
        self.masks = masks 
        self.imagelist = filelist
        # adding thing to optimize
        self.im_depthmaps = nn.ParameterList(torch.randn(H, W)/10-3 for H, W in self.imshapes)  # log(depth)
        self.im_poses = nn.ParameterList(self.rand_pose(self.POSE_DIM) for _ in range(self.n_imgs))  # camera poses
        self.shared_focal = shared_focal
        self.intrinsic_params = intrinsic_params
        if self.shared_focal:
            if intrinsic_params is None:
                self.im_focals = nn.ParameterList(
                    [nn.Parameter(torch.FloatTensor([self.focal_break * np.log(max(H, W))]), requires_grad=True) for H, W in self.imshapes[:1]]
                )
                self.im_pp = nn.ParameterList(torch.zeros((2,)) for _ in range(self.n_imgs))
                self.dist_coeffs = None
            else:
                # Use provided focal length within intrinsic_params
                focal = intrinsic_params['focal']
                self.im_focals = nn.ParameterList(
                    [nn.Parameter(torch.FloatTensor([self.focal_break*np.log(focal)]), requires_grad=False) for _ in range(self.n_imgs)]
                )
                px = intrinsic_params['pp'][0]
                py = intrinsic_params['pp'][1]
                self.im_pp = nn.ParameterList(
                    [nn.Parameter(torch.FloatTensor([px, py]), requires_grad=False) for _ in range(self.n_imgs)]
                )
                self.dist_coeffs = torch.tensor(dist_coeffs)


        else:
            self.im_focals = nn.ParameterList(
                [nn.Parameter(torch.FloatTensor([self.focal_break * np.log(max(H, W))]), requires_grad=True) for H, W in self.imshapes]
            )
            self.im_pp = nn.ParameterList(torch.zeros((2,)) for _ in range(self.n_imgs))  # camera intrinsics

        # Optimize robot poses
        if self.robot_poses is not None:
            self.quat_X = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0]), requires_grad=True)  # Identity quaternion
            self.trans_X = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]), requires_grad=True)  # Zero translation
            self.scale_factor = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        else:
            self.quat_X = None
            self.trans_X = None
            self.scale_factor = None
        
        self.im_pp.requires_grad_(optimize_pp)
        # print("IM PP: ", self.im_pp[0])

        self.imshape = self.imshapes[0]
        im_areas = [h*w for h, w in self.imshapes]
        self.max_area = max(im_areas)

        # adding thing to optimize
        if self.batchify:
            self.im_depthmaps = ParameterStack(self.im_depthmaps, is_param=True, fill=self.max_area) #(num_imgs, H*W)
            self.im_poses = ParameterStack(self.im_poses, is_param=True)
            self.im_focals = ParameterStack(self.im_focals, is_param=True)
            self.im_pp = ParameterStack(self.im_pp, is_param=True)
            self.register_buffer('_pp', torch.tensor([(w/2, h/2) for h, w in self.imshapes]))
            self.register_buffer('_grid', ParameterStack(
                [xy_grid(W, H, device=self.device) for H, W in self.imshapes], fill=self.max_area))
            # pre-compute pixel weights
            self.register_buffer('_weight_i', ParameterStack(
                [self.conf_trf(self.conf_i[i_j]) for i_j in self.str_edges], fill=self.max_area))
            self.register_buffer('_weight_j', ParameterStack(
                [self.conf_trf(self.conf_j[i_j]) for i_j in self.str_edges], fill=self.max_area))
            # precompute aa
            self.register_buffer('_stacked_pred_i', ParameterStack(self.pred_i, self.str_edges, fill=self.max_area))
            self.register_buffer('_stacked_pred_j', ParameterStack(self.pred_j, self.str_edges, fill=self.max_area))
            
        self.register_buffer('_ei', torch.tensor([i for i, j in self.edges]))
        self.register_buffer('_ej', torch.tensor([j for i, j in self.edges]))
        self.total_area_i = sum([im_areas[i] for i, j in self.edges])
        self.total_area_j = sum([im_areas[j] for i, j in self.edges])

        self.depth_wrapper = DepthBasedWarping()
        self.backward_warper = WarpImage()
        self.depth_regularizer = depth_regularization_si_weighted
        if flow_loss_fn == 'smooth_l1':
            self.flow_loss_fn = smooth_L1_loss_fn
        elif flow_loss_fn == 'mse':
            self.low_loss_fn = mse_loss_fn

        self.flow_loss_weight = flow_loss_weight
        self.depth_regularize_weight = depth_regularize_weight
        if self.flow_loss_weight > 0:
            self.flow_ij, self.flow_ji, self.flow_valid_mask_i, self.flow_valid_mask_j = self.get_flow(sintel_ckpt) # (num_pairs, 2, H, W)
            if use_self_mask: self.get_motion_mask_from_pairs(*args)
            # turn off the gradient for the flow
            self.flow_ij.requires_grad_(False)
            self.flow_ji.requires_grad_(False)
            self.flow_valid_mask_i.requires_grad_(False)
            self.flow_valid_mask_j.requires_grad_(False)
            if sam2_mask_refine: 
                with torch.no_grad():
                    self.refine_motion_mask_w_sam2()
            else:
                self.sam2_dynamic_masks = None

        # Dino feature extraction
        self.object_tracker = DinoTracker(self.imgs, self.masks)
        self.all_dino_descriptor, _ = self.object_tracker.extract_dino_features()
        for i, dino_descriptor_obj in enumerate(self.all_dino_descriptor):
            for j, dino_descriptor in enumerate(dino_descriptor_obj):
                self.all_dino_descriptor[i][j] = to_torch(dino_descriptor, self.device)
            

    def get_flow(self, sintel_ckpt=False): #TODO: test with gt flow
        print('precomputing flow...')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        get_valid_flow_mask = OccMask(th=3.0) # Default 3
        masks = self.masks

        pair_imgs = [np.stack(self.imgs)[self._ei], np.stack(self.imgs)[self._ej]]
        if masks is not None:
            pair_masks = [np.stack(masks)[self._ei], np.stack(masks)[self._ej]]

        flow_net = load_RAFT() if sintel_ckpt else load_RAFT("third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth")
        flow_net = flow_net.to(device)
        flow_net.eval()
        
        with torch.no_grad():
            chunk_size = 12
            flow_ij = []
            flow_ji = []
            num_pairs = len(pair_imgs[0])
            for i in tqdm(range(0, num_pairs, chunk_size)):
                end_idx = min(i + chunk_size, num_pairs)
                imgs_ij = [torch.tensor(pair_imgs[0][i:end_idx]).float().to(device),
                        torch.tensor(pair_imgs[1][i:end_idx]).float().to(device)]
                
                # if masks is not None:
                #     masks_ij = [torch.tensor(pair_masks[0][i:end_idx]).float().to(device),
                #             torch.tensor(pair_masks[1][i:end_idx]).float().to(device)]
                #     masks_ij_0 = masks_ij[0].unsqueeze(-1).expand(-1, -1, -1, 3)
                #     masks_ij_1 = masks_ij[1].unsqueeze(-1).expand(-1, -1, -1, 3)
                #     imgs_ij[0] *= masks_ij_0
                #     imgs_ij[1] *= masks_ij_1


                flow_ij_chunk = flow_net(imgs_ij[0].permute(0, 3, 1, 2) * 255, 
                                     imgs_ij[1].permute(0, 3, 1, 2) * 255, 
                                     iters=20, test_mode=True)[1]
                flow_ji_chunk = flow_net(imgs_ij[1].permute(0, 3, 1, 2) * 255, 
                                        imgs_ij[0].permute(0, 3, 1, 2) * 255, 
                                        iters=20, test_mode=True)[1]
                # Apply the mask to the flow
                # if masks is not None:
                #     mask_flow_ij = masks_ij[0].unsqueeze(1).expand(-1, 2, -1, -1)  # Mask for flow i → j
                #     mask_flow_ji = masks_ij[1].unsqueeze(1).expand(-1, 2, -1, -1)  # Mask for flow j → i
                #     flow_ij_chunk *= mask_flow_ij  # Masking flow i → j with image i mask
                #     flow_ji_chunk *= mask_flow_ji  # Masking flow j → i with image j mask

                flow_ij.append(flow_ij_chunk)
                flow_ji.append(flow_ji_chunk)
            
            flow_ij = torch.cat(flow_ij, dim=0)
            flow_ji = torch.cat(flow_ji, dim=0)
            valid_mask_i = get_valid_flow_mask(flow_ij, flow_ji)
            valid_mask_j = get_valid_flow_mask(flow_ji, flow_ij)
        print('flow precomputed')
        # delete the flow net
        if flow_net is not None: del flow_net
        return flow_ij, flow_ji, valid_mask_i, valid_mask_j

    
    def get_motion_mask_from_pairs(self, view1, view2, pred1, pred2):
        assert self.is_symmetrized, 'only support symmetric case'
        symmetry_pairs_idx = [(i, i+len(self.edges)//2) for i in range(len(self.edges)//2)]
        intrinsics_i = []
        intrinsics_j = []
        R_i = []
        R_j = []
        T_i = []
        T_j = []
        depth_maps_i = []
        depth_maps_j = []
        for i, j in tqdm(symmetry_pairs_idx):
            new_view1 = {}
            new_view2 = {}
            for key in view1.keys():
                if isinstance(view1[key], list):
                    new_view1[key] = [view1[key][i], view1[key][j]]
                    new_view2[key] = [view2[key][i], view2[key][j]]
                elif isinstance(view1[key], torch.Tensor):
                    new_view1[key] = torch.stack([view1[key][i], view1[key][j]])
                    new_view2[key] = torch.stack([view2[key][i], view2[key][j]])
            new_view1['idx'] = [0, 1]
            new_view2['idx'] = [1, 0]
            new_pred1 = {}
            new_pred2 = {}
            for key in pred1.keys():
                if isinstance(pred1[key], list):
                    new_pred1[key] = [pred1[key][i], pred1[key][j]]
                elif isinstance(pred1[key], torch.Tensor):
                    new_pred1[key] = torch.stack([pred1[key][i], pred1[key][j]])
            for key in pred2.keys():
                if isinstance(pred2[key], list):
                    new_pred2[key] = [pred2[key][i], pred2[key][j]]
                elif isinstance(pred2[key], torch.Tensor):
                    new_pred2[key] = torch.stack([pred2[key][i], pred2[key][j]])

            pair_viewer = PairViewer(new_view1, new_view2, new_pred1, new_pred2, verbose=False, intrinsics=self.intrinsic_params)
            intrinsics_i.append(pair_viewer.get_intrinsics()[0])
            intrinsics_j.append(pair_viewer.get_intrinsics()[1])
            R_i.append(pair_viewer.get_im_poses()[0][:3, :3])
            R_j.append(pair_viewer.get_im_poses()[1][:3, :3])
            T_i.append(pair_viewer.get_im_poses()[0][:3, 3:])
            T_j.append(pair_viewer.get_im_poses()[1][:3, 3:])
            depth_maps_i.append(pair_viewer.get_depthmaps()[0])
            depth_maps_j.append(pair_viewer.get_depthmaps()[1])
        
        self.intrinsics_i = torch.stack(intrinsics_i).to(self.flow_ij.device)
        self.intrinsics_j = torch.stack(intrinsics_j).to(self.flow_ij.device)
        self.R_i = torch.stack(R_i).to(self.flow_ij.device)
        self.R_j = torch.stack(R_j).to(self.flow_ij.device)
        self.T_i = torch.stack(T_i).to(self.flow_ij.device)
        self.T_j = torch.stack(T_j).to(self.flow_ij.device)
        self.depth_maps_i = torch.stack(depth_maps_i).unsqueeze(1).to(self.flow_ij.device)
        self.depth_maps_j = torch.stack(depth_maps_j).unsqueeze(1).to(self.flow_ij.device)

        ego_flow_1_2, _ = self.depth_wrapper(self.R_i, self.T_i, self.R_j, self.T_j, 1 / (self.depth_maps_i + 1e-6), self.intrinsics_j, torch.linalg.inv(self.intrinsics_i))
        ego_flow_2_1, _ = self.depth_wrapper(self.R_j, self.T_j, self.R_i, self.T_i, 1 / (self.depth_maps_j + 1e-6), self.intrinsics_i, torch.linalg.inv(self.intrinsics_j))

        err_map_i = torch.norm(ego_flow_1_2[:, :2, ...] - self.flow_ij[:len(symmetry_pairs_idx)], dim=1)
        err_map_j = torch.norm(ego_flow_2_1[:, :2, ...] - self.flow_ji[:len(symmetry_pairs_idx)], dim=1)
        # normalize the error map for each pair
        err_map_i = (err_map_i - err_map_i.amin(dim=(1, 2), keepdim=True)) / (err_map_i.amax(dim=(1, 2), keepdim=True) - err_map_i.amin(dim=(1, 2), keepdim=True))
        err_map_j = (err_map_j - err_map_j.amin(dim=(1, 2), keepdim=True)) / (err_map_j.amax(dim=(1, 2), keepdim=True) - err_map_j.amin(dim=(1, 2), keepdim=True))
        self.dynamic_masks = [[] for _ in range(self.n_imgs)]
        for i, j in symmetry_pairs_idx:
            i_idx = self._ei[i]
            j_idx = self._ej[i]
            self.dynamic_masks[i_idx].append(err_map_i[i])
            self.dynamic_masks[j_idx].append(err_map_j[i])
        
        for i in range(self.n_imgs):
            self.dynamic_masks[i] = torch.stack(self.dynamic_masks[i]).mean(dim=0) > self.motion_mask_thre
        
        if self.masks is not None:
            for i in range(len(self.masks)):
                
                self.dynamic_masks[i] = torch.from_numpy(self.masks[i])


    def get_object_masks(self):
        return self.masks
    
    def refine_motion_mask_w_sam2(self):
        """
        Refine the motion mask using SAM2.
        """
        print('Refining motion mask with SAM2...')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Save previous TF32 settings
        if device == 'cuda':
            prev_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
            prev_allow_cudnn_tf32 = torch.backends.cudnn.allow_tf32
            # Enable TF32 for Ampere GPUs
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        try:
            autocast_dtype = torch.bfloat16 if device == 'cuda' else torch.float32
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
                frame_tensors = torch.from_numpy(np.array((self.imgs))).permute(0, 3, 1, 2).to(device)
                inference_state = predictor.init_state(video_path=frame_tensors)
                
                # DAVIDE: save the init masks
                save_folder = 'init_dynamic_masks'
                os.makedirs(save_folder, exist_ok=True)
                for i in range(self.n_imgs):
                    dynamic_mask_np = self.dynamic_masks[i].cpu().numpy()
                    count = np.count_nonzero(dynamic_mask_np)
                    save_path = os.path.join(save_folder, f'dynamic_mask_{i}.png')

                    # Create an image with colored dynamic pixels and black static pixels
                    img1_np = self.imgs[i].astype(np.uint8)
                    colored_dynamic_pixels = np.zeros_like(img1_np)
                    colored_dynamic_pixels[dynamic_mask_np > 0.99] = [255, 255, 255]

                    # Save the image
                    dynamic_pixels_img = Image.fromarray(colored_dynamic_pixels)
                    dynamic_pixels_img.save(save_path)
                ###################



                mask_list = [self.dynamic_masks[i] for i in range(self.n_imgs)]
                
                ann_obj_id = 1
                self.sam2_dynamic_masks = [[] for _ in range(self.n_imgs)]
        
                # Process even frames
                predictor.reset_state(inference_state)
                for idx, mask in enumerate(mask_list):
                    if idx % 2 == 1:
                        _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                            inference_state,
                            frame_idx=idx,
                            obj_id=ann_obj_id,
                            mask=mask,
                        )


                video_segments = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                for out_frame_idx in range(self.n_imgs):
                    if out_frame_idx % 2 == 0:
                        self.sam2_dynamic_masks[out_frame_idx] = video_segments[out_frame_idx][ann_obj_id]
        
                # Process odd frames
                predictor.reset_state(inference_state)
                for idx, mask in enumerate(mask_list):
                    if idx % 2 == 0:
                        _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                            inference_state,
                            frame_idx=idx,
                            obj_id=ann_obj_id,
                            mask=mask,
                        )
                video_segments = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                for out_frame_idx in range(self.n_imgs):
                    if out_frame_idx % 2 == 1:
                        self.sam2_dynamic_masks[out_frame_idx] = video_segments[out_frame_idx][ann_obj_id]
        
                # Update dynamic masks
                for i in range(self.n_imgs):
                    self.sam2_dynamic_masks[i] = torch.from_numpy(self.sam2_dynamic_masks[i][0]).to(device)
                    self.dynamic_masks[i] = self.dynamic_masks[i].to(device)
                    self.dynamic_masks[i] = self.dynamic_masks[i] | self.sam2_dynamic_masks[i]

                # DAVIDE: save the final refined masks
                save_folder = 'ref_dynamic_masks'
                os.makedirs(save_folder, exist_ok=True)
                for i in range(self.n_imgs):
                    dynamic_mask_np = self.dynamic_masks[i].cpu().numpy()
                    count = np.count_nonzero(dynamic_mask_np)
                    save_path = os.path.join(save_folder, f'dynamic_mask_{i}.png')

                    # Create an image with colored dynamic pixels and black static pixels
                    img1_np = self.imgs[i].astype(np.uint8)
                    colored_dynamic_pixels = np.zeros_like(img1_np)
                    colored_dynamic_pixels[dynamic_mask_np > 0.99] = [255, 255, 255]


                    # Save the image
                    dynamic_pixels_img = Image.fromarray(colored_dynamic_pixels)
                    dynamic_pixels_img.save(save_path)
                #################################


                # Clean up
                del predictor
        finally:
            # Restore previous TF32 settings
            if device == 'cuda':
                torch.backends.cuda.matmul.allow_tf32 = prev_allow_tf32
                torch.backends.cudnn.allow_tf32 = prev_allow_cudnn_tf32


    def _check_all_imgs_are_selected(self, msk):
        self.msk = torch.from_numpy(np.array(msk, dtype=bool)).to(self.device)
        assert np.all(self._get_msk_indices(msk) == np.arange(self.n_imgs)), 'incomplete mask!'
        pass

    def preset_pose(self, known_poses, pose_msk=None, requires_grad=False):  # cam-to-world
        self._check_all_imgs_are_selected(pose_msk)

        if isinstance(known_poses, torch.Tensor) and known_poses.ndim == 2:
            known_poses = [known_poses]
        if known_poses.shape[-1] == 7: # xyz wxyz
            known_poses = [tum_to_pose_matrix(pose) for pose in known_poses]
        for idx, pose in zip(self._get_msk_indices(pose_msk), known_poses):
            if self.verbose:
                print(f' (setting pose #{idx} = {pose[:3,3]})')
            self._no_grad(self._set_pose(self.im_poses, idx, torch.tensor(pose)))

        # normalize scale if there's less than 1 known pose
        n_known_poses = sum((p.requires_grad is False) for p in self.im_poses)
        self.norm_pw_scale = (n_known_poses <= 1)
        if len(known_poses) == self.n_imgs:
            if requires_grad:
                self.im_poses.requires_grad_(True)
            else:
                self.im_poses.requires_grad_(False)
        self.norm_pw_scale = False

    def preset_intrinsics(self, known_intrinsics, msk=None):
        if isinstance(known_intrinsics, torch.Tensor) and known_intrinsics.ndim == 2:
            known_intrinsics = [known_intrinsics]
        for K in known_intrinsics:
            assert K.shape == (3, 3)
        self.preset_focal([K.diagonal()[:2].mean() for K in known_intrinsics], msk)
        if self.optimize_pp:
            self.preset_principal_point([K[:2, 2] for K in known_intrinsics], msk)

    def preset_focal(self, known_focals, msk=None, requires_grad=False):
        self._check_all_imgs_are_selected(msk)

        for idx, focal in zip(self._get_msk_indices(msk), known_focals):
            if self.verbose:
                print(f' (setting focal #{idx} = {focal})')
            self._no_grad(self._set_focal(idx, focal))
        if len(known_focals) == self.n_imgs:
            if requires_grad:
                self.im_focals.requires_grad_(True)
            else:
                self.im_focals.requires_grad_(False)

    def preset_principal_point(self, known_pp, msk=None):
        self._check_all_imgs_are_selected(msk)

        for idx, pp in zip(self._get_msk_indices(msk), known_pp):
            if self.verbose:
                print(f' (setting principal point #{idx} = {pp})')
            self._no_grad(self._set_principal_point(idx, pp))

        self.im_pp.requires_grad_(False)

    def _get_msk_indices(self, msk):
        if msk is None:
            return range(self.n_imgs)
        elif isinstance(msk, int):
            return [msk]
        elif isinstance(msk, (tuple, list)):
            return self._get_msk_indices(np.array(msk))
        elif msk.dtype in (bool, torch.bool, np.bool_):
            assert len(msk) == self.n_imgs
            return np.where(msk)[0]
        elif np.issubdtype(msk.dtype, np.integer):
            return msk
        else:
            raise ValueError(f'bad {msk=}')

    def _get_robot_poses(self):
        return self.robot_poses
    
    def _get_scale_factor(self):
        return self.scale_factor

    def _get_quat_X(self):
        return self.quat_X
    
    def _get_trans_X(self):
        return self.trans_X

    def _get_calibration_params(self):
        return self.quat_X, self.trans_X, self.scale_factor

    def _no_grad(self, tensor):
        assert tensor.requires_grad, 'it must be True at this point, otherwise no modification occurs'

    def _set_focal(self, idx, focal, force=False):
        param = self.im_focals[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = self.focal_break * np.log(focal)
        return param

    def get_focals(self):
        if self.shared_focal:
            log_focals = torch.stack([self.im_focals[0]] * self.n_imgs, dim=0)
        else:
            log_focals = torch.stack(list(self.im_focals), dim=0)
        return (log_focals / self.focal_break).exp()

    def get_known_focal_mask(self):
        return torch.tensor([not (p.requires_grad) for p in self.im_focals])

    def _set_principal_point(self, idx, pp, force=False):
        param = self.im_pp[idx]
        H, W = self.imshapes[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = to_cpu(to_numpy(pp) - (W/2, H/2)) / 10
        return param

    def get_principal_points_non_batch(self):
        return torch.stack([pp.new((W/2, H/2))+10*pp for pp, (H, W) in zip(self.im_pp, self.imshapes)])

    def get_principal_points_batch(self):
        # Return default principal points if not set
        if self.im_pp[0][0]==0.0:
            return self._pp + 10 * self.im_pp
        else:
            return self.im_pp
        
    def get_principal_points(self):
        if self.batchify:
            return self.get_principal_points_batch()
        else:
            return self.get_principal_points_non_batch()

    def get_intrinsics(self):
        K = torch.zeros((self.n_imgs, 3, 3), device=self.device)
        focals = self.get_focals().flatten()
        K[:, 0, 0] = K[:, 1, 1] = focals
        K[:, :2, 2] = self.get_principal_points()
        K[:, 2, 2] = 1
        return K

    def get_dist_coeffs(self):
        dist_coeff = torch.zeros((self.n_imgs, 5), device=self.device)
        for i in range(self.n_imgs):
            print("Coeff provided: ", self.dist_coeffs)
            dist_coeff[i, :] = self.dist_coeffs
        return dist_coeff
    

    def get_im_poses_batch(self):  # cam to world
        
        cam2world = self._get_poses(self.im_poses)

        return cam2world

    def get_im_poses_non_batch(self):  # cam to world
        cam2world = self._get_poses(torch.stack(list(self.im_poses)))
        return cam2world

    def get_im_poses(self):
        if self.batchify:
            return self.get_im_poses_batch()
        else:
            return self.get_im_poses_non_batch()

    def _set_depthmap_batch(self, idx, depth, force=False):
        depth = _ravel_hw(depth, self.max_area)

        param = self.im_depthmaps[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = depth.log().nan_to_num(neginf=0)
        return param

    def _set_depthmap_non_batch(self, idx, depth, force=False):
        param = self.im_depthmaps[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = depth.log().nan_to_num(neginf=0)
        return param

    def _set_depthmap(self, idx, depth, force=False):
        if self.batchify:
            return self._set_depthmap_batch(idx, depth, force)
        else:
            return self._set_depthmap_non_batch(idx, depth, force)
    
    def preset_depthmap(self, known_depthmaps, msk=None, requires_grad=False):
        self._check_all_imgs_are_selected(msk)

        for idx, depth in zip(self._get_msk_indices(msk), known_depthmaps):
            if self.verbose:
                print(f' (setting depthmap #{idx})')
            self._no_grad(self._set_depthmap(idx, depth))

        if len(known_depthmaps) == self.n_imgs:
            if requires_grad:
                self.im_depthmaps.requires_grad_(True)
            else:
                self.im_depthmaps.requires_grad_(False)
    
    def _set_init_depthmap(self):
        depth_maps = self.get_depthmaps(raw=True)
        self.init_depthmap = [dm.detach().clone() for dm in depth_maps]

    def get_init_depthmaps(self, raw=False):
        res = self.init_depthmap
        if not raw:
            res = [dm[:h*w].view(h, w) for dm, (h, w) in zip(res, self.imshapes)]
        return res

    def get_depthmaps_batch(self, raw=False):
        res = self.im_depthmaps.exp()
        if not raw:
            res = [dm[:h*w].view(h, w) for dm, (h, w) in zip(res, self.imshapes)]
        return res

    def get_depthmaps_non_batch(self):
        return [d.exp() for d in self.im_depthmaps]

    def get_depthmaps(self, raw=False):
        if self.batchify:
            return self.get_depthmaps_batch(raw)
        else:
            return self.get_depthmaps_non_batch()

    def get_object_pts3d(self, masks_list, rel_ptmaps):
        masked_pts_dict = {}
        for i, mask in enumerate(masks_list):
            unique_masks = np.unique(mask)  # Find unique values in mask_flat
            for mask_value in unique_masks:
                if mask_value == 0:
                    continue
                object_mask = (mask == mask_value)
                mask_tensor = torch.tensor(object_mask, dtype=torch.bool, device=rel_ptmaps.device)
                flattened_mask = mask_tensor.view(-1)

                # Collect points corresponding to the current mask_value
                masked_points = rel_ptmaps[i][flattened_mask] 

                # Store in a dictionary where the key is the mask value
                if mask_value not in masked_pts_dict:
                    masked_pts_dict[mask_value] = []
                masked_pts_dict[mask_value].append(masked_points)
        
        # Convert the dictionary to a list format if needed
        masked_pts_list_transposed = list(masked_pts_dict.values()) 

        return masked_pts_list_transposed


    def depth_to_pts3d(self):
        # Get depths and  projection params if not provided
        focals = self.get_focals()
        pp = self.get_principal_points()
        im_poses = self.get_im_poses()
        depth = self.get_depthmaps(raw=True)

        # get pointmaps in camera frame
        rel_ptmaps = _fast_depthmap_to_pts3d(depth, self._grid, focals, pp=pp)
        transformed_ptmaps = geotrf(im_poses, rel_ptmaps)
        masks = self.get_object_masks()
        if masks is not None:
            object_pts3d = self.get_object_pts3d(masks, transformed_ptmaps)
        else:
            object_pts3d = None
        # project to world frame
        return transformed_ptmaps, object_pts3d

    def depth_to_pts3d_partial(self):
        # Get depths and  projection params if not provided
        focals = self.get_focals()
        pp = self.get_principal_points()
        im_poses = self.get_im_poses()
        depth = self.get_depthmaps()

        # convert focal to (1,2,H,W) constant field
        def focal_ex(i): return focals[i][..., None, None].expand(1, *focals[i].shape, *self.imshapes[i])
        # get pointmaps in camera frame
        rel_ptmaps = [depthmap_to_pts3d(depth[i][None], focal_ex(i), pp=pp[i:i+1])[0] for i in range(im_poses.shape[0])]
        # project to world frame
        return [geotrf(pose, ptmap) for pose, ptmap in zip(im_poses, rel_ptmaps)]
    
    def get_pts3d_batch(self, raw=False, **kwargs):
        res, res_object = self.depth_to_pts3d()
        if not raw:
            res = [dm[:h*w].view(h, w, 3) for dm, (h, w) in zip(res, self.imshapes)]
        return res, res_object

    def get_pts3d(self, raw=False, **kwargs):
        if self.batchify:
            return self.get_pts3d_batch(raw, **kwargs)
        else:
            return self.depth_to_pts3d_partial()

    def get_relative_poses(self, cam2w, scale_factor=None):
        # Inverse of the first camera pose
        cam2w_0_inv = torch.inverse(cam2w[0])
        
        # Compute relative poses
        ccam2pcam = [cam2w_0_inv @ cam2w_pose for cam2w_pose in cam2w]
        if scale_factor is not None:
            for i in range(len(ccam2pcam)):
                ccam2pcam[i][:3, 3] *= scale_factor
        return ccam2pcam
    
    

    def calibration_loss(self, cam2w, robot_poses, scale_factor, quat_X, trans_X):
        """
        Loss function exploiting robot kinematics with quaternion rotation representation.
        """
        loss = 0.0

        # Normalize quaternion
        quat_X = quat_X / quat_X.norm()
        X_rot = quaternion_to_matrix(quat_X)
        
        # Let's compute the inverse of camera poses
        w2cam = torch.linalg.inv(cam2w)
        # w2cam = cam2w
        # Ensure tensors are on the correct device and dtype
        device = w2cam[0].device
        dtype = w2cam[0].dtype  # Use the same dtype as w2cam tensors
        trans_X = trans_X.to(device).to(dtype)
        scale_factor = scale_factor.to(device).to(dtype)
        scale_factor = torch.abs(scale_factor)
        print(f"Scale factor: {scale_factor}")
        quat_X = quat_X.to(device).to(dtype)
        X_rot = X_rot.to(device).to(dtype)

        # Construct transformation matrix X
        X = torch.cat([torch.cat([X_rot, trans_X.view(3, 1)], dim=1), 
                    torch.tensor([[0, 0, 0, 1]], device=device, dtype=dtype)], dim=0)
        
        # Compute the rotation magnitude
        rotation_magnitude_list = []
        for i in range(1, len(w2cam)):
            A = robot_poses[i - 1]
            angle_axis = matrix_to_axis_angle(A[:3, :3])
            rotation_magnitude = torch.norm(angle_axis, dim=0)
            rotation_magnitude_list.append(rotation_magnitude)
        max_val = max(rotation_magnitude_list)
        min_val = min(rotation_magnitude_list)
        rotation_magnitude_list = [(val - min_val) / (max_val - min_val) for val in rotation_magnitude_list]
        rotation_magnitude_list = [val**2 for val in rotation_magnitude_list]

        for i in range(1, len(w2cam)):
            # Compute relative pose for robot and camera
            A = robot_poses[i - 1]
            # B = R_z@(w2cam[i - 1]) @ torch.linalg.inv(w2cam[i]) @ R_z
            B = w2cam[i - 1] @ cam2w[i] 
            
            # Ensure all tensors are on the same device
            A = A.to(device).to(dtype)
            B = B.to(device).to(dtype)
            B_rotated = B.clone()
            B_rotated[:3, :3] =  B_rotated[:3, :3]
            # Compute chain transformations
            chain1 = A
            chain2 = X @ B_rotated @ torch.linalg.inv(X)
        
            # Scale the translation part of chain2
            chain2 = chain2.clone()
            chain2[:3, 3] *= scale_factor
            
            angle_axis_camera = matrix_to_axis_angle(chain2[:3, :3])
            rotation_magnitude_camera = torch.norm(angle_axis_camera, dim=0)

            # Compute rotation loss
            chain1_quat = matrix_to_quaternion(chain1[:3, :3])
            chain2_quat = matrix_to_quaternion(chain2[:3, :3])
            rotation_magnitude = rotation_magnitude_list[i - 1]
            rotation_loss = rotation_magnitude * torch.nn.functional.mse_loss(chain1_quat, chain2_quat)

            # Compute translation loss
            translation_loss = rotation_magnitude * torch.nn.functional.mse_loss(chain1[:3, 3], chain2[:3, 3])
            # print(f"Rotation loss {torch.nn.functional.mse_loss(chain1_quat, chain2_quat)} - Translation loss {torch.nn.functional.mse_loss(chain1[:3, 3], chain2[:3, 3])}")
            # Combine losses
            loss += rotation_loss + translation_loss
        
        return loss

    def estimate_rigid_transform(self, src_pts, tgt_pts, src_feats, tgt_feats):
        """
        Stima la trasformazione rigida tra due nuvole di punti usando feature matching.
        
        Args:
            src_pts (torch.Tensor): Nuvola di punti sorgente (N, 3)
            tgt_pts (torch.Tensor): Nuvola di punti target (M, 3)
            src_feats (torch.Tensor): Feature DINO per src_pts (N, D)
            tgt_feats (torch.Tensor): Feature DINO per tgt_pts (M, D)

        Returns:
            R (torch.Tensor): Rotazione stimata (3,3)
            T (torch.Tensor): Traslazione stimata (3,)
        """
        src_pts = src_pts.cpu().numpy()
        tgt_pts = tgt_pts.cpu().numpy()
        src_feats = src_feats.cpu().numpy()
        tgt_feats = tgt_feats.cpu().numpy()
        
        # Trova le corrispondenze con KDTree
        print("Start KDTree")
        tree = cKDTree(tgt_feats)
        print("End KDTree")
        dists, indices = tree.query(src_feats, k=1)
        
        # Seleziona i punti corrispondenti
        matched_src_pts = src_pts
        matched_tgt_pts = tgt_pts[indices]

        # Centroidi
        centroid_src = np.mean(matched_src_pts, axis=0)
        centroid_tgt = np.mean(matched_tgt_pts, axis=0)

        # Centra i punti
        src_pts_centered = matched_src_pts - centroid_src
        tgt_pts_centered = matched_tgt_pts - centroid_tgt

        # SVD per trovare R
        H = np.dot(src_pts_centered.T, tgt_pts_centered)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # Assicuriamoci che sia una rotazione valida
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)

        # Calcola la traslazione
        T = centroid_tgt - np.dot(R, centroid_src)

        return torch.tensor(R, dtype=torch.float32), torch.tensor(T, dtype=torch.float32)

    def object_trajectory_loss(self, proj_obj_pts3d, all_dino_descriptor, weight=1.0):
        """
        Ottimizza la traiettoria 3D dell'oggetto minimizzando la distanza tra le pose successive.
        
        Args:
            proj_obj_pts3d: Lista di nuvole di punti 3D dell'oggetto (B, N, 3)
            all_dino_descriptor: Lista delle feature DINO corrispondenti ai punti 3D (B, N, D)
            weight: Peso della loss

        Returns:
            trajectory_loss: Penalità per incoerenza della traiettoria
        """
        loss = 0
        device = proj_obj_pts3d[0].device
        print("Device: ", device)
        for i in range(len(proj_obj_pts3d) - 1):
            
            print("Debug: ", i)
            src_pts = proj_obj_pts3d[i].to(device)  # Move to same device
            tgt_pts = proj_obj_pts3d[i + 1].to(device)
            print("DEBUG")
            src_feats = all_dino_descriptor[i]
            tgt_feats = all_dino_descriptor[i + 1]


            # Stima la trasformazione rigida tra le due pose
            R, T = self.estimate_rigid_transform(src_pts, tgt_pts, src_feats, tgt_feats)
            print("R: ", R)
            print("T: ", T)
            R, T = R.to(device), T.to(device)  # Move R and T to same device

            # Trasforma i punti della vista i nella vista i+1
            src_pts_transformed = (R @ src_pts.T).T + T

            # Chamfer Distance tra punti trasformati e punti reali nella vista successiva
            loss += chamfer_loss(src_pts_transformed, tgt_pts)
            print("Loss: ", loss)
        return weight * loss / (len(proj_obj_pts3d) - 1)


    def forward_batchify(self, epoch=9999, niter=None):
        pw_poses = self.get_pw_poses()  # cam-to-world

        pw_adapt = self.get_adaptors().unsqueeze(1)
        
        all_dino_descriptor = self.all_dino_descriptor
        proj_pts3d, proj_obj_pts3d = self.get_pts3d(raw=True)
                                     
        # rotate pairwise prediction according to pw_poses
        aligned_pred_i = geotrf(pw_poses, pw_adapt * self._stacked_pred_i)
        aligned_pred_j = geotrf(pw_poses, pw_adapt * self._stacked_pred_j)

        # compute the loss
        li = self.dist(proj_pts3d[self._ei], aligned_pred_i, weight=self._weight_i).sum() / self.total_area_i
        lj = self.dist(proj_pts3d[self._ej], aligned_pred_j, weight=self._weight_j).sum() / self.total_area_j

        # camera temporal loss
        if self.temporal_smoothing_weight > 0:
            # Compute the temporal smoothing loss
            temporal_smoothing_loss = self.relative_pose_loss(self.get_im_poses()[:-1], self.get_im_poses()[1:]).sum()
        else:
            temporal_smoothing_loss = 0

        if self.flow_loss_weight > 0 and epoch >= self.num_total_iter * self.flow_loss_start_epoch: # enable flow loss after certain epoch
            R_all, T_all = self.get_im_poses()[:,:3].split([3, 1], dim=-1)
            R1, T1 = R_all[self._ei], T_all[self._ei]
            R2, T2 = R_all[self._ej], T_all[self._ej]
            K_all = self.get_intrinsics()
            inv_K_all = torch.linalg.inv(K_all)
            K_1, inv_K_1 = K_all[self._ei], inv_K_all[self._ei]
            K_2, inv_K_2 = K_all[self._ej], inv_K_all[self._ej]
            depth_all = torch.stack(self.get_depthmaps(raw=False)).unsqueeze(1)
            depth1, depth2 = depth_all[self._ei], depth_all[self._ej]
            disp_1, disp_2 = 1 / (depth1 + 1e-6), 1 / (depth2 + 1e-6)
            ego_flow_1_2, _ = self.depth_wrapper(R1, T1, R2, T2, disp_1, K_2, inv_K_1)
            ego_flow_2_1, _ = self.depth_wrapper(R2, T2, R1, T1, disp_2, K_1, inv_K_2)
            dynamic_masks_all = torch.stack(self.dynamic_masks).to(self.device).unsqueeze(1)
            dynamic_mask1, dynamic_mask2 = dynamic_masks_all[self._ei], dynamic_masks_all[self._ej]
            flow_loss_i = self.flow_loss_fn(ego_flow_1_2[:, :2, ...], self.flow_ij, ~dynamic_mask1, per_pixel_thre=self.pxl_thre)
            flow_loss_j = self.flow_loss_fn(ego_flow_2_1[:, :2, ...], self.flow_ji, ~dynamic_mask2, per_pixel_thre=self.pxl_thre)
            flow_loss = flow_loss_i + flow_loss_j
            print(f'flow loss: {flow_loss.item()}')
            if flow_loss.item() > self.flow_loss_thre and self.flow_loss_thre > 0: 
                flow_loss = 0
                self.flow_loss_flag = True
        else:    
            flow_loss = 0

        if self.depth_regularize_weight > 0:
            init_depthmaps = torch.stack(self.get_init_depthmaps(raw=False)).unsqueeze(1)
            depthmaps = torch.stack(self.get_depthmaps(raw=False)).unsqueeze(1)
            dynamic_masks_all = torch.stack(self.dynamic_masks).to(self.device).unsqueeze(1)
            depth_prior_loss = self.depth_regularizer(depthmaps, init_depthmaps, dynamic_masks_all)
        else:
            depth_prior_loss = 0

        robot_poses = self._get_robot_poses()
        
        quat_X = self._get_quat_X()
        trans_X = self._get_trans_X()
        cam2w = self.get_im_poses()
        scale_factor = self._get_scale_factor()
        if scale_factor is not None:
            scale_factor = torch.abs(scale_factor)

        if quat_X is not None:
            calibration_loss_value = self.calibration_loss(cam2w, robot_poses, scale_factor, quat_X, trans_X)
            if niter is not None:
                weight_calib = 1 + epoch / niter
            else:
                weight_calib = 1
        else:
            weight_calib = 0
            calibration_loss_value = 0

        
        # Manually set to 0
        # self.temporal_smoothing_weight = 0 # It enabes the temporal smoothing loss; i.e., the similarity between adjacent frames
        if self.masks is not None:
            self.flow_loss_weight = 0
        # obj_traj_loss = self.object_trajectory_loss(proj_obj_pts3d[0], all_dino_descriptor[0], weight=0.1)
        
        loss = (li + lj) * 1 + self.temporal_smoothing_weight * temporal_smoothing_loss + \
                self.flow_loss_weight * flow_loss + self.depth_regularize_weight * depth_prior_loss + weight_calib* calibration_loss_value 
        
        if quat_X is not None:
            quat_X.data = quat_X.data / quat_X.data.norm()
        return loss
    
    def forward_non_batchify(self, epoch=9999):

        # --(1) Perform the original pairwise 3D consistency loss (pairwise 3D consistency)--
        pw_poses = self.get_pw_poses()  # pair-wise poses (or adaptive poses)
        pw_adapt = self.get_adaptors()
        proj_pts3d, _ = self.get_pts3d()   # 3D point clouds for each image
        weight_i = {i_j: self.conf_trf(c) for i_j, c in self.conf_i.items()}
        weight_j = {i_j: self.conf_trf(c) for i_j, c in self.conf_j.items()}

        loss = 0.0
        for e, (i, j) in enumerate(self.edges):
            i_j = edge_str(i, j)
            # Transform the pairwise predictions to the world coordinate system
            aligned_pred_i = geotrf(pw_poses[e], pw_adapt[e] * self.pred_i[i_j])
            aligned_pred_j = geotrf(pw_poses[e], pw_adapt[e] * self.pred_j[i_j])
            # Compute the distance loss between the projected point clouds and the predictions
            li = self.dist(proj_pts3d[i], aligned_pred_i, weight=weight_i[i_j]).mean()
            lj = self.dist(proj_pts3d[j], aligned_pred_j, weight=weight_j[i_j]).mean()
            loss += (li + lj)

        # Average the loss
        loss /= self.n_edges

        # --(2) Add temporal smoothing constraint between adjacent frames (temporal smoothing)--
        temporal_smoothing_loss = 0.0
        if self.temporal_smoothing_weight > 0:
            # Get the global poses (4x4) for all images
            im_poses = self.get_im_poses()  # shape: (n_imgs, 4, 4)
            # Stack the relative poses between adjacent frames and use the existing relative_pose_loss function
            rel_RT1, rel_RT2 = [], []
            for idx in range(self.n_imgs - 1):
                rel_RT1.append(im_poses[idx])
                rel_RT2.append(im_poses[idx + 1])
            if len(rel_RT1) > 0:
                rel_RT1 = torch.stack(rel_RT1, dim=0)  # shape: (n_imgs-1, 4, 4)
                rel_RT2 = torch.stack(rel_RT2, dim=0)
                # Compute the pose difference between adjacent frames
                temporal_smoothing_loss = self.relative_pose_loss(rel_RT1, rel_RT2).sum()
                loss += self.temporal_smoothing_weight * temporal_smoothing_loss

        # --(3) Add flow constraint (flow_loss), similar to forward_batchify--
        flow_loss = 0.0
        if self.flow_loss_weight > 0 and epoch >= self.num_total_iter * self.flow_loss_start_epoch:
            print(f"Computing flow loss at epoch {epoch}")
            # Iterate through each pair of images and compute the depth map and flow comparison
            im_poses = self.get_im_poses()   # (n_imgs, 4, 4)
            K_all = self.get_intrinsics()    # (n_imgs, 3, 3)
            inv_K_all = torch.linalg.inv(K_all)
            depthmaps = self.get_depthmaps(raw=False)  # list of depth maps (H, W)

            for e, (i, j) in enumerate(self.edges):
                # Get the rotation, translation, and intrinsics for the two frames
                R1 = im_poses[i][:3, :3].unsqueeze(0)  # shape: (1, 3, 3)
                T1 = im_poses[i][:3, 3].unsqueeze(-1).unsqueeze(0)  # (1, 3, 1)
                R2 = im_poses[j][:3, :3].unsqueeze(0)
                T2 = im_poses[j][:3, 3].unsqueeze(-1).unsqueeze(0)
                K1 = K_all[i].unsqueeze(0)     # (1, 3, 3)
                K2 = K_all[j].unsqueeze(0)
                inv_K1 = inv_K_all[i].unsqueeze(0)
                inv_K2 = inv_K_all[j].unsqueeze(0)

                # Construct disparity: disp = 1/depth
                depth1 = depthmaps[i].unsqueeze(0).unsqueeze(1)  # (1, 1, H, W)
                depth2 = depthmaps[j].unsqueeze(0).unsqueeze(1)
                disp_1 = 1.0 / (depth1 + 1e-6)
                disp_2 = 1.0 / (depth2 + 1e-6)

                # Compute "ego-motion flow" by projecting using DepthBasedWarping
                # Note that DepthBasedWarping expects batch dimension, so add unsqueeze(0)
                ego_flow_1_2, _ = self.depth_wrapper(R1, T1, R2, T2, disp_1, K2, inv_K1)
                ego_flow_2_1, _ = self.depth_wrapper(R2, T2, R1, T1, disp_2, K1, inv_K2)

                # Get the corresponding dynamic region masks (if any)
                dynamic_mask_i = self.dynamic_masks[i]  # shape: (H, W)
                dynamic_mask_j = self.dynamic_masks[j]

                # When computing flow loss, exclude or ignore dynamic regions
                flow_loss_i = self.flow_loss_fn(
                    ego_flow_1_2[0, :2, ...],   # shape: (2, H, W)
                    self.flow_ij[e],           # shape: (2, H, W),  i->j
                    ~dynamic_mask_i,           # mask: True = keep, False = ignore
                    per_pixel_thre=self.pxl_thre
                )
                flow_loss_j = self.flow_loss_fn(
                    ego_flow_2_1[0, :2, ...],
                    self.flow_ji[e],           # j->i
                    ~dynamic_mask_j,
                    per_pixel_thre=self.pxl_thre
                )
                flow_loss += (flow_loss_i + flow_loss_j)

            # Optional: handle cases where the flow loss is too large (e.g., early stop)
            # divide by the number of edges
            flow_loss /= self.n_edges
            print(f'flow loss: {flow_loss.item()}')
            if flow_loss.item() > self.flow_loss_thre and self.flow_loss_thre > 0:
                flow_loss = 0.0

            loss += self.flow_loss_weight * flow_loss

        # --(4) Add depth regularization (depth_prior_loss) to constrain the initial depth--
        if self.depth_regularize_weight > 0:
            init_depthmaps = self.get_init_depthmaps(raw=False)  # initial depth maps
            current_depthmaps = self.get_depthmaps(raw=False)     # current optimized depth maps
            depth_prior_loss = 0.0
            for i in range(self.n_imgs):
                # Apply constraints on static regions (ignore dynamic regions)
                # Make sure the shape has the batch dimension (B,1,H,W)
                depth_prior_loss += self.depth_regularizer(
                    current_depthmaps[i].unsqueeze(0).unsqueeze(1),
                    init_depthmaps[i].unsqueeze(0).unsqueeze(1),
                    self.dynamic_masks[i].unsqueeze(0).unsqueeze(1)
                )
            loss += self.depth_regularize_weight * depth_prior_loss

        return loss

    def forward(self, epoch=9999, niter=None):
        if self.batchify:
            return self.forward_batchify(epoch, niter=niter)
        else:
            return self.forward_non_batchify(epoch)

    def relative_pose_loss(self, RT1, RT2):
        relative_RT = torch.matmul(torch.inverse(RT1), RT2)
        rotation_diff = relative_RT[:, :3, :3]
        translation_diff = relative_RT[:, :3, 3]

        # Frobenius norm for rotation difference
        rotation_loss = torch.norm(rotation_diff - (torch.eye(3, device=RT1.device)), dim=(1, 2))

        # L2 norm for translation difference
        translation_loss = torch.norm(translation_diff, dim=1)

        # Combined loss (one can weigh these differently if needed)
        pose_loss = rotation_loss + translation_loss * self.translation_weight
        return pose_loss

def _fast_depthmap_to_pts3d(depth, pixel_grid, focal, pp):
    pp = pp.unsqueeze(1)
    focal = focal.unsqueeze(1)
    assert focal.shape == (len(depth), 1, 1)
    assert pp.shape == (len(depth), 1, 2)
    assert pixel_grid.shape == depth.shape + (2,)
    depth = depth.unsqueeze(-1)
    return torch.cat((depth * (pixel_grid - pp) / focal, depth), dim=-1)


def ParameterStack(params, keys=None, is_param=None, fill=0):
    if keys is not None:
        params = [params[k] for k in keys]

    if fill > 0:
        params = [_ravel_hw(p, fill) for p in params]

    requires_grad = params[0].requires_grad
    assert all(p.requires_grad == requires_grad for p in params)

    params = torch.stack(list(params)).float().detach()
    if is_param or requires_grad:
        params = nn.Parameter(params)
        params.requires_grad_(requires_grad)
    return params


def _ravel_hw(tensor, fill=0):
    # ravel H,W
    tensor = tensor.view((tensor.shape[0] * tensor.shape[1],) + tensor.shape[2:])

    if len(tensor) < fill:
        tensor = torch.cat((tensor, tensor.new_zeros((fill - len(tensor),)+tensor.shape[1:])))
    return tensor


def acceptable_focal_range(H, W, minf=0.5, maxf=3.5):
    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))  # size / 1.1547005383792515
    return minf*focal_base, maxf*focal_base


def apply_mask(img, msk):
    img = img.copy()
    img[msk] = 0
    return img

def ordered_ratio(disp_a, disp_b, mask=None):
    ratio_a = torch.maximum(disp_a, disp_b) / \
        (torch.minimum(disp_a, disp_b)+1e-5)
    if mask is not None:
        ratio_a = ratio_a[mask]
    return ratio_a - 1