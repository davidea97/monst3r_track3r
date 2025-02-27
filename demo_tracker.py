# --------------------------------------------------------
# gradio demo
# --------------------------------------------------------

import matplotlib
matplotlib.use('Agg')

import argparse
import math
import gradio
import os
import torch
import numpy as np
import tempfile
import functools
import copy
import glob
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb, enlarge_seg_masks, load_masks
from dust3r.utils.device import to_numpy
from dust3r.utils.general_utils import generate_image_list, generate_mask_list, read_intrinsics
from dust3r.utils.file_utils import *
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.viz_demo import convert_scene_output_to_glb, get_dynamic_mask_from_pairviewer
from mast3r.mast3r import forward_mast3r, convert_dust3r_pairs_naming
import shutil
import matplotlib.pyplot as pl
import open3d as o3d
import cv2
import sys
from PIL import Image

sys.path.append(os.path.abspath("Grounded_SAM_2"))
from Grounded_SAM_2.sam2_mask_tracking import MaskGenerator 
from object_track import ObjectTrack

from mast3r.model import AsymmetricMASt3R

pl.ion()
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser_url = parser.add_mutually_exclusive_group()
    parser_url.add_argument("--local_network", action='store_true', default=False,
                            help="make app accessible on local network: address will be set to 0.0.0.0")
    parser_url.add_argument("--server_name", type=str, default=None, help="server url, default is 127.0.0.1")
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    parser.add_argument("--server_port", type=int, help=("will start gradio app on this port (if available). "
                                                         "If None, will search for an available port starting at 7860."),
                        default=None)
    parser.add_argument("--weights", type=str, help="path to the model weights", default='checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth')
    parser.add_argument("--model_name", type=str, default='Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt', help="model name")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--output_dir", type=str, default='./demo_tmp', help="value for tempfile.tempdir")
    parser.add_argument("--silent", action='store_true', default=False,
                        help="silence logs")
    # parser.add_argument("--input_dir", type=str, help="Path to input images directory", default=None)
    parser.add_argument("--seq_name", type=str, help="Sequence name for evaluation", default='NULL')
    parser.add_argument('--use_gt_davis_masks', action='store_true', default=False, help='Use ground truth masks for DAVIS')
    parser.add_argument('--not_batchify', action='store_true', default=False, help='Use non batchify mode for global optimization')
    parser.add_argument('--real_time', action='store_true', default=False, help='Realtime mode')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')

    parser.add_argument('--fps', type=int, default=0, help='FPS for video processing')
    parser.add_argument('--num_frames', type=int, default=200, help='Maximum number of frames for video processing')
    
    # Add "share" argument if you want to make the demo accessible on the public internet
    parser.add_argument("--share", action='store_true', default=False, help="Share the demo")
    return parser

def get_3D_model_from_scene(outdir, silent, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05, show_cam=True, save_name=None, thr_for_init_conf=True, parameters_X=None, object_masks=None):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None
    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    imagelist = scene.imagelist

    if parameters_X[0] is not None:
        cams2world = scene.get_relative_poses(cams2world, scale_factor=parameters_X[0].item())
    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d, object_pts3d = to_numpy(scene.get_pts3d(raw_pts=True)) # Object pts include the 3d points belonging to each object 
    if parameters_X[0] is not None:
        scaled_pts3d = [p * parameters_X[0].item() for p in pts3d]
        scaled_object_pts3d = [p * parameters_X[0].item() for p in object_pts3d]
        pts3d = scaled_pts3d
        object_pts3d = scaled_object_pts3d
    # scene.min_conf_thr = min_conf_thr
    # scene.thr_for_init_conf = thr_for_init_conf
    msk = to_numpy(scene.get_masks())

    cmap = pl.get_cmap('viridis')
    cam_color = [cmap(i/len(rgbimg))[:3] for i in range(len(rgbimg))]
    cam_color = [(255*c[0], 255*c[1], 255*c[2]) for c in cam_color]

    # Add the tracking part
    # object_tracker = ObjectTracker(all_3d_obj_pts=object_pts3d, obj_msks=object_masks, imagelist=rgbimg, pts3d=pts3d)

    object_tracker = ObjectTrack(obj_msks=object_masks, all_3d_obj_pts=object_pts3d, imagelist=rgbimg, pts3d=pts3d)
    all_dino_descriptors, dino_images, mask_coords, all_sp_features, all_3d_features = object_tracker.extract_dino_features()
    all_gedi_descriptors = object_tracker.extract_gedi_features(all_3d_features)

    obj_transformation, obj_obj2world = object_tracker._obj_track(all_dino_descriptors, all_gedi_descriptors=all_gedi_descriptors, all_sp_features=all_sp_features, all_3d_features=all_3d_features)
    for i, cam2w_object in enumerate(obj_obj2world):
        print("Length of object poses: ", len(cam2w_object))
        save_obj_traj = object_tracker.save_obj_poses(f'{outdir}/obj_traj_{i}.txt', obj_obj2world[i])
        
    # print("Length of dino descriptors: ", len(dino_descriptors_all))
    # print("Length descriptors for each object: ", [len(dino_descriptors) for dino_descriptors in dino_descriptors_all])
    # print("Shape dino descriptor 0: ", dino_descriptors_all[0][0].shape)

    # # Get Dino features from images
    # dino_valid_patch_features, valid_3d_pts = object_tracker._get_3d_dino_patch_features()
    
    # valid_3d_pts_t = list(map(list, zip(*valid_3d_pts)))
    # dino_valid_patch_features = list(map(list, zip(*dino_valid_patch_features)))


    # dino_transformation, dino_cam2w = object_tracker._get_dino_obj_track(valid_3d_pts_t, dino_valid_patch_features)
    # for i, dino_cam2w_object in enumerate(dino_cam2w):
    #     save_obj_traj = object_tracker.save_obj_poses(f'{outdir}/obj_traj_{i}.txt', dino_cam2w[i])
    
    glb = convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, show_cam=show_cam, silent=silent, save_name=save_name,
                                        cam_color=cam_color, all_object_pts3d=object_pts3d, all_msk_obj=object_masks, tracking_transformation=obj_obj2world)
    return glb, dino_images[0]


# def draw_matches(img1, img2, corres, save_path, mask1=None, mask2=None):

#     if isinstance(img1, torch.Tensor):
#         img1 = img1.squeeze(0).permute(1, 2, 0).cpu().numpy()
#     if isinstance(img2, torch.Tensor):
#         img2 = img2.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
#     img1 = (img1 - img1.min()) / (img1.max() - img1.min())
#     img2 = (img2 - img2.min()) / (img2.max() - img2.min())

#     img1 = (img1 * 255).astype(np.uint8)
#     img2 = (img2 * 255).astype(np.uint8)

#     h1, w1, _ = img1.shape
#     h2, w2, _ = img2.shape
#     img_match = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
#     img_match[:h1, :w1] = img1
#     img_match[:h2, w1:w1 + w2] = img2

#     xy1, xy2, conf = corres  
#     xy1 = xy1.cpu().numpy().astype(int)
#     xy2 = xy2.cpu().numpy().astype(int)
#     conf = conf.cpu().numpy() 
#     xy2[:, 0] += w1  

#     k = 0
#     for (x1, y1), (x2, y2), score in zip(xy1, xy2, conf):
#         # Random color based on score
#         color = (0, 255, 0) if score > 0.5 else (0, 0, 255) 
#         cv2.line(img_match, (x1, y1), (x2, y2), color, 1)
#         cv2.circle(img_match, (x1, y1), 2, color, -1)
#         cv2.circle(img_match, (x2, y2), 2, color, -1)
#         k += 1

#     img_match = cv2.cvtColor(img_match, cv2.COLOR_RGB2BGR)
#     cv2.imwrite(save_path, img_match)


# def draw_matches(img1, img2, corres, mask1, mask2, save_path=None):
#     """ Draw matches only for pixels inside the provided masks, using random colors. """

#     # Convert tensors to numpy if necessary
#     if isinstance(img1, torch.Tensor):
#         img1 = img1.squeeze(0).permute(1, 2, 0).cpu().numpy()
#     if isinstance(img2, torch.Tensor):
#         img2 = img2.squeeze(0).permute(1, 2, 0).cpu().numpy()

#     # Normalize images
#     img1 = (img1 - img1.min()) / (img1.max() - img1.min())
#     img2 = (img2 - img2.min()) / (img2.max() - img2.min())
#     img1 = (img1 * 255).astype(np.uint8)
#     img2 = (img2 * 255).astype(np.uint8)

#     # Ensure masks are numpy arrays
#     if isinstance(mask1, torch.Tensor):
#         mask1 = mask1.cpu().numpy()
#     if isinstance(mask2, torch.Tensor):
#         mask2 = mask2.cpu().numpy()

#     # Create concatenated image
#     h1, w1, _ = img1.shape
#     h2, w2, _ = img2.shape
#     img_match = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
#     img_match[:h1, :w1] = img1
#     img_match[:h2, w1:w1 + w2] = img2

#     # Extract correspondences
#     xy1, xy2, conf = corres  
#     xy1 = xy1.cpu().numpy().astype(int)
#     xy2 = xy2.cpu().numpy().astype(int)
#     conf = conf.cpu().numpy()
#     xy2[:, 0] += w1  # Adjust x-coordinates of img2

#     # Filter correspondences using masks
#     valid_matches = []
#     for (x1, y1), (x2, y2), score in zip(xy1, xy2, conf):
#         print(f"First pixel: ({x1}, {y1}), Second pixel: ({x2}, {y2})")
#         if mask1[y1, x1] > 0 and mask2[y2, x2 - w1] > 0:  # Check if in mask
#             valid_matches.append(((x1, y1), (x2, y2), score))

#     # Draw correspondences with random colors
#     correspondences = 0
#     for (x1, y1), (x2, y2), score in valid_matches:
        
#         color = tuple(np.random.randint(0, 255, 3).tolist())  # Generate random color
#         cv2.line(img_match, (x1, y1), (x2, y2), color, 1)
#         cv2.circle(img_match, (x1, y1), 2, color, -1)
#         cv2.circle(img_match, (x2, y2), 2, color, -1)
#         correspondences += 1
#     print(f"Number of valid correspondences: {correspondences}")
#     # Convert RGB to BGR for OpenCV saving
#     img_match = cv2.cvtColor(img_match, cv2.COLOR_RGB2BGR)
#     cv2.imwrite(save_path, img_match)
#     print(f"Saved match image: {save_path}")

def get_reconstructed_scene(args, outdir, model, device, silent, image_size, filelist, intrinsic_params, dist_coeffs, mask_list, robot_poses, schedule, niter, min_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, show_cam, scenegraph_type, winsize, refid, 
                            seq_name, new_model_weights, temporal_smoothing_weight, translation_weight, shared_focal, 
                            flow_loss_weight, flow_loss_start_iter, flow_loss_threshold, use_gt_mask, fps, num_frames):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """

    translation_weight = float(translation_weight)
    if new_model_weights != args.weights:
        model = AsymmetricCroCo3DStereo.from_pretrained(new_model_weights).to(device)
    model.eval()
    if seq_name != "NULL":
        dynamic_mask_path = f'data/davis/DAVIS/masked_images/480p/{seq_name}'
    else:
        dynamic_mask_path = None
    imgs = load_images(filelist, size=image_size, verbose=not silent, dynamic_mask_root=dynamic_mask_path, fps=fps, num_frames=num_frames)

    if args.mask:
        msks = load_masks(mask_list, filelist, size=config['image_size'], verbose=not config['silent'])
        msks = msks[0]
    else:
        msks = None

    if args.use_intrinsics:
        intrinsic_params = intrinsic_params
        dist_coeffs = dist_coeffs
        print("Intrinsic parameters: ", intrinsic_params)
        print("Distortion coefficients: ", dist_coeffs)
    else:
        intrinsic_params = None
        dist_coeffs = None
        print("Intrinsic parameters not provided. Estimating intrinsics..")

    if args.use_robot_motion:
        robot_poses = robot_poses

    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1

    print("WINSIZE: ", winsize)
    if scenegraph_type == "swin" or scenegraph_type == "swinstride" or scenegraph_type == "swin2stride":
        scenegraph_type = scenegraph_type + "-" + str(winsize) + "-noncyclic"
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)

    output = inference(pairs, model, device, batch_size=args.batch_size, verbose=not silent)

    # # Add Mast3r inference to extract correspondences
    # pairs_in = convert_dust3r_pairs_naming(filelist, pairs)
    # weights_path_mast3r = 'checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
    # mast3r_model = AsymmetricMASt3R.from_pretrained(weights_path_mast3r).to(args.device)
    # corres, matching_score = forward_mast3r(pairs_in, mast3r_model, msks, device=device)
    # os.makedirs("matches", exist_ok=True)
    # for img1, img2 in pairs_in:
    #     idx1 = img1['idx']
    #     idx2 = img2['idx']
    #     img1 = img1['img'].to(device, non_blocking=True)
    #     img2 = img2['img'].to(device, non_blocking=True)

    #     save_path = f"matches/match_{idx1}_{idx2}.png"        
    #     print(f"Processing pair {idx1} and {idx2}...")
    #     draw_matches(img1, img2, corres, mask1=msks[idx1], mask2=msks[idx2], save_path=save_path) 



    if len(imgs) > 2:
        mode = GlobalAlignerMode.PointCloudOptimizer  
        scene = global_aligner(output, device=device, mode=mode, verbose=not silent, shared_focal = shared_focal, filelist=filelist, intrinsic_params=intrinsic_params, dist_coeffs=dist_coeffs, robot_poses=robot_poses, masks=msks, temporal_smoothing_weight=temporal_smoothing_weight, translation_weight=translation_weight,
                               flow_loss_weight=flow_loss_weight, flow_loss_start_epoch=flow_loss_start_iter, flow_loss_thre=flow_loss_threshold, use_self_mask=not use_gt_mask,
                               num_total_iter=niter, empty_cache= len(filelist) > 72, batchify=not args.not_batchify)
    else:
        mode = GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
    
    lr = 0.02

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

    print('Global alignment done!')

    # PARAMETERS
    scale_factor = scene._get_scale_factor()
    translation_X = scene._get_trans_X()
    rotation_X = scene._get_quat_X()
    parameters_X = [scale_factor, translation_X, rotation_X]
    print("----- PARAMETERS -----")
    if scale_factor is not None:
        scale_factor = torch.abs(scale_factor)
        print(f"Scale factor: {scale_factor}")
        print(f"Scaled translation: {scale_factor * translation_X}")
        print(f"Rotation: {rotation_X}")
        parameters_X = [scale_factor, scale_factor * translation_X, rotation_X]

    save_folder = f'{args.output_dir}/{seq_name}'  #default is 'demo_tmp/NULL'
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder, exist_ok=True)

    outfile, dino_images = get_3D_model_from_scene(save_folder, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
                            clean_depth, transparent_cams, cam_size, show_cam, parameters_X=parameters_X, object_masks=msks)
    
    poses = scene.save_tum_poses(f'{save_folder}/pred_traj.txt')
    K = scene.save_intrinsics(f'{save_folder}/pred_intrinsics.txt')
    depth_maps = scene.save_depth_maps(save_folder)
    dynamic_masks = scene.save_dynamic_masks(save_folder)
    conf = scene.save_conf_maps(save_folder)
    init_conf = scene.save_init_conf_maps(save_folder)
    rgbs = scene.save_rgb_imgs(save_folder)
    enlarge_seg_masks(save_folder, kernel_size=5 if use_gt_mask else 3) 

    # also return rgb, depth and confidence imgs
    # depth is normalized with the max value for all images
    # we apply the jet colormap on the confidence maps
    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    init_confs = to_numpy([c for c in scene.init_conf_maps])
    cmap = pl.get_cmap('jet')
    depths_max = max([d.max() for d in depths])
    depths = [cmap(d/depths_max) for d in depths]
    confs_max = max([d.max() for d in confs])
    confs = [cmap(d/confs_max) for d in confs]
    init_confs_max = max([d.max() for d in init_confs])
    init_confs = [cmap(d/init_confs_max) for d in init_confs]

    imgs = []
    for i in range(len(rgbimg)):
        imgs.append(rgbimg[i])
        imgs.append(rgb(depths[i]))
        dino_image = cv2.cvtColor(dino_images[i], cv2.COLOR_RGB2BGR)
        imgs.append(dino_image)

    # if two images, and the shape is same, we can compute the dynamic mask
    if len(rgbimg) == 2 and rgbimg[0].shape == rgbimg[1].shape:
        motion_mask_thre = 0.35 # Default 0.35
        error_map = get_dynamic_mask_from_pairviewer(scene, both_directions=True, output_dir=args.output_dir, motion_mask_thre=motion_mask_thre)
        normalized_error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min())
        error_map_max = normalized_error_map.max()
        error_map = cmap(normalized_error_map/error_map_max)
        imgs.append(rgb(error_map))
        binary_error_map = (normalized_error_map > motion_mask_thre).astype(np.uint8)
        imgs.append(rgb(binary_error_map*255))

    return scene, outfile, imgs


def set_scenegraph_options(inputfiles, winsize, refid, scenegraph_type):
    # if inputfiles[0] is a video, set the num_files to 200
    if inputfiles is not None and len(inputfiles) == 1 and inputfiles[0].name.endswith(('.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV')):
        num_files = 200
    else:
        num_files = len(inputfiles) if inputfiles is not None else 1
    max_winsize = max(1, math.ceil((num_files-1)/2))
    
    if scenegraph_type == "swin" or scenegraph_type == "swin2stride" or scenegraph_type == "swinstride":
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=min(max_winsize,15),
                                minimum=1, maximum=max_winsize, step=1, visible=True)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files-1, step=1, visible=False)
    elif scenegraph_type == "oneref":
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=max_winsize,
                                minimum=1, maximum=max_winsize, step=1, visible=False)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files-1, step=1, visible=True)
    else:
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=max_winsize,
                                minimum=1, maximum=max_winsize, step=1, visible=False)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files-1, step=1, visible=False)
    return winsize, refid


def get_reconstructed_scene_realtime(args, model, device, silent, image_size, filelist, scenegraph_type, refid, seq_name, fps, num_frames):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    model.eval()
    imgs = load_images(filelist, size=image_size, verbose=not silent, fps=fps, num_frames=num_frames)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    
    if scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)
    elif scenegraph_type == "oneref_mid":
        scenegraph_type = "oneref-" + str(len(imgs) // 2)
    else:
        raise ValueError(f"Unknown scenegraph type for realtime mode: {scenegraph_type}")
    
    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=False)
    output = inference(pairs, model, device, batch_size=args.batch_size, verbose=not silent)

    save_folder = f'{args.output_dir}/{seq_name}'  #default is 'demo_tmp/NULL'
    os.makedirs(save_folder, exist_ok=True)

    view1, view2, pred1, pred2 = output['view1'], output['view2'], output['pred1'], output['pred2']
    pts1 = pred1['pts3d'].detach().cpu().numpy()
    pts2 = pred2['pts3d_in_other_view'].detach().cpu().numpy()
    for batch_idx in range(len(view1['img'])):
        colors1 = rgb(view1['img'][batch_idx])
        colors2 = rgb(view2['img'][batch_idx])
        xyzrgb1 = np.concatenate([pts1[batch_idx], colors1], axis=-1)   #(H, W, 6)
        xyzrgb2 = np.concatenate([pts2[batch_idx], colors2], axis=-1)
        np.save(save_folder + '/pts3d1_p' + str(batch_idx) + '.npy', xyzrgb1)
        np.save(save_folder + '/pts3d2_p' + str(batch_idx) + '.npy', xyzrgb2)

        conf1 = pred1['conf'][batch_idx].detach().cpu().numpy()
        conf2 = pred2['conf'][batch_idx].detach().cpu().numpy()
        np.save(save_folder + '/conf1_p' + str(batch_idx) + '.npy', conf1)
        np.save(save_folder + '/conf2_p' + str(batch_idx) + '.npy', conf2)

        # save the imgs of two views
        img1_rgb = cv2.cvtColor(colors1 * 255, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(colors2 * 255, cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_folder + '/img1_p' + str(batch_idx) + '.png', img1_rgb)
        cv2.imwrite(save_folder + '/img2_p' + str(batch_idx) + '.png', img2_rgb)

    return save_folder


def main_demo(tmpdirname, model, device, image_size, server_name, server_port, silent=False, args=None, input_files=None, intrinsic_params=None, dist_coeffs=None, mask_list=None, robot_poses=None):
    recon_fun = functools.partial(get_reconstructed_scene, args, tmpdirname, model, device, silent, image_size)
    model_from_scene_fun = functools.partial(get_3D_model_from_scene, tmpdirname, silent)
    
    # Convert input_files to a format suitable for display
    input_file_paths = [file for file in input_files]
    formatted_list = "\n".join(input_file_paths)
    print("Formatted list: ", formatted_list)
    with gradio.Blocks(css=""".gradio-container {margin: 0 !important; min-width: 100%};""", title="MonST3R Track Demo") as demo:
        # scene state is save so that you can change conf_thr, cam_size... without rerunning the inference
        scene = gradio.State(None)
        gradio.HTML(f'<h2 style="text-align: center;">MonST3R Track3r Demo</h2>')
        with gradio.Column():

            # Placeholder for dynamically updated file input (disabled)
            inputfiles = gradio.File(
                file_count="multiple",
                label="Files to Process",
                interactive=False,
                value=input_file_paths
            )

            intrinsic_state = gradio.State(intrinsic_params)
            dist_coeff_state = gradio.State(dist_coeffs)
            robot_pose_state = gradio.State(robot_poses)
            masks_state = gradio.State(mask_list)
       
            with gradio.Row():
                schedule = gradio.Dropdown(["linear", "cosine"],
                                           value='linear', label="schedule", info="For global alignment!")
                niter = gradio.Number(value=500, precision=0, minimum=0, maximum=5000,
                                      label="num_iterations", info="For global alignment!")
                seq_name = gradio.Textbox(label="Sequence Name", placeholder="NULL", value=args.seq_name, info="For evaluation")
                scenegraph_type = gradio.Dropdown(["complete", "swin", "oneref", "swinstride", "swin2stride"],
                                                  value='swinstride', label="Scenegraph",
                                                  info="Define how to make pairs",
                                                  interactive=True)
                winsize = gradio.Slider(label="Scene Graph: Window Size", value=5,
                                        minimum=1, maximum=1, step=1, visible=False)
                refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0, maximum=0, step=1, visible=False)

            run_btn = gradio.Button("Run")

            with gradio.Row():
                # adjust the confidence thresholdx
                min_conf_thr = gradio.Slider(label="min_conf_thr", value=1.1, minimum=0.0, maximum=20, step=0.01)
                # adjust the camera size in the output pointcloud
                cam_size = gradio.Slider(label="cam_size", value=0.05, minimum=0.001, maximum=0.1, step=0.001)
                # adjust the temporal smoothing weight
                temporal_smoothing_weight = gradio.Slider(label="temporal_smoothing_weight", value=0.01, minimum=0.0, maximum=0.1, step=0.001)
                # add translation weight
                translation_weight = gradio.Textbox(label="translation_weight", placeholder="1.0", value="1.0", info="For evaluation")
                # change to another model
                new_model_weights = gradio.Textbox(label="New Model", placeholder=args.weights, value=args.weights, info="Path to updated model weights")
            with gradio.Row():
                as_pointcloud = gradio.Checkbox(value=True, label="As pointcloud")
                # two post process implemented
                mask_sky = gradio.Checkbox(value=False, label="Mask sky")
                clean_depth = gradio.Checkbox(value=True, label="Clean-up depthmaps")
                transparent_cams = gradio.Checkbox(value=False, label="Transparent cameras")
                # not to show camera
                show_cam = gradio.Checkbox(value=True, label="Show Camera")
                shared_focal = gradio.Checkbox(value=True, label="Shared Focal Length")
                use_davis_gt_mask = gradio.Checkbox(value=False, label="Use GT Mask (DAVIS)")
            with gradio.Row():
                flow_loss_weight = gradio.Slider(label="Flow Loss Weight", value=0.01, minimum=0.0, maximum=1.0, step=0.001)
                flow_loss_start_iter = gradio.Slider(label="Flow Loss Start Iter", value=0.1, minimum=0, maximum=1, step=0.01)
                flow_loss_threshold = gradio.Slider(label="Flow Loss Threshold", value=25, minimum=0, maximum=100, step=1)
                # for video processing
                fps = gradio.Slider(label="FPS", value=0, minimum=0, maximum=60, step=1)
                num_frames = gradio.Slider(label="Num Frames", value=100, minimum=0, maximum=200, step=1)

            outmodel = gradio.Model3D()
            outgallery = gradio.Gallery(label='rgb,depth,dino', columns=3, height="100%")

            # events
            scenegraph_type.change(set_scenegraph_options,
                                   inputs=[inputfiles, winsize, refid, scenegraph_type],
                                   outputs=[winsize, refid])
            inputfiles.change(set_scenegraph_options,
                              inputs=[inputfiles, winsize, refid, scenegraph_type],
                              outputs=[winsize, refid])
            run_btn.click(fn=recon_fun,
                          inputs=[inputfiles, intrinsic_state, dist_coeff_state, masks_state, robot_pose_state,
                                  schedule, niter, min_conf_thr, as_pointcloud,
                                  mask_sky, clean_depth, transparent_cams, cam_size, show_cam,
                                  scenegraph_type, winsize, refid, seq_name, new_model_weights, 
                                  temporal_smoothing_weight, translation_weight, shared_focal, 
                                  flow_loss_weight, flow_loss_start_iter, flow_loss_threshold, use_davis_gt_mask,
                                  fps, num_frames],
                          outputs=[scene, outmodel, outgallery])
            min_conf_thr.release(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                         clean_depth, transparent_cams, cam_size, show_cam],
                                 outputs=outmodel)
            cam_size.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size, show_cam],
                            outputs=outmodel)
            as_pointcloud.change(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                         clean_depth, transparent_cams, cam_size, show_cam],
                                 outputs=outmodel)
            mask_sky.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size, show_cam],
                            outputs=outmodel)
            clean_depth.change(fn=model_from_scene_fun,
                               inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                       clean_depth, transparent_cams, cam_size, show_cam],
                               outputs=outmodel)
            transparent_cams.change(model_from_scene_fun,
                                    inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                            clean_depth, transparent_cams, cam_size, show_cam],
                                    outputs=outmodel)
    demo.launch(share=args.share, server_name=server_name, server_port=server_port)


if __name__ == '__main__':
    parser = get_args_parser()
    parser.add_argument('--input_folder', type=str, default="dust3r/croco/assets")
    # parser.add_argument('--outdir', type=str, default="output")
    parser.add_argument('--config', type=str, default="config.yaml", help="Path to the configuration file")
    parser.add_argument('--mask', type=str2bool, default=False, help="True or False for mask generation")
    parser.add_argument('--subset_size', type=int, default=0, help="Number of images to use for the reconstruction")
    parser.add_argument('--use_intrinsics', type=str2bool, default=False, help="Use intrinsic parameters for the cameras")
    parser.add_argument('--use_robot_motion', type=str2bool, default=False, help="Use robot motion to improve camera poses and perform the calibration step")

    args = parser.parse_args()

    if args.output_dir is not None:
        tmp_path = args.output_dir
        os.makedirs(tmp_path, exist_ok=True)
        tempfile.tempdir = tmp_path

    if args.server_name is not None:
        server_name = args.server_name
    else:
        server_name = '0.0.0.0' if args.local_network else '127.0.0.1'

    if args.weights is not None and os.path.exists(args.weights):
        weights_path = args.weights
    else:
        weights_path = args.model_name

    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(args.device)

    # Use the provided output_dir or create a temporary directory
    tmpdirname = args.output_dir if args.output_dir is not None else tempfile.mkdtemp(suffix='monst3r_gradio_demo')

    if not args.silent:
        print('Outputting stuff in', tmpdirname)

    config = load_config(args.config)

    image_list, subfolders = generate_image_list(args.input_folder)
    robot_poses = [[] for _ in range(len(subfolders))]
    image_sublist = [[] for _ in range(len(subfolders))]

    if args.use_robot_motion:
        print("Using robot motion to improve camera poses and perform the calibration step")
        for i, subfolder in enumerate(subfolders):
            yaml_files = sorted(glob.glob(f"{subfolder}/relative_rob_poses/*.yaml"))
            for yaml_file in yaml_files:

                # Read matrix from YAML file using OpenCV
                fs = cv2.FileStorage(yaml_file, cv2.FILE_STORAGE_READ)
                matrix = fs.getNode("matrix").mat()

                # Scale translation of the matrix
                scale = 1.0
                # scale = 0.84
                matrix[:3, 3] = matrix[:3, 3] * scale

                fs.release()
                robot_poses[i].append(torch.tensor(matrix))
    else:
        robot_poses = None


    if args.subset_size > 0:
        for i in range(len(image_list)):
            image_sublist[i] = image_list[i][:args.subset_size]
        if robot_poses is not None:
            for i in range(len(robot_poses)):
                robot_poses[i] = robot_poses[i][:args.subset_size-1]
    else:
        image_sublist = image_list

    image_ext = None
    mask_list = None
    if args.mask:
        mask_generator = MaskGenerator(config, image_sublist, subfolders)
        print("Generating masks...")
        objects, image_ext = mask_generator.generate_masks()
        print("Objects: ", objects)
        mask_list = generate_mask_list(args.input_folder, image_sublist, image_ext=image_ext)

    intrinsic_params_vec = []
    dist_coeffs = []
    if args.use_intrinsics:
        intrinsic_params_vec, dist_coeffs = read_intrinsics(subfolders, config)
        print("Intrinsics: ", intrinsic_params_vec)
        print("Distortion coefficients: ", dist_coeffs)
    else:
        for i in range(len(subfolders)):
            intrinsic_params_vec.append(None)
            dist_coeffs.append(None)
            print("Intrinsic parameters not provided. Estimating intrinsics..")

    # TODO: handle multiple subfolders
    camera_selected = 0
    input_dir = subfolders[camera_selected]
    image_list = image_list[camera_selected]
    image_sublist = image_sublist[camera_selected]
    intrinsic_params = intrinsic_params_vec[camera_selected]
    dist_coeff = dist_coeffs[camera_selected]

    if args.mask:
        mask_list = mask_list[camera_selected]
    if args.use_robot_motion:
        robot_pose = robot_poses[camera_selected]
    else:
        robot_pose = None


    num_frames = len(image_sublist)
    input_files = image_sublist

    main_demo(tmpdirname, model, args.device, args.image_size, server_name, args.server_port, silent=args.silent, args=args, 
                  input_files=input_files, intrinsic_params=intrinsic_params, dist_coeffs=dist_coeff, mask_list=mask_list, robot_poses=robot_pose)
