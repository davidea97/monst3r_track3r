import torch
import numpy as np
import open3d as o3d
from gedi.gedi import GeDi
from dust3r.utils.vo_eval import save_trajectory_tum_format
from dust3r.utils.general_utils import visualize_matches
from dust3r.cloud_opt.base_opt import c2w_to_tumpose
from transformers import AutoProcessor, AutoModel
from PIL import Image
import dino_extract
import superpoint_extract
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os
from sklearn.preprocessing import normalize

DINO_FEATURE_DIM_B = 768
DINO_FEATURE_DIM_L = 1024
MATCH_THRESHOLD = 1e-3

class ObjectTrack:
    def __init__(self, obj_msks, all_3d_obj_pts=None, imagelist=None, pts3d=None):
        self.all_3d_obj_pts = all_3d_obj_pts
        self.obj_msks = obj_msks
        self.imagelist = imagelist
        self.pts3d = pts3d
        dino_export="./models/dinov2_vitl14_pretrain.pth"
        sp_export="./models/sp_v6"
        og_export="./models/og_export"

        self.dino_extract = dino_extract.DINOExtract(dino_export, feature_layer=1)
        self.sp_extract = superpoint_extract.SuperPointExtract(sp_export)
        self.matcher = tf.saved_model.load(og_export)
        config = {'dim': 32,                                                # descriptor output dimension
                'samples_per_batch': 500,                                   # batches to process the data on GPU
                'samples_per_patch_lrf': 4000,                              # num. of point to process with LRF
                'samples_per_patch_out': 512,                               # num. of points to sample for pointnet++
                'r_lrf': .5,                                                # LRF radius
                'fchkpt_gedi_net': 'gedi/data/chkpts/3dmatch/chkpt.tar'}    # path to checkpoint
        
        self.voxel_size = .01
        self.patches_per_pair = 3000
 
        # initialising class
        self.gedi = GeDi(config=config)

    def _get_object_quantity(self):
        return len(self.all_3d_obj_pts)
    
    def _get_all_3d_object_pts(self):
        return self.all_3d_obj_pts
    
    def _get_all_object_masks(self):
        return self.obj_msks
    
    def _get_obj_poses(self, obj2w):
        poses = obj2w
        tt = np.arange(len(poses)).astype(float)
        tum_poses = [c2w_to_tumpose(p) for p in poses]
        tum_poses = np.stack(tum_poses, 0)
        return [tum_poses, tt]
    
    def save_obj_poses(self, path, obj2w):
        traj = self._get_obj_poses(obj2w)
        save_trajectory_tum_format(traj, path)
        return traj[0] # return the poses
    

    def sample_mask_on_grid_center(self, mask, dino_feature_shape):
        """
        Samples the pixel closest to the center of each grid cell corresponding to the DINO feature map.

        Args:
            mask (np.ndarray): Binary mask (H, W) with object regions.
            dino_feature_shape (tuple): Shape of the DINO feature map (H', W', C).

        Returns:
            np.ndarray: Array of sampled coordinates (N, 2) in (y, x) format.
        """
        H, W = mask.shape
        H_feat, W_feat = dino_feature_shape[:2]  # Feature map dimensions from DINO

        # Compute the size of each grid cell
        cell_h = H // (H_feat)
        cell_w = W // (W_feat)

        sampled_coords = []

        for i in range(H_feat):
            for j in range(W_feat):
                # Define the boundaries of the grid cell in the original mask
                y_start, y_end = i * cell_h, min((i + 1) * cell_h, H)
                x_start, x_end = j * cell_w, min((j + 1) * cell_w, W)

                # Extract the mask portion for the current cell
                cell_mask = mask[y_start:y_end, x_start:x_end]

                # Find pixels within the mask
                mask_pixels = np.argwhere(cell_mask > 0)

                if len(mask_pixels) > 0:
                    # Compute the center of the cell
                    center_y, center_x = (y_end + y_start) // 2, (x_end + x_start) // 2

                    # Find the pixel closest to the center
                    distances = np.linalg.norm(mask_pixels - np.array([(center_y - y_start), (center_x - x_start)]), axis=1)
                    closest_idx = np.argmin(distances)
                    y_offset, x_offset = mask_pixels[closest_idx]

                    sampled_coords.append((y_start + y_offset, x_start + x_offset))

        return np.array(sampled_coords)


    def create_dino_video(self, all_images): 
        output_video_path = "output_video.mp4"
        fps = 5 

        images = all_images

        frame = images[0]
        h, w, _ = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
        
        for image in images:
            video_writer.write(image)

        video_writer.release()
        print(f"Video saved as {output_video_path}")

        return output_video_path


    def dino_image_visualization(self, all_dino_descriptors, imagelist, masks, all_sp_features, all_3d_sp_features):

        all_dino_images = []

        for j in range(self._get_object_quantity()):
            all_dino_descriptors_vis = np.concatenate(all_dino_descriptors[j], axis=0)

            pca = PCA(n_components=3)
            pca.fit(all_dino_descriptors_vis)
            dino_images = []
            os.makedirs("./output_images_DINO", exist_ok=True)

            for i in range(len(imagelist)):
                image = imagelist[i]
                output_image = image.copy()

                mask_coords = np.argwhere(masks[i] == j+1)
                mask_coords = mask_coords[:, [1, 0]]
                sp_features = all_sp_features[j][i][0]
                sp_3d_features = all_3d_sp_features[j][i]
                dino_descriptors = all_dino_descriptors[j][i]

                dino_descriptors_np = dino_descriptors.numpy()

                # Apply the SAME PCA transformation
                pca_features = pca.transform(dino_descriptors_np)  # Shape (N, 3)

                # Normalize PCA output to [0, 255]
                pca_features = (pca_features - pca_features.min(axis=0)) / (pca_features.max(axis=0) - pca_features.min(axis=0) + 1e-6)
                pca_features = (pca_features * 255).astype(np.uint8)


                # Draw PCA-colored points
                for k, (x, y) in enumerate(sp_features):
                    x, y = int(x), int(y)
                    color = tuple(map(int, pca_features[k]))  
                    cv2.circle(output_image, (x, y), radius=1, color=color, thickness=-1, lineType=cv2.LINE_AA)

                # Draw the 3d points with open3d
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(sp_3d_features)
                pca_features_rgb = pca_features[:, [2, 1, 0]]  # Swap Red and Blue
                pcd.colors = o3d.utility.Vector3dVector(pca_features_rgb.astype(np.float64)/255.0)
                # o3d.visualization.draw_geometries([pcd])
                # Convert from BGR to RGB
                # output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                if output_image.dtype != np.uint8:
                    output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min() + 1e-6)  # Normalizza tra 0 e 1
                    output_image = (output_image * 255).astype(np.uint8)
                    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

                dino_images.append(output_image)
                # dino_image = (dino_images[i] * 255).astype(np.uint8)
                cv2.imwrite(f"output_images_DINO/output_image_{j}_{i}.png", output_image)

            all_dino_images.append(dino_images)
            video = self.create_dino_video(dino_images)

        return all_dino_images

    def extract_dino_features(self):

        all_dino_descriptors = []
        all_sp_features = []
        all_3d_sp_features = []
        print(">> Extracting DINO features...")
        for i in tqdm(range(len(self.imagelist))):
            scene_3d = self.pts3d[i]
            image = self.imagelist[i]
            height, width = image.shape[:2]
            dino_descriptors_objs = []
            sp_features_objs = []
            sp_3d_features_objs = []

            # Extract DINO features for each object in the image    
            for j in range(self._get_object_quantity()):  # Loop over all objects in the image
                image =(image * 255).round().astype(np.uint8)
                dino_features = self.dino_extract(image)

                mask = (self.obj_msks[i] == j+1).astype(np.uint8)
                sp_features = self.sp_extract(image, mask)
                sp_features_objs.append(sp_features)
                mask_coords = np.argwhere(self.obj_msks[i] == j+1)
                mask_coords = mask_coords[:, [1, 0]]                

                dino_descriptors = dino_extract.get_dino_descriptors(
                    dino_features,
                    tf.convert_to_tensor(sp_features[0], dtype=tf.float32),
                    tf.convert_to_tensor(height, dtype=tf.int32),
                    tf.convert_to_tensor(width, dtype=tf.int32),
                    DINO_FEATURE_DIM_L,
                )

                dino_descriptors_objs.append(dino_descriptors)
                indices = sp_features[0]

                indices = np.array(indices)  # Ensure it's a NumPy array
                indices = indices.astype(int)  # Convert all values to integers

                y_indices, x_indices = indices[:, 1], indices[:, 0]
                extracted_3d_points = scene_3d[(y_indices, x_indices)]

                sp_3d_features_objs.append(extracted_3d_points)
            all_dino_descriptors.append(dino_descriptors_objs)
            all_sp_features.append(sp_features_objs)
            all_3d_sp_features.append(sp_3d_features_objs)
        all_dino_descriptors = list(map(list, zip(*all_dino_descriptors)))
        all_sp_features = list(map(list, zip(*all_sp_features)))
        all_3d_sp_features = list(map(list, zip(*all_3d_sp_features)))
        dino_images = self.dino_image_visualization(all_dino_descriptors, self.imagelist, self.obj_msks, all_sp_features, all_3d_sp_features)

        return all_dino_descriptors, dino_images, mask_coords, all_sp_features, all_3d_sp_features

    def extract_gedi_features(self, all_3d_features):
        
        all_gedi_descriptors = []
        for i, obj in enumerate(self._get_all_3d_object_pts()):
            
            gedi_desc_objs = []
            print(">> Extracting GeDi features for object ", i)
            for j in tqdm(range(0, len(obj))):
                pcd0_pts = np.vstack(obj[j])  # Object pts for scene j-1 (Nx3)
                pts_initial = np.vstack(all_3d_features[i][j])
                
                # Convert to Open3D PointCloud
                pcd0 = o3d.geometry.PointCloud()
                pcd0.points = o3d.utility.Vector3dVector(pcd0_pts)

                # o3d.visualization.draw_geometries([pcd0])
                # Estimate normals (only for visualization)
                pcd0.estimate_normals()
                pts0 = torch.tensor(np.asarray(pts_initial)).float()

                _pcd0 = torch.tensor(np.asarray(pcd0.points)).float()

                # computing descriptors
                # gedi_desc = self.gedi.compute(pts=pts0, pcd=_pcd0)
                gedi_desc = self.gedi.compute_multi_scale(pts=pts0, pcd=_pcd0)

                gedi_desc = tf.convert_to_tensor(
                    np.array(gedi_desc), dtype=tf.float32
                )

                gedi_desc_objs.append(gedi_desc)
            all_gedi_descriptors.append(gedi_desc_objs)

        # self.gedi_visualization(all_gedi_descriptors, all_3d_sp_features)

        return all_gedi_descriptors


    def gedi_visualization(self, all_gedi_descriptors, all_3d_sp_features):

        for j in range(self._get_object_quantity()):
            all_gedi_descriptors_vis = np.concatenate(all_gedi_descriptors[j], axis=0)

            pca = PCA(n_components=3)
            pca.fit(all_gedi_descriptors_vis)

            for i in range(len(all_gedi_descriptors[j])):

                gedi_descriptors = all_gedi_descriptors[j][i]

                # gedi_descriptors_np = gedi_descriptors.numpy()

                # Apply the SAME PCA transformation
                pca_features = pca.transform(gedi_descriptors)  # Shape (N, 3)

                # Normalize PCA output to [0, 255]
                pca_features = (pca_features - pca_features.min(axis=0)) / (pca_features.max(axis=0) - pca_features.min(axis=0) + 1e-6)
                pca_features = (pca_features * 255).astype(np.uint8)
                sp_3d_features = all_3d_sp_features[j][i]
                # Draw the 3d points with open3d
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(sp_3d_features)
                pca_features_rgb = pca_features[:, [2, 1, 0]]  # Swap Red and Blue
                pcd.colors = o3d.utility.Vector3dVector(pca_features_rgb.astype(np.float64)/255.0)
                o3d.visualization.draw_geometries([pcd])    

    def create_first_frame(self, obj):
        pcd0_pts = np.vstack(obj[0])  # First point cloud
        obj_centroid = np.mean(pcd0_pts, axis=0)

        # Create identity transformation with translation to centroid
        identity_transform = np.eye(4)
        identity_transform[:3, 3] = obj_centroid  # Set translation
        return identity_transform
    

    def reduce_dino_descriptors(self, dino_desc, output_dim=64):
        """
        Apply PCA to reduce DINOv2 feature vectors from 1536 to output_dim (e.g., 64).
        
        Args:
            dino_features (numpy.ndarray): Input DINO features of shape (N, 1536)
            output_dim (int): Target feature dimension after PCA

        Returns:
            numpy.ndarray: Reduced features of shape (N, output_dim)
        """
        
        # Initialize PCA
        pca = PCA(n_components=output_dim)
        
        # Fit PCA on the DINO feature set
        reduced_features = pca.fit_transform(dino_desc)

        print(f"PCA Variance Ratio: {sum(pca.explained_variance_ratio_):.4f}")  # Check explained variance

        return reduced_features

    
    def _construct_inputs(
        self,
        width0,
        height0,
        width1,
        height1,
        sp_features0,
        sp_features1,
        dino_descriptors0,
        dino_descriptors1,
    ):
        inputs = {
            'keypoints0': tf.convert_to_tensor(
                np.expand_dims(sp_features0[0], axis=0),
                dtype=tf.float32,
            ),
            'keypoints1': tf.convert_to_tensor(
                np.expand_dims(sp_features1[0], axis=0), dtype=tf.float32
            ),
            'descriptors0': tf.convert_to_tensor(
                np.expand_dims(sp_features0[1], axis=0), dtype=tf.float32
            ),
            'descriptors1': tf.convert_to_tensor(
                np.expand_dims(sp_features1[1], axis=0), dtype=tf.float32
            ),
            'scores0': tf.convert_to_tensor(
                np.expand_dims(np.expand_dims(sp_features0[2], axis=0), axis=-1),
                dtype=tf.float32,
            ),
            'scores1': tf.convert_to_tensor(
                np.expand_dims(np.expand_dims(sp_features1[2], axis=0), axis=-1),
                dtype=tf.float32,
            ),
            'descriptors0_dino': tf.expand_dims(dino_descriptors0, axis=0),
            'descriptors1_dino': tf.expand_dims(dino_descriptors1, axis=0),
            'width0': tf.convert_to_tensor(
                np.expand_dims(width0, axis=0), dtype=tf.int32
            ),
            'width1': tf.convert_to_tensor(
                np.expand_dims(width1, axis=0), dtype=tf.int32
            ),
            'height0': tf.convert_to_tensor(
                np.expand_dims(height0, axis=0), dtype=tf.int32
            ),
            'height1': tf.convert_to_tensor(
                np.expand_dims(height1, axis=0), dtype=tf.int32
            ),
        }
        return inputs

    def soft_assignment_to_match_matrix(
        self, soft_assignment: tf.Tensor, match_threshold: float
    ) -> tf.Tensor:
        """Converts a matrix of soft assignment values to binary yes/no match matrix.

        Searches soft_assignment for row- and column-maximum values, which indicate
        mutual nearest neighbor matches between two unique sets of keypoints. Also,
        ensures that score values for matches are above the specified threshold.

        Args:
            soft_assignment: (B, N, M) tensor, contains matching likelihood value
            between features of different sets. N is number of features in image0, and
            M is number of features in image1. Higher value indicates more likely to
            match.
            match_threshold: float, thresholding value to consider a match valid.

        Returns:
            (B, N, M) tensor of binary values. A value of 1 at index (x, y) indicates
            a match between index 'x' (out of N) in image0 and index 'y' (out of M) in
            image 1.
        """

        def _range_like(x, dim):
            """Returns tensor with values (0, 1, 2, ..., N) for dimension in input x."""
            return tf.range(tf.shape(x)[dim], dtype=x.dtype)

        # TODO(omniglue): batch loop & SparseTensor are slow. Optimize with tf ops.
        matches = tf.TensorArray(tf.float32, size=tf.shape(soft_assignment)[0])
        for i in range(tf.shape(soft_assignment)[0]):
            # Iterate through batch and process one example at a time.
            scores = tf.expand_dims(soft_assignment[i, :], 0)  # Shape: (1, N, M).

            # Find indices for max values per row and per column.
            max0 = tf.math.reduce_max(scores, axis=2)  # Shape: (1, N).
            indices0 = tf.math.argmax(scores, axis=2)  # Shape: (1, N).
            indices1 = tf.math.argmax(scores, axis=1)  # Shape: (1, M).

            # Find matches from mutual argmax indices of each set of keypoints.
            mutual = tf.expand_dims(_range_like(indices0, 1), 0) == tf.gather(
                indices1, indices0, axis=1
            )

            # Create match matrix from sets of index pairs and values.
            kp_ind_pairs = tf.stack(
                [_range_like(indices0, 1), tf.squeeze(indices0)], axis=1
            )
            mutual_max0 = tf.squeeze(tf.squeeze(tf.where(mutual, max0, 0), 0))
            sparse = tf.sparse.SparseTensor(
                kp_ind_pairs, mutual_max0, tf.shape(scores, out_type=tf.int64)[1:]
            )
            match_matrix = tf.sparse.to_dense(sparse)
            matches = matches.write(i, match_matrix)

        # Threshold on match_threshold value and convert to binary (0, 1) values.
        match_matrix = matches.stack()
        match_matrix = match_matrix > match_threshold
        return match_matrix

    def _get_matches(self, dino_desc0, dino_desc1, gedi_desc0, gedi_desc1, sp_features0, sp_features1, image0, image1, feature_3d_0, feature_3d_1):
        
        ################## OMNIGLUE PROCESS ######################
        width0, height0 = image0.shape[1], image0.shape[0]
        width1, height1 = image1.shape[1], image1.shape[0]

        # L2 Normalization (Row-wise)
        # dino_desc0 = normalize(dino_desc0, axis=1, norm='l2')  # Shape remains (57, 1024)
        # gedi_desc0 = normalize(gedi_desc0, axis=1, norm='l2')  # Shape remains (57, 96)
        # dino_desc1 = normalize(dino_desc1, axis=1, norm='l2')
        # gedi_desc1 = normalize(gedi_desc1, axis=1, norm='l2')

        # Stack horizontally
        combined_desc0 = np.hstack((dino_desc0, gedi_desc0))  
        combined_desc1 = np.hstack((dino_desc1, gedi_desc1))  
        
        combined_desc0 = tf.convert_to_tensor(
            np.array(combined_desc0), dtype=tf.float32
        )
        combined_desc1 = tf.convert_to_tensor(
            np.array(combined_desc1), dtype=tf.float32
        )

        inputs = self._construct_inputs(
                    width0,
                    height0,
                    width1,
                    height1,
                    sp_features0,
                    sp_features1,
                    combined_desc0,
                    combined_desc1,
                )

        og_outputs = self.matcher.signatures['serving_default'](**inputs)
        soft_assignment = og_outputs['soft_assignment'][:, :-1, :-1]

        match_matrix = (
            self.soft_assignment_to_match_matrix(soft_assignment, MATCH_THRESHOLD)
            .numpy()
            .squeeze()
        )

        # Filter out any matches with 0.0 confidence keypoints.
        match_indices = np.argwhere(match_matrix)
        keep = []
        for k in range(match_indices.shape[0]):
            match = match_indices[k, :]
            if (sp_features0[2][match[0]] > 0.0) and (
                sp_features1[2][match[1]] > 0.0
            ):
                keep.append(k)
        match_indices = match_indices[keep]

        # Format matches in terms of keypoint locations.
        match_kp0s = []
        match_kp1s = []
        pts3d_0 = []
        pts3d_1 = [] 
        match_confidences = []
        for match in match_indices:
            match_kp0s.append(sp_features0[0][match[0], :])
            match_kp1s.append(sp_features1[0][match[1], :])
            match_confidences.append(soft_assignment[0, match[0], match[1]])
            pts3d_0.append(feature_3d_0[match[0]])
            pts3d_1.append(feature_3d_1[match[1]])
        match_kp0 = np.array(match_kp0s)
        match_kp1 = np.array(match_kp1s)
        match_confidences = np.array(match_confidences)

        num_matches = match_kp0.shape[0]
        print(f"> \tFound {num_matches} matches.")

        # Filter by confidence (0.02).
        match_threshold = 0.02  # Choose any value [0.0, 1.0).
        keep_idx = []
        for k in range(match_kp0.shape[0]):
            if match_confidences[k] > match_threshold:
                keep_idx.append(k)
        num_filtered_matches = len(keep_idx)

        match_kp0 = match_kp0[keep_idx]
        match_kp1 = match_kp1[keep_idx]
        pts3d_0 = np.array(pts3d_0)
        pts3d_1 = np.array(pts3d_1)
        pts3d_0 = pts3d_0[keep_idx]
        pts3d_1 = pts3d_1[keep_idx]

        match_confidences = match_confidences[keep_idx]
        print(f"> \tFound {num_filtered_matches}/{num_matches} above threshold {match_threshold}")

        # Visualize.
        # print("> Visualizing matches...")

        viz = visualize_matches(
            image0,
            image1,
            match_kp0,
            match_kp1,
            np.eye(num_filtered_matches),
            show_keypoints=True,
            highlight_unmatched=True,
            title=f"{num_filtered_matches} matches",
            line_width=2,
        )

        return pts3d_0, pts3d_1, viz
        

    def _obj_track(self, all_dino_descriptors, all_gedi_descriptors=None, all_sp_features=None, all_3d_features=None):
        print(">> Starting GeDi & Dino tracking...")
        gedi_transformation = [[] for _ in range(self._get_object_quantity())]
        optimized_relative_transforms = [[] for _ in range(self._get_object_quantity())]
        gedi_obj2world = [[] for _ in range(self._get_object_quantity())]
        optimized_absolute_transforms = [[] for _ in range(self._get_object_quantity())]
        for i, obj in enumerate(self._get_all_3d_object_pts()):
            print(">> Tracking object ", i)
            # Compute centroid of the first object
            identity_transform = self.create_first_frame(obj) # T_{Obj0}^{Cam}

            # Insert identity transformation at the start
            gedi_transformation[i].append(identity_transform)
            optimized_relative_transforms[i].append(identity_transform)

            # TODO: add the first frame initializations, for example PCA, etc..
            gedi_obj2world[i].append(identity_transform)
            current_cam2world = identity_transform

            # all_dino_descriptors_vis = np.concatenate(all_dino_descriptors[i], axis=0)
            # pca = PCA(n_components=3)
            # pca.fit(all_dino_descriptors_vis)
            
            pose_graph = o3d.pipelines.registration.PoseGraph()
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(identity_transform))  # Nodo iniziale (identità)
            folder_name = f"./output_images_matches_{i}"
            os.makedirs(folder_name, exist_ok=True)
            # reduced_dino_features = self.reduce_dino_descriptors(all_dino_descriptors[i], output_dim=64)
            for j in range(1, len(obj)):
                image0 = (self.imagelist[j-1] * 255).round().astype(np.uint8)
                image1 = (self.imagelist[j] * 255).round().astype(np.uint8)
                
                dino_desc0 = all_dino_descriptors[i][j-1].numpy()  
                gedi_desc0 = all_gedi_descriptors[i][j-1].numpy()  
                dino_desc1 = all_dino_descriptors[i][j].numpy()  
                gedi_desc1 = all_gedi_descriptors[i][j].numpy() 
                sp_features0 = all_sp_features[i][j-1]
                sp_features1 = all_sp_features[i][j]
                feature_3d_0 = all_3d_features[i][j-1]
                feature_3d_1 = all_3d_features[i][j]

                pts3d_0, pts3d_1, viz = self._get_matches(dino_desc0, dino_desc1, gedi_desc0, gedi_desc1, sp_features0, sp_features1, image0, image1, feature_3d_0, feature_3d_1)
                
                plt.figure(figsize=(20, 10), dpi=100, facecolor="w", edgecolor="k")
                plt.axis("off")
                plt.imshow(viz)
                
                plt.imsave(f"{folder_name}/demo_output_{j-1}_{j}.png", viz)

                pcd0_pts = np.array(obj[j-1])  # Nx3
                pcd1_pts = np.array(obj[j])    # Nx3

                pcd0 = o3d.geometry.PointCloud()
                pcd0.points = o3d.utility.Vector3dVector(pcd0_pts)

                pcd1 = o3d.geometry.PointCloud()
                pcd1.points = o3d.utility.Vector3dVector(pcd1_pts)
                
                # Estimate normals
                pcd0.estimate_normals()
                pcd1.estimate_normals()
                
                # TODO: Add as initial guess the transformation that we can find with RANSAC by optimizing Dino descriptors
                transformation_icp, information_icp = self.pairwise_registration(pcd0, pcd1)
        
                gedi_transformation[i].append(transformation_icp)
                
                current_cam2world = gedi_obj2world[i][j-1] @ transformation_icp
                gedi_obj2world[i].append(current_cam2world)
            
            # pcds = []
            # for j in range(len(obj)):
            #     pcd_pts = np.array(obj[j])  # Nx3

            #     pcd = o3d.geometry.PointCloud()
            #     pcd.points = o3d.utility.Vector3dVector(pcd_pts)
                
            #     # Estimate normals
            #     pcd.estimate_normals()
            #     pcds.append(pcd)

            # pose_graph, odometry_vec = self.full_registration(pcds)
            # print("Len odometry vec: ", len(odometry_vec))
            # print("Optimizing PoseGraph ...")
            # max_correspondence_distance_fine = 0.0005
            # option = o3d.pipelines.registration.GlobalOptimizationOption(
            #     max_correspondence_distance=max_correspondence_distance_fine,
            #     edge_prune_threshold=0.05,
            #     reference_node=0,
            #     preference_loop_closure=1.5)
            # with o3d.utility.VerbosityContextManager(
            #         o3d.utility.VerbosityLevel.Debug) as cm:
            #     o3d.pipelines.registration.global_optimization(
            #         pose_graph,
            #         o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            #         o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            #         option)
                
            # for j in range(len(obj)):
            #     # optimized_absolute_transforms[i].append(identity_transform @ odometry_vec[j])
            #     optimized_absolute_transforms[i].append(identity_transform @ np.linalg.inv(pose_graph.nodes[j].pose))

            # print("Pose graph optimization complete!")
                
        return gedi_transformation, gedi_obj2world
    

    def pairwise_registration(self, source, target):
        max_correspondence_distance_coarse = 0.05
        max_correspondence_distance_fine = 0.005
        icp_coarse = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance_coarse, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        icp_fine = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance_fine,
            icp_coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        transformation_icp = icp_fine.transformation
        information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, max_correspondence_distance_fine,
            icp_fine.transformation)
        return transformation_icp, information_icp
    

    def full_registration(self, pcds):
        pose_graph = o3d.pipelines.registration.PoseGraph()
        odometry = np.identity(4)
        odometry_vec = []
        odometry_vec.append(odometry)
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
        n_pcds = len(pcds)
        window_size = 6
        for source_id in range(n_pcds):
            for target_id in range(source_id + 1, min(source_id + window_size, n_pcds)):
                transformation_icp, information_icp = self.pairwise_registration(pcds[source_id], pcds[target_id])
                if target_id == source_id + 1:  # odometry case
                    print("Add node ", source_id, " and ", target_id, ".")
                    odometry = odometry @ transformation_icp
                    odometry_vec.append(odometry)
                    pose_graph.nodes.append(
                        o3d.pipelines.registration.PoseGraphNode(
                            odometry))
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                                target_id,
                                                                transformation_icp,
                                                                information_icp,
                                                                uncertain=False))
                else:  
                    print(f"Add edge {source_id} - {target_id}.")
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                                target_id,
                                                                transformation_icp,
                                                                information_icp,
                                                                uncertain=True))
        return pose_graph, odometry_vec



# pcd0_pts = np.vstack(obj[j-1])  # Object pts for scene j-1 (Nx3)
                # pcd1_pts = np.vstack(obj[j])    # Object pts for scene j (Mx3)

                # # Convert to Open3D PointCloud
                # pcd0 = o3d.geometry.PointCloud()
                # pcd0.points = o3d.utility.Vector3dVector(pcd0_pts)

                # pcd1 = o3d.geometry.PointCloud()
                # pcd1.points = o3d.utility.Vector3dVector(pcd1_pts)

                # # Estimate normals (only for visualization)
                # pcd0.estimate_normals()
                # pcd1.estimate_normals()
                
                # pts0 = torch.tensor(np.asarray(pcd0.points)).float()
                # pts1 = torch.tensor(np.asarray(pcd1.points)).float()

                # _pcd0 = torch.tensor(np.asarray(pcd0.points)).float()
                # _pcd1 = torch.tensor(np.asarray(pcd1.points)).float()

                # gedi0_desc = np.array(all_gedi_descriptors[i][j-1])
                # gedi1_desc = np.array(all_gedi_descriptors[i][j])

                # dino0_desc = np.array(all_dino_descriptors[i][j-1])
                # dino1_desc = np.array(all_dino_descriptors[i][j])

                # combined0_desc = np.hstack((gedi0_desc, dino0_desc))  # (S0, GeDi_dim + 768)
                # combined1_desc = np.hstack((gedi1_desc, dino1_desc))

                # # preparing format for open3d ransac
                # pcd0_dsdv = o3d.pipelines.registration.Feature()
                # pcd1_dsdv = o3d.pipelines.registration.Feature()

                # pcd0_dsdv.data = combined0_desc.T
                # pcd1_dsdv.data = combined1_desc.T

                # _pcd0 = o3d.geometry.PointCloud()
                # _pcd0.points = o3d.utility.Vector3dVector(pts0)
                # _pcd1 = o3d.geometry.PointCloud()
                # _pcd1.points = o3d.utility.Vector3dVector(pts1)

                # # applying ransac
                # est_result01 = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                #     _pcd0,
                #     _pcd1,
                #     pcd0_dsdv,
                #     pcd1_dsdv,
                #     mutual_filter=False,
                #     max_correspondence_distance=.02,
                #     estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                #     ransac_n=3,
                #     checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(.95),
                #             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(.05)],
                #     criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000))

                # # applying estimated transformation
                # pcd0.transform(est_result01.transformation)
                # # o3d.visualization.draw_geometries([pcd0, pcd1])
                # gedi_transformation[i].append(est_result01.transformation)
                # current_cam2world = current_cam2world @ est_result01.transformation
                # gedi_obj2world[i].append(current_cam2world)
                # print(f"Transformation between scene {j-1} and {j} is: ", est_result01.transformation)

                # Let's visualize the aligned point clouds

    # def get_squares_bboxes_from_mask(self, mask):
    #     """
    #     Extracts squared bounding boxes (BBoxes) from a multi-instance mask.

    #     Parameters:
    #         mask (numpy.array): Mask (H, W) with multiple object values (each object has a unique integer value).

    #     Returns:
    #         list of tuples: List of squared bounding boxes [(x_min, y_min, x_max, y_max)].
    #     """
    #     unique_values = np.unique(mask)  # Get unique object values
    #     bboxes = []
    #     H, W = mask.shape  # Image dimensions

    #     for value in unique_values:
    #         if value == 0:
    #             continue  # Skip background

    #         # Get non-zero pixels (object region for this value)
    #         coords = np.argwhere(mask == value)
    #         if coords.shape[0] == 0:
    #             continue  # No object found for this value

    #         # Compute bounding box from min/max coordinates
    #         y_min, x_min = coords.min(axis=0)
    #         y_max, x_max = coords.max(axis=0)

    #         # Compute bbox width and height
    #         w = x_max - x_min
    #         h = y_max - y_min
    #         side_length = max(w, h)  # Make it square by using the larger dimension

    #         # Compute center of the original bbox
    #         center_x = (x_min + x_max) // 2
    #         center_y = (y_min + y_max) // 2

    #         # Compute new square bbox (expanding symmetrically)
    #         new_x_min = max(0, center_x - side_length // 2)
    #         new_y_min = max(0, center_y - side_length // 2)
    #         new_x_max = min(W, new_x_min + side_length)
    #         new_y_max = min(H, new_y_min + side_length)

    #         # Append square bbox
    #         bboxes.append((new_x_min, new_y_min, new_x_max, new_y_max))

    #     return bboxes
    
    # def get_bboxes_from_mask(self, mask):
    #     """
    #     Extracts squared bounding boxes (BBoxes) from a multi-instance mask.

    #     Parameters:
    #         mask (numpy.array): Mask (H, W) with multiple object values (each object has a unique integer value).

    #     Returns:
    #         list of tuples: List of squared bounding boxes [(x_min, y_min, x_max, y_max)].
    #     """
    #     unique_values = np.unique(mask)  # Get unique object values
    #     bboxes = []
    #     H, W = mask.shape  # Image dimensions

    #     for value in unique_values:
    #         if value == 0:
    #             continue  # Skip background

    #         # Get non-zero pixels (object region for this value)
    #         coords = np.argwhere(mask == value)
    #         if coords.shape[0] == 0:
    #             continue  # No object found for this value

    #         # Compute bounding box from min/max coordinates
    #         y_min, x_min = coords.min(axis=0)
    #         y_max, x_max = coords.max(axis=0)

    #         # Append square bbox
    #         bboxes.append((x_min, y_min, x_max, y_max))

    #     return bboxes
    
    # def extract_dino_features(self, image_path):
    #     """Extracts DINO features from an image."""
    #     # Load and preprocess the image
    #     # image = Image.open(image_path).convert("RGB")
    #     image = Image.fromarray(image_path.astype(np.uint8))  # Convert NumPy array to PIL Image
    #     inputs = self.processor(images=image, return_tensors="pt")
        
    #     with torch.no_grad():
    #         outputs = self.model(**inputs)

    #     # Extract feature representations
    #     cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token (global representation)
    #     patch_features = outputs.last_hidden_state[:, 1:, :]  # Patch embeddings (spatial features)
        
    #     return cls_embedding, patch_features

    # def get_bboxes_from_mask(self, mask):
    #     """
    #     Extracts bounding boxes (BBoxes) from a multi-instance mask.

    #     Parameters:
    #         mask (numpy.array): Mask (H, W) with multiple object values (each object has a unique integer value).

    #     Returns:
    #         list of tuples: List of bounding boxes [(x_min, y_min, x_max, y_max)] for each unique object in the mask.
    #     """
    #     unique_values = np.unique(mask)  # Get unique object values
    #     bboxes = []
    #     for value in unique_values:
    #         if value == 0:
    #             continue  # Skip background

    #         # Get non-zero pixels (object region for this value)

    #         coords = np.argwhere(mask == value)
    #         if coords.shape[0] == 0:
    #             continue  # No object found for this value

    #         # Compute bounding box from min/max coordinates
    #         y_min, x_min = coords.min(axis=0)
    #         y_max, x_max = coords.max(axis=0)
    #         bboxes.append((x_min, y_min, x_max, y_max))

    #     return bboxes
    
    



    # def _extract_dino_features_from_mask(self, image_array, bboxes, mask, pts3d):
    #     """
    #     Extracts DINO features for multiple object regions in the given mask.

    #     Parameters:
    #         image_array (numpy array): Preprocessed image (H, W, C).
    #         bboxes (list of tuples): List of bounding boxes [(x_min, y_min, x_max, y_max)].
    #         mask (numpy array): Mask (H, W) containing different values for each object.

    #     Returns:
    #         list of torch.Tensor: Extracted patch features for each object [(N_obj_patches, 768)].
    #         list of numpy.array: Pixel positions of the selected patches for each object.
    #     """
    #     image = Image.fromarray((image_array*255).astype(np.uint8))  # Convert NumPy array to PIL Image
    #     image.save("original_image.png")

    #     all_features = []  # List to store features of all objects
    #     all_patches_centers = []  # List to store selected patch positions
    #     all_valid_features = []  # List to store features of valid patches
    #     all_valid_patch_centers = []  # List to store valid patch positions
    #     all_3d_pts = []

    #     # Ensure mask is uint8 (values in [0,255] for correct visualization)
    #     mask = (mask.astype(np.float32) / mask.max() * 255).astype(np.uint8)  # Normalize & scale
        
    #     for j, bbox in enumerate(bboxes):
    #         x_min, y_min, x_max, y_max = bbox

    #         # Crop the image and mask to the bounding box region
    #         cropped_image = image.crop((x_min, y_min, x_max, y_max))
    #         cropped_mask = mask[y_min:y_max, x_min:x_max]   # The value are between 0 and 255


    #         ################# DEBUG #################
    #         cropped_image.save(f"cropped_image_{j}.png")

    #         # Ensure mask is uint8 (values in [0,255] for correct visualization)
    #         if cropped_mask.max() <= 1:  
    #             cropped_mask = (cropped_mask * 255).astype(np.uint8)  # Scale to 0-255

    #         # Convert to PIL Image (grayscale mode 'L')
    #         mask_image = Image.fromarray(cropped_mask, mode="L")

    #         # Save the mask
    #         mask_image.save(f"cropped_mask_{j}.png")
    #         ########################################

    #         cropped_mask = (cropped_mask > 0).astype(np.uint8)  # Convert to binary mask (0 or 1)

    #         # Process cropped image for DINO
    #         inputs = self.processor(images=cropped_image, return_tensors="pt")

    #         with torch.no_grad():
    #             outputs = self.model(**inputs)

    #         patch_features = outputs.last_hidden_state[:, 1:, :]  # Patch embeddings (spatial features)

    #         patch_features = patch_features.reshape(1, 16, 16, -1).squeeze(0)  # Shape (16, 16, 768)

    #         H_bbox, W_bbox = cropped_image.size
    #         patch_H = H_bbox // 16  # Patch height in original bbox
    #         patch_W = W_bbox // 16  # Patch width in original bbox
    #         # print(f"Patch H: {patch_H}, Patch W: {patch_W}")

    #         valid_patch_features = []
    #         valid_3d_points = []

    #         for py in range(16):
    #             for px in range(16):
    #                 # Define patch boundaries on the original image
    #                 x_start = int(x_min + px * patch_W)
    #                 x_end = int(x_min + (px + 1) * patch_W)
    #                 y_start = int(y_min + py * patch_H)
    #                 y_end = int(y_min + (py + 1) * patch_H)
    #                 # print(f"Patch {px}, {py} - X: {x_start}-{x_end}, Y: {y_start}-{y_end}")

    #                 # Ensure within image bounds
    #                 x_end = min(x_end, x_max)
    #                 y_end = min(y_end, y_max)

    #                 # Get all pixels within this patch
    #                 patch_pixels_x, patch_pixels_y = np.meshgrid(
    #                     np.arange(x_start, x_end),
    #                     np.arange(y_start, y_end),
    #                     indexing="xy"
    #                 )
    #                 patch_pixels_x = patch_pixels_x.flatten()
    #                 patch_pixels_y = patch_pixels_y.flatten()
    #                 # print(f"Patch pixels X: {patch_pixels_x.shape}, Y: {patch_pixels_y.shape}")

    #                 # Store patch features for all pixels
    #                 # patch_features_all.extend([patch_features[py, px, :].cpu().numpy()] * len(patch_pixels_x))

    #                 # Check if they are inside the mask
    #                 for idx in range(len(patch_pixels_x)):
    #                     px_x, px_y = patch_pixels_x[idx], patch_pixels_y[idx]
    #                     if cropped_mask[px_y - y_min, px_x - x_min] > 0:  # Valid pixel inside mask
    #                         valid_patch_features.append(patch_features[py, px, :].cpu().numpy())
    #                         valid_3d_points.append(pts3d[px_y, px_x])  # Assign feature to all 3D points in patch


    #         if len(valid_patch_features) > 0:
    #             all_valid_features.append(np.array(valid_patch_features))  # Shape (N_valid_patches, 768)
    #             # all_valid_patch_centers.append(np.array(valid_patch_centers))  # Shape (N_valid_patches, 2)
    #             all_3d_pts.append(np.array(valid_3d_points))  # Shape (N_valid_patches, 3)
    #         else:
    #             all_valid_features.append(np.empty((0, 768)))  # Empty case
    #             # all_valid_patch_centers.append(np.empty((0, 2)))  # Empty case
    #             all_3d_pts.append(np.empty((0, 3)))


    #     return all_valid_features, all_3d_pts
    

    # def _get_3d_dino_patch_features(self):
    #     """Extracts DINO features from all images in the imagelist."""
    #     dino_valid_patch_features = []
    #     all_3d_pts_features = []
    #     for i, image_path in enumerate(self.imagelist):
    #         bboxes = self.get_squares_bboxes_from_mask(self.obj_msks[i])
    #         all_valid_dino_features, all_3d_pts = self._extract_dino_features_from_mask(image_path, bboxes, self.obj_msks[i], self.pts3d[i])
    #         dino_valid_patch_features.append(all_valid_dino_features)
    #         all_3d_pts_features.append(all_3d_pts)

    #     return dino_valid_patch_features, all_3d_pts_features
    
    # def _get_dino_cls_embeddings(self):
    #     """Extracts DINO CLS embeddings from all images in the imagelist."""
    #     dino_cls_embeddings = []
    #     for image_path in self.imagelist:
    #         cls_embedding, _ = self.extract_dino_features(image_path)
    #         dino_cls_embeddings.append(cls_embedding)
    #         # print(f"CLS Embedding Shape: {cls_embedding.shape}")
    #     return dino_cls_embeddings

    
    
    
    

    # def _get_icp_obj_track(self):
    #     """
    #     Performs classical point cloud registration using ICP instead of GeDi descriptors.
    #     """
    #     print(">> Starting ICP tracking...")
    #     icp_transformation = [[] for _ in range(self._get_object_quantity())]
    #     icp_obj2world = [[] for _ in range(self._get_object_quantity())]
    #     for i, obj in enumerate(self._get_all_3d_object_pts()):
    #         print(">> Tracking object ", i)
    #         # Compute centroid of the first object
    #         pcd0_pts = np.vstack(obj[0])  # First point cloud
    #         obj_centroid = np.mean(pcd0_pts, axis=0)

    #         # Create identity transformation with translation to centroid
    #         identity_transform = np.eye(4)
    #         identity_transform[:3, 3] = obj_centroid  # Set translation

    #         # Insert identity transformation at the start
    #         icp_transformation[i].append(identity_transform)
    #         icp_obj2world[i].append(identity_transform)
    #         current_cam2world = identity_transform
    #         print(f"Initial transformation (Identity at centroid) for object {i}:")
    #         print(identity_transform)

    #         for j in range(1, len(obj)):
    #             # Extract consecutive point clouds
    #             pcd0_pts = np.vstack(obj[j-1])  # Flatten list of numpy arrays
    #             pcd1_pts = np.vstack(obj[j])

    #             # Convert to Open3D PointCloud
    #             pcd0 = o3d.geometry.PointCloud()
    #             pcd0.points = o3d.utility.Vector3dVector(pcd0_pts)

    #             pcd1 = o3d.geometry.PointCloud()
    #             pcd1.points = o3d.utility.Vector3dVector(pcd1_pts)

    #             # Estimate normals (important for ICP)
    #             pcd0.estimate_normals()
    #             pcd1.estimate_normals()

    #             # Apply voxel downsampling for efficiency
    #             # pcd0 = pcd0.voxel_down_sample(self.voxel_size)
    #             # pcd1 = pcd1.voxel_down_sample(self.voxel_size)

    #             # Perform ICP registration
    #             icp_result = o3d.pipelines.registration.registration_icp(
    #                 pcd0, 
    #                 pcd1, 
    #                 max_correspondence_distance=0.005,  # Correspondence threshold
    #                 init=np.identity(4),  # Initial transformation
    #                 estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    #             )

    #             # Apply estimated transformation
    #             pcd0.transform(icp_result.transformation)
    #             icp_transformation[i].append(icp_result.transformation)
    #             current_cam2world = current_cam2world @ icp_result.transformation
    #             icp_obj2world[i].append(current_cam2world)
    #             # Print transformation matrix
    #             print(f"ICP Transformation between scene {j-1} and {j} is: \n", icp_result.transformation)

    #     return icp_transformation, icp_obj2world
    

    # def _get_dino_obj_track(self, valid_3d_pts_t, dino_valid_patch_features):
    #     print(">> Starting GeDi tracking...")
    #     dino_transformation = [[] for _ in range(self._get_object_quantity())]
    #     dino_obj2world = [[] for _ in range(self._get_object_quantity())]
    #     for i, obj in enumerate(self._get_all_3d_object_pts()):

    #         print(">> Tracking object ", i)

    #         identity_transform = self.create_first_frame(obj)
    #         dino_transformation[i].append(identity_transform)
    #         dino_obj2world[i].append(identity_transform)
    #         current_cam2world = identity_transform

    #         for j in range(1, len(obj)):

    #             pcd0_pts = np.vstack(valid_3d_pts_t[i][j-1])  # Scene j-1 (Nx3)
    #             pcd1_pts = np.vstack(valid_3d_pts_t[i][j])    # Scene j (Mx3)

    #             # Convert to Open3D PointCloud
    #             pcd0 = o3d.geometry.PointCloud()
    #             pcd0.points = o3d.utility.Vector3dVector(pcd0_pts)

    #             pcd1 = o3d.geometry.PointCloud()
    #             pcd1.points = o3d.utility.Vector3dVector(pcd1_pts)   
                
    #             print("Before outlier removal:")
    #             print(f"Number of points in scene {j-1}: {np.asarray(pcd0.points).shape}")
    #             print(f"Number of points in scene {j}: {np.asarray(pcd1.points).shape}")

    #             _, ind = pcd0.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
    #             pcd0 = pcd0.select_by_index(ind)  # Keep only the inliers

    #             _, ind = pcd1.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
    #             pcd1 = pcd1.select_by_index(ind)  # Keep only the inliers

    #             print("After outlier removal:")
    #             print(f"Number of points in scene {j-1}: {np.asarray(pcd0.points).shape}")
    #             print(f"Number of points in scene {j}: {np.asarray(pcd1.points).shape}")
                
    #             pcd0.paint_uniform_color([1, 0.706, 0])
    #             pcd1.paint_uniform_color([0, 0.651, 0.929])
                
    #             o3d.visualization.draw_geometries([pcd0, pcd1])
                
    #             # Estimate normals (only for visualization)
    #             pcd0.estimate_normals()
    #             pcd1.estimate_normals()


    #             num_points_pcd0 = np.asarray(pcd0.points).shape[0]
    #             num_points_pcd1 = np.asarray(pcd1.points).shape[0]

    #             patches0 = min(self.patches_per_pair, num_points_pcd0)
    #             patches1 = min(self.patches_per_pair, num_points_pcd1)

    #             # randomly sampling some points from the point cloud
    #             inds0 = np.random.choice(np.asarray(pcd0.points).shape[0], patches0, replace=False)
    #             inds1 = np.random.choice(np.asarray(pcd1.points).shape[0], patches1, replace=False)
                
    #             pts0 = torch.tensor(np.asarray(pcd0.points)[inds0]).float()
    #             pts1 = torch.tensor(np.asarray(pcd1.points)[inds1]).float()
                
    #             # pts0 = torch.tensor(np.asarray(pcd0.points)).float()
    #             # pts1 = torch.tensor(np.asarray(pcd1.points)).float()

    #             print("Number of pts0: ", pts0.shape)
    #             print("Number of pts1: ", pts1.shape)

    #             pcd0 = pcd0.voxel_down_sample(self.voxel_size)
    #             pcd1 = pcd1.voxel_down_sample(self.voxel_size)

    #             _pcd0 = torch.tensor(np.asarray(pcd0.points)).float()
    #             _pcd1 = torch.tensor(np.asarray(pcd1.points)).float()

    #             print("Gedi descriptor extraction...")
    #             # Compute GeDi descriptors
    #             gedi0_desc = self.gedi.compute(pts=torch.tensor(valid_3d_pts_t[i][j-1][inds0]).float(), pcd=_pcd0)
    #             gedi1_desc = self.gedi.compute(pts=torch.tensor(valid_3d_pts_t[i][j][inds1]).float(), pcd=_pcd1)
                
    #             print(f"Number of GeDi descriptors in scene {j-1}: {gedi0_desc.shape}")
    #             print(f"Number of GeDi descriptors in scene {j}: {gedi1_desc.shape}")

    #             print("Dino descriptor extraction...")
    #             # random_indices = np.random.choice(768, 64, replace=False)

    #             dino0_desc = np.vstack(dino_valid_patch_features[i][j-1][inds0])   # (S, 768)
    #             dino1_desc = np.vstack(dino_valid_patch_features[i][j][inds1])     # (S, 768)

    #             # Reduce DINO descriptors to 64 dimensions
    #             dino0_desc_reduced = dino0_desc  
    #             dino1_desc_reduced = dino1_desc
    #             print(f"Number of DINO descriptors in scene {j-1}: {dino0_desc_reduced.shape}")
    #             print(f"Number of DINO descriptors in scene {j}: {dino1_desc_reduced.shape}")

    #             # Concatenate GeDi + DINO descriptors
    #             combined0_desc = np.hstack((gedi0_desc, dino0_desc_reduced))  # (S0, GeDi_dim + 768)
    #             combined1_desc = np.hstack((gedi1_desc, dino1_desc_reduced))  # (S1, GeDi_dim + 768)

    #             combined0_desc = combined0_desc / np.linalg.norm(combined0_desc, axis=1, keepdims=True)
    #             combined1_desc = combined1_desc / np.linalg.norm(combined1_desc, axis=1, keepdims=True)

    #             # preparing format for open3d ransac
    #             pcd0_dsdv = o3d.pipelines.registration.Feature()
    #             pcd1_dsdv = o3d.pipelines.registration.Feature()

    #             pcd0_dsdv.data = combined0_desc.T
    #             pcd1_dsdv.data = combined1_desc.T

    #             _pcd0 = o3d.geometry.PointCloud()
    #             _pcd0.points = o3d.utility.Vector3dVector(pts0)
    #             _pcd1 = o3d.geometry.PointCloud()
    #             _pcd1.points = o3d.utility.Vector3dVector(pts1)

    #             # applying ransac
    #             est_result01 = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    #                 _pcd0,
    #                 _pcd1,
    #                 pcd0_dsdv,
    #                 pcd1_dsdv,
    #                 mutual_filter=False,
    #                 max_correspondence_distance=.0001,
    #                 estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    #                 ransac_n=3,
    #                 checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(.9),
    #                         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(.02)],
    #                 criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 5000))


    #             # Refine with ICP
    #             icp_refinement_result = o3d.pipelines.registration.registration_icp(
    #                 _pcd0,  # Source point cloud
    #                 _pcd1,  # Target point cloud
    #                 max_correspondence_distance=0.005,  # Stricter threshold for ICP
    #                 init=est_result01.transformation,  # Use RANSAC transformation as the initial guess
    #                 estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    #             )


    #             # applying estimated transformation
    #             pcd0.transform(est_result01.transformation)
    #             dino_transformation[i].append(icp_refinement_result.transformation)
    #             current_cam2world = current_cam2world @ icp_refinement_result.transformation
    #             dino_obj2world[i].append(current_cam2world)
    #             print(f"Transformation between scene {j-1} and {j} is: ", icp_refinement_result.transformation)
        
    #     return dino_transformation, dino_obj2world