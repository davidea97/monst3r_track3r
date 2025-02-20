import torch
import numpy as np
import open3d as o3d
from gedi.gedi import GeDi
from dust3r.utils.vo_eval import save_trajectory_tum_format
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

DINO_FEATURE_DIM_B = 768
DINO_FEATURE_DIM_L = 1024

class DinoTracker:
    def __init__(self, images, masks):

        dino_export="./models/dinov2_vitl14_pretrain.pth"
        self.dino_extract = dino_extract.DINOExtract(dino_export, feature_layer=1)
        self.images = images
        self.masks = masks

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
                for k, (x, y) in enumerate(mask_coords):
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
                cv2.imwrite(f"output_image_{j}_{i}.png", output_image)

            all_dino_images.append(dino_images)
            video = self.create_dino_video(dino_images)

        return all_dino_images

    def extract_dino_features(self):
        masks = self.masks
        images = self.images

        all_dino_descriptors = []
        print(">> Extracting DINO features...")
        objects_num = len(np.unique(masks[0]))-1

        for i in tqdm(range(len(images))):
            image = images[i]
            height, width = image.shape[:2]
            dino_descriptors_objs = []

            # Extract DINO features for each object in the image    
            for j in range(objects_num):  # Loop over all objects in the image
                image =(image * 255).round().astype(np.uint8)
                dino_features = self.dino_extract(image)
                mask_coords = np.argwhere(masks[i] == j+1)
                mask_coords = mask_coords[:, [1, 0]]                

                dino_descriptors = dino_extract.get_dino_descriptors(
                    dino_features,
                    tf.convert_to_tensor(mask_coords, dtype=tf.float32),
                    tf.convert_to_tensor(height, dtype=tf.int32),
                    tf.convert_to_tensor(width, dtype=tf.int32),
                    DINO_FEATURE_DIM_L,
                )

                dino_descriptors_objs.append(dino_descriptors)

            all_dino_descriptors.append(dino_descriptors_objs)
        all_dino_descriptors = list(map(list, zip(*all_dino_descriptors)))

        return all_dino_descriptors, mask_coords
    

    def extract_gedi_features(self):

        all_gedi_descriptors = []
        for i, obj in enumerate(self._get_all_3d_object_pts()):
            
            gedi_desc_objs = []
            print(">> Extracting GeDi features for object ", i)
            for j in tqdm(range(0, len(obj))):
                pcd0_pts = np.vstack(obj[j])  # Object pts for scene j-1 (Nx3)
               
                # Convert to Open3D PointCloud
                pcd0 = o3d.geometry.PointCloud()
                pcd0.points = o3d.utility.Vector3dVector(pcd0_pts)

                # Estimate normals (only for visualization)
                pcd0.estimate_normals()
                pts0 = torch.tensor(np.asarray(pcd0.points)).float()
                _pcd0 = torch.tensor(np.asarray(pcd0.points)).float()

                # computing descriptors
                # gedi_desc = self.gedi.compute(pts=pts0, pcd=_pcd0)
                gedi_desc = self.gedi.compute_multi_scale(pts=pts0, pcd=_pcd0)
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

        
    def _obj_track(self, all_dino_descriptors, all_gedi_descriptors):
        print(">> Starting GeDi & Dino tracking...")
        gedi_transformation = [[] for _ in range(self._get_object_quantity())]
        gedi_obj2world = [[] for _ in range(self._get_object_quantity())]
        for i, obj in enumerate(self._get_all_3d_object_pts()):
            print(">> Tracking object ", i)
            # Compute centroid of the first object
            identity_transform = self.create_first_frame(obj) # T_{Obj0}^{Cam}

            # Insert identity transformation at the start
            gedi_transformation[i].append(identity_transform)

            # TODO: add the first frame initializations, for example PCA, etc..
            gedi_obj2world[i].append(identity_transform)
            current_cam2world = identity_transform

            all_dino_descriptors_vis = np.concatenate(all_dino_descriptors[i], axis=0)
            pca = PCA(n_components=3)
            pca.fit(all_dino_descriptors_vis)
            
            # reduced_dino_features = self.reduce_dino_descriptors(all_dino_descriptors[i], output_dim=64)
            for j in range(1, len(obj)):
                print(f"Processing frame {j}...")
                if j % 5 == 0:
                    print(f"PROCESSING FRAME {j}...")
                pcd0_pts = np.vstack(obj[j-1])  # Object pts for scene j-1 (Nx3)
                pcd1_pts = np.vstack(obj[j])    # Object pts for scene j (Mx3)

                # Convert to Open3D PointCloud
                pcd0 = o3d.geometry.PointCloud()
                pcd0.points = o3d.utility.Vector3dVector(pcd0_pts)

                pcd1 = o3d.geometry.PointCloud()
                pcd1.points = o3d.utility.Vector3dVector(pcd1_pts)

                # Estimate normals (only for visualization)
                pcd0.estimate_normals()
                pcd1.estimate_normals()
                
                pts0 = torch.tensor(np.asarray(pcd0.points)).float()
                pts1 = torch.tensor(np.asarray(pcd1.points)).float()

                _pcd0 = torch.tensor(np.asarray(pcd0.points)).float()
                _pcd1 = torch.tensor(np.asarray(pcd1.points)).float()

                # computing descriptors
                gedi0_desc = np.array(all_gedi_descriptors[i][j-1])
                gedi1_desc = np.array(all_gedi_descriptors[i][j])

                dino0_desc = np.array(all_dino_descriptors[i][j-1])
                dino1_desc = np.array(all_dino_descriptors[i][j])

                combined0_desc = np.hstack((gedi0_desc, dino0_desc))  # (S0, GeDi_dim + 768)
                combined1_desc = np.hstack((gedi1_desc, dino1_desc))

                # preparing format for open3d ransac
                pcd0_dsdv = o3d.pipelines.registration.Feature()
                pcd1_dsdv = o3d.pipelines.registration.Feature()

                pcd0_dsdv.data = combined0_desc.T
                pcd1_dsdv.data = combined1_desc.T

                _pcd0 = o3d.geometry.PointCloud()
                _pcd0.points = o3d.utility.Vector3dVector(pts0)
                _pcd1 = o3d.geometry.PointCloud()
                _pcd1.points = o3d.utility.Vector3dVector(pts1)

                # applying ransac
                est_result01 = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                    _pcd0,
                    _pcd1,
                    pcd0_dsdv,
                    pcd1_dsdv,
                    mutual_filter=False,
                    max_correspondence_distance=.02,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                    ransac_n=3,
                    checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(.95),
                            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(.05)],
                    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000))

                # Read a local pcl
                pcd_gt0 = o3d.io.read_point_cloud("../TRELLIS/sample.ply")

                # Use ICP for estimating the transformation with pcd_gt0 as the target
                # est_result01_icp = o3d.pipelines.registration.registration_icp(
                #     _pcd0,  # Source point cloud
                #     pcd_gt0,  # Target point cloud
                #     max_correspondence_distance=0.005,  # Stricter threshold for ICP
                #     init=est_result01.transformation,  # Use RANSAC transformation as the initial guess
                #     estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
                # )

                # est_result02_icp = o3d.pipelines.registration.registration_icp(
                #     _pcd1,  # Source point cloud
                #     pcd_gt0,  # Target point cloud
                #     max_correspondence_distance=0.005,  # Stricter threshold for ICP
                #     init=est_result01.transformation,  # Use RANSAC transformation as the initial guess
                #     estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
                # )

                # print("Transformation is: ", est_result01_icp.transformation)

                # applying estimated transformation
                pcd0.transform(est_result01.transformation)
                # o3d.visualization.draw_geometries([pcd0, pcd1])
                gedi_transformation[i].append(est_result01.transformation)
                current_cam2world = current_cam2world @ est_result01.transformation
                gedi_obj2world[i].append(current_cam2world)
                print(f"Transformation between scene {j-1} and {j} is: ", est_result01.transformation)

                # Let's visualize the aligned point clouds
                # pcd0.paint_uniform_color([1, 0.706, 0])
                # pcd1.paint_uniform_color([0, 0.651, 0.929])

                # o3d.visualization.draw_geometries([pcd0, pcd1])
        
        return gedi_transformation, gedi_obj2world
    
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