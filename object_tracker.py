import torch
import numpy as np
import open3d as o3d
from gedi.gedi import GeDi
from dust3r.utils.vo_eval import save_trajectory_tum_format
from dust3r.cloud_opt.base_opt import c2w_to_tumpose
from transformers import AutoProcessor, AutoModel
from PIL import Image

class ObjectTracker:
    def __init__(self, all_3d_obj_pts, obj_msks, imagelist, pts3d):
        self.all_3d_obj_pts = all_3d_obj_pts
        self.obj_msks = obj_msks
        self.imagelist = imagelist
        self.pts3d = pts3d
        
        config = {'dim': 32,                                                # descriptor output dimension
                'samples_per_batch': 500,                                   # batches to process the data on GPU
                'samples_per_patch_lrf': 4000,                              # num. of point to process with LRF
                'samples_per_patch_out': 512,                               # num. of points to sample for pointnet++
                'r_lrf': .5,                                                # LRF radius
                'fchkpt_gedi_net': 'gedi/data/chkpts/3dmatch/chkpt.tar'}    # path to checkpoint
        
        self.voxel_size = .01
        self.patches_per_pair = 5000

        # initialising class
        self.gedi = GeDi(config=config)

        # Load DINO model and processor
        model_name = "facebook/dinov2-base" # dinov2-small (384), dinov2-base (768), dinov2-large (1024), dinov2-giant (1536)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode

    def _get_object_quantity(self):
        return len(self.all_3d_obj_pts)

    def _get_3d_object_pts(self, obj_id):
        return self.all_3d_obj_pts[obj_id]
    
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
    
    def extract_dino_features(self, image_path):
        """Extracts DINO features from an image."""
        # Load and preprocess the image
        # image = Image.open(image_path).convert("RGB")
        image = Image.fromarray(image_path.astype(np.uint8))  # Convert NumPy array to PIL Image
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract feature representations
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token (global representation)
        patch_features = outputs.last_hidden_state[:, 1:, :]  # Patch embeddings (spatial features)
        
        return cls_embedding, patch_features

    def get_bboxes_from_mask(self, mask):
        """
        Extracts bounding boxes (BBoxes) from a multi-instance mask.

        Parameters:
            mask (numpy.array): Mask (H, W) with multiple object values (each object has a unique integer value).

        Returns:
            list of tuples: List of bounding boxes [(x_min, y_min, x_max, y_max)] for each unique object in the mask.
        """
        unique_values = np.unique(mask)  # Get unique object values
        bboxes = []
        for value in unique_values:
            if value == 0:
                continue  # Skip background

            # Get non-zero pixels (object region for this value)

            coords = np.argwhere(mask == value)
            if coords.shape[0] == 0:
                continue  # No object found for this value

            # Compute bounding box from min/max coordinates
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            bboxes.append((x_min, y_min, x_max, y_max))

        return bboxes
    
    def get_squares_bboxes_from_mask(self, mask):
        """
        Extracts squared bounding boxes (BBoxes) from a multi-instance mask.

        Parameters:
            mask (numpy.array): Mask (H, W) with multiple object values (each object has a unique integer value).

        Returns:
            list of tuples: List of squared bounding boxes [(x_min, y_min, x_max, y_max)].
        """
        unique_values = np.unique(mask)  # Get unique object values
        bboxes = []
        H, W = mask.shape  # Image dimensions

        for value in unique_values:
            if value == 0:
                continue  # Skip background

            # Get non-zero pixels (object region for this value)
            coords = np.argwhere(mask == value)
            if coords.shape[0] == 0:
                continue  # No object found for this value

            # Compute bounding box from min/max coordinates
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            # Compute bbox width and height
            w = x_max - x_min
            h = y_max - y_min
            side_length = max(w, h)  # Make it square by using the larger dimension

            # Compute center of the original bbox
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2

            # Compute new square bbox (expanding symmetrically)
            new_x_min = max(0, center_x - side_length // 2)
            new_y_min = max(0, center_y - side_length // 2)
            new_x_max = min(W, new_x_min + side_length)
            new_y_max = min(H, new_y_min + side_length)

            # Append square bbox
            bboxes.append((new_x_min, new_y_min, new_x_max, new_y_max))

        return bboxes



    def _extract_dino_features_from_mask(self, image_array, bboxes, mask, pts3d):
        """
        Extracts DINO features for multiple object regions in the given mask.

        Parameters:
            image_array (numpy array): Preprocessed image (H, W, C).
            bboxes (list of tuples): List of bounding boxes [(x_min, y_min, x_max, y_max)].
            mask (numpy array): Mask (H, W) containing different values for each object.

        Returns:
            list of torch.Tensor: Extracted patch features for each object [(N_obj_patches, 768)].
            list of numpy.array: Pixel positions of the selected patches for each object.
        """
        image = Image.fromarray((image_array*255).astype(np.uint8))  # Convert NumPy array to PIL Image
        image.save("original_image.png")

        all_features = []  # List to store features of all objects
        all_patches_centers = []  # List to store selected patch positions
        all_valid_features = []  # List to store features of valid patches
        all_valid_patch_centers = []  # List to store valid patch positions
        all_3d_pts = []

        # Ensure mask is uint8 (values in [0,255] for correct visualization)
        mask = (mask.astype(np.float32) / mask.max() * 255).astype(np.uint8)  # Normalize & scale
        
        for j, bbox in enumerate(bboxes):
            x_min, y_min, x_max, y_max = bbox

            # Crop the image and mask to the bounding box region
            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            cropped_mask = mask[y_min:y_max, x_min:x_max]   # The value are between 0 and 255


            ################# DEBUG #################
            cropped_image.save(f"cropped_image_{j}.png")

            # Ensure mask is uint8 (values in [0,255] for correct visualization)
            if cropped_mask.max() <= 1:  
                cropped_mask = (cropped_mask * 255).astype(np.uint8)  # Scale to 0-255

            # Convert to PIL Image (grayscale mode 'L')
            mask_image = Image.fromarray(cropped_mask, mode="L")

            # Save the mask
            mask_image.save(f"cropped_mask_{j}.png")
            ########################################

            cropped_mask = (cropped_mask > 0).astype(np.uint8)  # Convert to binary mask (0 or 1)

            # Process cropped image for DINO
            inputs = self.processor(images=cropped_image, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model(**inputs)

            patch_features = outputs.last_hidden_state[:, 1:, :]  # Patch embeddings (spatial features)

            patch_features = patch_features.reshape(1, 16, 16, -1).squeeze(0)  # Shape (16, 16, 768)

            H_bbox, W_bbox = cropped_image.size
            patch_H = H_bbox // 16  # Patch height in original bbox
            patch_W = W_bbox // 16  # Patch width in original bbox
            # print(f"Patch H: {patch_H}, Patch W: {patch_W}")

            valid_patch_features = []
            valid_3d_points = []

            for py in range(16):
                for px in range(16):
                    # Define patch boundaries on the original image
                    x_start = int(x_min + px * patch_W)
                    x_end = int(x_min + (px + 1) * patch_W)
                    y_start = int(y_min + py * patch_H)
                    y_end = int(y_min + (py + 1) * patch_H)
                    # print(f"Patch {px}, {py} - X: {x_start}-{x_end}, Y: {y_start}-{y_end}")

                    # Ensure within image bounds
                    x_end = min(x_end, x_max)
                    y_end = min(y_end, y_max)

                    # Get all pixels within this patch
                    patch_pixels_x, patch_pixels_y = np.meshgrid(
                        np.arange(x_start, x_end),
                        np.arange(y_start, y_end),
                        indexing="xy"
                    )
                    patch_pixels_x = patch_pixels_x.flatten()
                    patch_pixels_y = patch_pixels_y.flatten()
                    # print(f"Patch pixels X: {patch_pixels_x.shape}, Y: {patch_pixels_y.shape}")

                    # Store patch features for all pixels
                    # patch_features_all.extend([patch_features[py, px, :].cpu().numpy()] * len(patch_pixels_x))

                    # Check if they are inside the mask
                    for idx in range(len(patch_pixels_x)):
                        px_x, px_y = patch_pixels_x[idx], patch_pixels_y[idx]
                        if cropped_mask[px_y - y_min, px_x - x_min] > 0:  # Valid pixel inside mask
                            valid_patch_features.append(patch_features[py, px, :].cpu().numpy())
                            valid_3d_points.append(pts3d[px_y, px_x])  # Assign feature to all 3D points in patch


            if len(valid_patch_features) > 0:
                all_valid_features.append(np.array(valid_patch_features))  # Shape (N_valid_patches, 768)
                # all_valid_patch_centers.append(np.array(valid_patch_centers))  # Shape (N_valid_patches, 2)
                all_3d_pts.append(np.array(valid_3d_points))  # Shape (N_valid_patches, 3)
            else:
                all_valid_features.append(np.empty((0, 768)))  # Empty case
                # all_valid_patch_centers.append(np.empty((0, 2)))  # Empty case
                all_3d_pts.append(np.empty((0, 3)))


        return all_valid_features, all_3d_pts
    

    def _get_3d_dino_patch_features(self):
        """Extracts DINO features from all images in the imagelist."""
        dino_valid_patch_features = []
        all_3d_pts_features = []
        for i, image_path in enumerate(self.imagelist):
            # _, patch_features = self.extract_dino_features(image_path)
            # bboxes = self.get_bboxes_from_mask(self.obj_msks[i])
            bboxes = self.get_squares_bboxes_from_mask(self.obj_msks[i])
            # TODO: first step directly add a square box 224x224, then a square box smaller fitting the object, at the end try with bbox from mask
            # print(f"Bboxes for image {i}: {bboxes}")
            all_valid_dino_features, all_3d_pts = self._extract_dino_features_from_mask(image_path, bboxes, self.obj_msks[i], self.pts3d[i])
            dino_valid_patch_features.append(all_valid_dino_features)
            all_3d_pts_features.append(all_3d_pts)

        return dino_valid_patch_features, all_3d_pts_features
    
    def _get_dino_cls_embeddings(self):
        """Extracts DINO CLS embeddings from all images in the imagelist."""
        dino_cls_embeddings = []
        for image_path in self.imagelist:
            cls_embedding, _ = self.extract_dino_features(image_path)
            dino_cls_embeddings.append(cls_embedding)
            # print(f"CLS Embedding Shape: {cls_embedding.shape}")
        return dino_cls_embeddings

    
    def create_first_frame(self, obj):
        pcd0_pts = np.vstack(obj[0])  # First point cloud
        obj_centroid = np.mean(pcd0_pts, axis=0)

        # Create identity transformation with translation to centroid
        identity_transform = np.eye(4)
        identity_transform[:3, 3] = obj_centroid  # Set translation
        return identity_transform
    
    def _get_gedi_obj_track(self):
        print(">> Starting GeDi tracking...")
        gedi_transformation = [[] for _ in range(self._get_object_quantity())]
        gedi_obj2world = [[] for _ in range(self._get_object_quantity())]
        for i, obj in enumerate(self._get_all_3d_object_pts()):
            # Compute tracking for object i
            print(">> Tracking object ", i)
            # Compute centroid of the first object
            identity_transform = self.create_first_frame(obj)

            # Insert identity transformation at the start
            gedi_transformation[i].append(identity_transform)

            # TODO: add the first frame initializations, for example PCA, etc..
            gedi_obj2world[i].append(identity_transform)
            current_cam2world = identity_transform

            for j in range(1, len(obj)):

                pcd0_pts = np.vstack(obj[j-1])  # Object pts for scene j-1 (Nx3)
                pcd1_pts = np.vstack(obj[j])    # Object pts for scene j (Mx3)

                # print(f"Number of points in scene {j-1}: {pcd0_pts.shape}") 
                # print(f"Number of points in scene {j}: {pcd1_pts.shape}")   

                # Convert to Open3D PointCloud
                pcd0 = o3d.geometry.PointCloud()
                pcd0.points = o3d.utility.Vector3dVector(pcd0_pts)

                pcd1 = o3d.geometry.PointCloud()
                pcd1.points = o3d.utility.Vector3dVector(pcd1_pts)

                pcd0.paint_uniform_color([1, 0.706, 0])
                pcd1.paint_uniform_color([0, 0.651, 0.929])

                # Estimate normals (only for visualization)
                pcd0.estimate_normals()
                pcd1.estimate_normals()

                # o3d.visualization.draw_geometries([pcd0, pcd1])

                num_points_pcd0 = np.asarray(pcd0.points).shape[0]
                num_points_pcd1 = np.asarray(pcd1.points).shape[0]

                patches0 = min(self.patches_per_pair, num_points_pcd0)
                patches1 = min(self.patches_per_pair, num_points_pcd1)

                # randomly sampling some points from the point cloud
                inds0 = np.random.choice(np.asarray(pcd0.points).shape[0], patches0, replace=False)
                inds1 = np.random.choice(np.asarray(pcd1.points).shape[0], patches1, replace=False)

                pts0 = torch.tensor(np.asarray(pcd0.points)[inds0]).float()
                pts1 = torch.tensor(np.asarray(pcd1.points)[inds1]).float()

                # applying voxelisation to the point cloud
                pcd0 = pcd0.voxel_down_sample(self.voxel_size)
                pcd1 = pcd1.voxel_down_sample(self.voxel_size)

                _pcd0 = torch.tensor(np.asarray(pcd0.points)).float()
                _pcd1 = torch.tensor(np.asarray(pcd1.points)).float()

                # computing descriptors
                pcd0_desc = self.gedi.compute(pts=pts0, pcd=_pcd0)
                pcd1_desc = self.gedi.compute(pts=pts1, pcd=_pcd1)

                # preparing format for open3d ransac
                pcd0_dsdv = o3d.pipelines.registration.Feature()
                pcd1_dsdv = o3d.pipelines.registration.Feature()

                pcd0_dsdv.data = pcd0_desc.T
                pcd1_dsdv.data = pcd1_desc.T

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
                    mutual_filter=True,
                    max_correspondence_distance=.02,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                    ransac_n=3,
                    checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(.9),
                            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(.02)],
                    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000))

                # applying estimated transformation
                pcd0.transform(est_result01.transformation)
                # o3d.visualization.draw_geometries([pcd0, pcd1])
                gedi_transformation[i].append(est_result01.transformation)
                current_cam2world = current_cam2world @ est_result01.transformation
                gedi_obj2world[i].append(current_cam2world)
                print(f"Transformation between scene {j-1} and {j} is: ", est_result01.transformation)
        
        return gedi_transformation, gedi_obj2world

    def _get_icp_obj_track(self):
        """
        Performs classical point cloud registration using ICP instead of GeDi descriptors.
        """
        print(">> Starting ICP tracking...")
        icp_transformation = [[] for _ in range(self._get_object_quantity())]
        icp_obj2world = [[] for _ in range(self._get_object_quantity())]
        for i, obj in enumerate(self._get_all_3d_object_pts()):
            print(">> Tracking object ", i)
            # Compute centroid of the first object
            pcd0_pts = np.vstack(obj[0])  # First point cloud
            obj_centroid = np.mean(pcd0_pts, axis=0)

            # Create identity transformation with translation to centroid
            identity_transform = np.eye(4)
            identity_transform[:3, 3] = obj_centroid  # Set translation

            # Insert identity transformation at the start
            icp_transformation[i].append(identity_transform)
            icp_obj2world[i].append(identity_transform)
            current_cam2world = identity_transform
            print(f"Initial transformation (Identity at centroid) for object {i}:")
            print(identity_transform)

            for j in range(1, len(obj)):
                # Extract consecutive point clouds
                pcd0_pts = np.vstack(obj[j-1])  # Flatten list of numpy arrays
                pcd1_pts = np.vstack(obj[j])

                # Convert to Open3D PointCloud
                pcd0 = o3d.geometry.PointCloud()
                pcd0.points = o3d.utility.Vector3dVector(pcd0_pts)

                pcd1 = o3d.geometry.PointCloud()
                pcd1.points = o3d.utility.Vector3dVector(pcd1_pts)

                # Estimate normals (important for ICP)
                pcd0.estimate_normals()
                pcd1.estimate_normals()

                # Apply voxel downsampling for efficiency
                # pcd0 = pcd0.voxel_down_sample(self.voxel_size)
                # pcd1 = pcd1.voxel_down_sample(self.voxel_size)

                # Perform ICP registration
                icp_result = o3d.pipelines.registration.registration_icp(
                    pcd0, 
                    pcd1, 
                    max_correspondence_distance=0.005,  # Correspondence threshold
                    init=np.identity(4),  # Initial transformation
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
                )

                # Apply estimated transformation
                pcd0.transform(icp_result.transformation)
                icp_transformation[i].append(icp_result.transformation)
                current_cam2world = current_cam2world @ icp_result.transformation
                icp_obj2world[i].append(current_cam2world)
                # Print transformation matrix
                print(f"ICP Transformation between scene {j-1} and {j} is: \n", icp_result.transformation)

        return icp_transformation, icp_obj2world
    

    def _get_dino_obj_track(self, valid_3d_pts_t, dino_valid_patch_features):
        print(">> Starting GeDi tracking...")
        dino_transformation = [[] for _ in range(self._get_object_quantity())]
        dino_obj2world = [[] for _ in range(self._get_object_quantity())]
        for i, obj in enumerate(self._get_all_3d_object_pts()):
            # Compute tracking for object i
            print(">> Tracking object ", i)
            # Compute centroid of the first object
            identity_transform = self.create_first_frame(obj)

            # Insert identity transformation at the start
            dino_transformation[i].append(identity_transform)

            # TODO: add the first frame initializations, for example PCA, etc..
            dino_obj2world[i].append(identity_transform)
            current_cam2world = identity_transform
            print(f"Initial transformation (Identity at centroid) for object {i}:")
            print(identity_transform)

            for j in range(1, len(obj)):

                pcd0_pts = np.vstack(obj[j-1])  # Scene j-1 (Nx3)
                pcd1_pts = np.vstack(obj[j])    # Scene j (Mx3)

                print("Number of points in scene ", j-1, ": ", pcd0_pts.shape)
                print("Number of points in scene ", j, ": ", pcd1_pts.shape)

                dino0_desc = np.vstack(dino_valid_patch_features[i][j-1])   # (S, 768)
                dino1_desc = np.vstack(dino_valid_patch_features[i][j])     # (S, 768)

                # Convert to Open3D PointCloud
                pcd0 = o3d.geometry.PointCloud()
                pcd0.points = o3d.utility.Vector3dVector(pcd0_pts)

                pcd1 = o3d.geometry.PointCloud()
                pcd1.points = o3d.utility.Vector3dVector(pcd1_pts)
                
                
                print("Before outlier removal:")
                print(f"Number of points in scene {j-1}: {np.asarray(pcd0.points).shape}")
                print(f"Number of points in scene {j}: {np.asarray(pcd1.points).shape}")

                cl, ind = pcd0.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
                pcd0 = pcd0.select_by_index(ind)  # Keep only the inliers

                cl, ind = pcd1.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
                pcd1 = pcd1.select_by_index(ind)  # Keep only the inliers

                pcd0.paint_uniform_color([1, 0.706, 0])
                pcd1.paint_uniform_color([0, 0.651, 0.929])
                
                # o3d.visualization.draw_geometries([pcd0, pcd1])
                # SAVE THE POINT CLOUDS
                # o3d.io.write_point_cloud(f"pcd0_{j-1}.ply", pcd0)

                print("After outlier removal:")
                print(f"Number of points in scene {j-1}: {np.asarray(pcd0.points).shape}")
                print(f"Number of points in scene {j}: {np.asarray(pcd1.points).shape}")

                # Estimate normals (only for visualization)
                pcd0.estimate_normals()
                pcd1.estimate_normals()

                pts0 = torch.tensor(np.asarray(pcd0.points)).float()
                pts1 = torch.tensor(np.asarray(pcd1.points)).float()

                print("Number of pts0: ", pts0.shape)
                print("Number of pts1: ", pts1.shape)

                _pcd0 = torch.tensor(np.asarray(pcd0.points)).float()
                _pcd1 = torch.tensor(np.asarray(pcd1.points)).float()

                print("Gedi descriptor extraction...")
                # Compute GeDi descriptors
                gedi0_desc = self.gedi.compute(pts=torch.tensor(valid_3d_pts_t[i][j-1]).float(), pcd=_pcd0)
                gedi1_desc = self.gedi.compute(pts=torch.tensor(valid_3d_pts_t[i][j]).float(), pcd=_pcd1)

                random_indices = np.random.choice(768, 64, replace=False)

                # Reduce DINO descriptors to 64 dimensions
                dino0_desc_reduced = dino0_desc[:, random_indices]  # (S0, 64)
                dino1_desc_reduced = dino1_desc[:, random_indices]  # (S1, 64)

                # Concatenate GeDi + DINO descriptors
                combined0_desc = np.hstack((gedi0_desc, dino0_desc_reduced))  # (S0, GeDi_dim + 768)
                combined1_desc = np.hstack((gedi1_desc, dino1_desc_reduced))  # (S1, GeDi_dim + 768)

                combined0_desc = combined0_desc / np.linalg.norm(combined0_desc, axis=1, keepdims=True)
                combined1_desc = combined1_desc / np.linalg.norm(combined1_desc, axis=1, keepdims=True)

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
                    max_correspondence_distance=.005,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                    ransac_n=3,
                    checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(.9),
                            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(.02)],
                    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000))


                # Refine with ICP
                icp_refinement_result = o3d.pipelines.registration.registration_icp(
                    _pcd0,  # Source point cloud
                    _pcd1,  # Target point cloud
                    max_correspondence_distance=0.005,  # Stricter threshold for ICP
                    init=est_result01.transformation,  # Use RANSAC transformation as the initial guess
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
                )


                # applying estimated transformation
                pcd0.transform(est_result01.transformation)
                dino_transformation[i].append(icp_refinement_result.transformation)
                current_cam2world = current_cam2world @ icp_refinement_result.transformation
                dino_obj2world[i].append(current_cam2world)
                print(f"Transformation between scene {j-1} and {j} is: ", icp_refinement_result.transformation)
        
        return dino_transformation, dino_obj2world