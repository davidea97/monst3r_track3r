import torch
import numpy as np
import open3d as o3d
from gedi.gedi import GeDi
from dust3r.utils.vo_eval import save_trajectory_tum_format
from dust3r.cloud_opt.base_opt import c2w_to_tumpose
from transformers import AutoProcessor, AutoModel
from PIL import Image

class ObjectTracker:
    def __init__(self, all_3d_obj_pts, obj_msks, imagelist):
        self.all_3d_obj_pts = all_3d_obj_pts
        self.obj_msks = obj_msks
        self.imagelist = imagelist
        
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

    def _extract_dino_features_from_mask(self, image_array, bboxes, mask):
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
        all_valid_patches = []  # List to store selected patch positions

        # Ensure mask is uint8 (values in [0,255] for correct visualization)
        mask = (mask.astype(np.float32) / mask.max() * 255).astype(np.uint8)  # Normalize & scale

        # # Convert to PIL Image (grayscale mode 'L')
        # mask_image = Image.fromarray(mask, mode="L")

        # # Save the mask
        # mask_image.save("cropped_mask.png")
        

        for j, bbox in enumerate(bboxes):
            x_min, y_min, x_max, y_max = bbox

            # Crop the image and mask to the bounding box region
            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            cropped_mask = mask[y_min:y_max, x_min:x_max]   # The value are between 0 and 255
            # print(f"Unique values in cropped mask {j}: {np.unique(cropped_mask)}")



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


            # Process cropped image for DINO
            inputs = self.processor(images=cropped_image, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model(**inputs)

            patch_features = outputs.last_hidden_state[:, 1:, :]  # Patch embeddings (spatial features)
            print(f"Patch features shape: {patch_features.shape}")

            # Resize mask to match the 16x16 patch grid
            mask_resized = np.array(Image.fromarray(cropped_mask).resize((16, 16)))  
            mask_flattened = mask_resized.flatten()  # Flatten mask

            # Select only patches within the object
            valid_patches = torch.tensor(mask_flattened > 0, dtype=torch.bool)

            # Extract features for valid patches
            object_patch_features = patch_features[:, valid_patches, :]

            all_features.append(object_patch_features.squeeze(0))  # Store features
            all_valid_patches.append(valid_patches)  # Store valid patch locations

        return all_features, all_valid_patches

    def _get_dino_patch_features(self):
        """Extracts DINO features from all images in the imagelist."""
        dino_patch_features = []
        bbox_vec = []
        valid_patches_vec = []
        for i, image_path in enumerate(self.imagelist):
            # _, patch_features = self.extract_dino_features(image_path)
            bboxes = self.get_bboxes_from_mask(self.obj_msks[i])

            # TODO: first step directly add a square box 224x224, then a square box smaller fitting the object, at the end try with bbox from mask
            print(f"Bboxes for image {i}: {bboxes}")
            obj_patch_features, valid_patches = self._extract_dino_features_from_mask(image_path, bboxes, self.obj_msks[i])
            valid_patches_vec.append(valid_patches)
            dino_patch_features.append(obj_patch_features)
            # print(f"Patch Features Shape: {obj_patch_features.shape}")
        return dino_patch_features, bbox_vec, valid_patches_vec
    
    def _get_dino_cls_embeddings(self):
        """Extracts DINO CLS embeddings from all images in the imagelist."""
        dino_cls_embeddings = []
        for image_path in self.imagelist:
            cls_embedding, _ = self.extract_dino_features(image_path)
            dino_cls_embeddings.append(cls_embedding)
            # print(f"CLS Embedding Shape: {cls_embedding.shape}")
        return dino_cls_embeddings

    def map_dino_features_to_3d(self, dino_patch_features, valid_patches, pts3d, bbox):
        """
        Assigns valid DINO features to the corresponding 3D points using the same mask.

        Parameters:
            dino_patch_features (torch.Tensor): Extracted DINO features (N_valid_patches, 768).
            valid_patches (torch.Tensor): Boolean mask for selected patches (256,).
            pts3d (numpy.array): 3D points of the full scene (H, W, 3).
            bbox (tuple): Bounding box (x_min, y_min, x_max, y_max).

        Returns:
            object_3d_pts (numpy.array): 3D positions for the valid DINO patches (N_valid_patches, 3).
            object_3d_features (numpy.array): Corresponding DINO features (N_valid_patches, 768).
        """
        x_min, y_min, x_max, y_max = bbox

        # Extract the cropped 3D points using the same BBox
        pts3d_cropped = pts3d[y_min:y_max, x_min:x_max]  # Shape: (H_bbox, W_bbox, 3)

        # Resize the cropped 3D region to match the 16×16 grid
        h_cropped, w_cropped, _ = pts3d_cropped.shape
        resized_3d_pts = np.array(Image.fromarray(pts3d_cropped.reshape(h_cropped, w_cropped, 3))
                                .resize((16, 16), resample=Image.BILINEAR))  # Resize to match patch grid

        # Flatten the resized 3D points (now a 16×16 = 256 grid)
        flattened_3d_pts = resized_3d_pts.reshape(-1, 3)  # Shape: (256, 3)

        # Use `valid_patches` to filter only the relevant 3D points
        object_3d_pts = flattened_3d_pts[valid_patches.cpu().numpy()]

        # Ensure a one-to-one mapping between DINO patches and 3D points
        assert object_3d_pts.shape[0] == dino_patch_features.shape[0], "Mismatch between DINO features and 3D points!"

        # Convert DINO features to NumPy
        object_3d_features = dino_patch_features.cpu().numpy()

        return object_3d_pts, object_3d_features
    
    
    def _get_gedi_obj_track(self):
        print(">> Starting GeDi tracking...")
        gedi_transformation = [[] for _ in range(self._get_object_quantity())]
        gedi_obj2world = [[] for _ in range(self._get_object_quantity())]
        for i, obj in enumerate(self._get_all_3d_object_pts()):
            # Compute tracking for object i
            print(">> Tracking object ", i)
            # Compute centroid of the first object
            pcd0_pts = np.vstack(obj[0])  # First point cloud
            obj_centroid = np.mean(pcd0_pts, axis=0)

            # Create identity transformation with translation to centroid
            identity_transform = np.eye(4)
            identity_transform[:3, 3] = obj_centroid  # Set translation

            # Insert identity transformation at the start
            gedi_transformation[i].append(identity_transform)

            # TODO: add the first frame initializations, for example PCA, etc..
            gedi_obj2world[i].append(identity_transform)
            current_cam2world = identity_transform
            print(f"Initial transformation (Identity at centroid) for object {i}:")
            print(identity_transform)

            for j in range(1, len(obj)):

                pcd0_pts = np.vstack(obj[j-1])  # Flatten list of numpy arrays
                pcd1_pts = np.vstack(obj[j])

                # Convert to Open3D PointCloud
                pcd0 = o3d.geometry.PointCloud()
                pcd0.points = o3d.utility.Vector3dVector(pcd0_pts)

                pcd1 = o3d.geometry.PointCloud()
                pcd1.points = o3d.utility.Vector3dVector(pcd1_pts)

                # Estimate normals (only for visualization)
                pcd0.estimate_normals()
                pcd1.estimate_normals()

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
                pcd0 = pcd0.voxel_down_sample(self.voxel_size)
                pcd1 = pcd1.voxel_down_sample(self.voxel_size)

                # Perform ICP registration
                icp_result = o3d.pipelines.registration.registration_icp(
                    pcd0, 
                    pcd1, 
                    max_correspondence_distance=0.02,  # Correspondence threshold
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