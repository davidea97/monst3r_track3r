import torch
import numpy as np
import open3d as o3d
from gedi.gedi import GeDi
from dust3r.utils.vo_eval import save_trajectory_tum_format
from dust3r.cloud_opt.base_opt import c2w_to_tumpose


class ObjectTracker:
    def __init__(self, all_3d_obj_pts, all_obj_msks):
        self.all_3d_obj_pts = all_3d_obj_pts
        self.all_obj_msks = all_obj_msks
        
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


    def _get_object_quantity(self):
        return len(self.all_3d_obj_pts)

    def _get_3d_object_pts(self, obj_id):
        return self.all_3d_obj_pts[obj_id]
    
    def _get_object_mask(self, obj_id):
        return self.all_obj_msks[obj_id]
    
    def _get_all_3d_object_pts(self):
        return self.all_3d_obj_pts
    
    def _get_all_object_masks(self):
        return self.all_obj_msks
    
    def get_obj_poses(self, obj2w):
        poses = obj2w
        tt = np.arange(len(poses)).astype(float)
        tum_poses = [c2w_to_tumpose(p) for p in poses]
        tum_poses = np.stack(tum_poses, 0)
        return [tum_poses, tt]
    
    def save_obj_poses(self, path, obj2w):
        traj = self.get_obj_poses(obj2w)
        save_trajectory_tum_format(traj, path)
        return traj[0] # return the poses
    
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