from scipy.spatial.transform import Rotation
import numpy as np
import trimesh
from dust3r.utils.device import to_numpy
import torch
import os
import cv2
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from third_party.raft import load_RAFT
from datasets_preprocess.sintel_get_dynamics import compute_optical_flow
from dust3r.utils.flow_vis import flow_to_image

def convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05, show_cam=True,
                                 cam_color=None, as_pointcloud=False, transparent_cams=False, silent=False, 
                                 save_name=None, all_object_pts3d=None, all_msk_obj=None, tracking_transformation=None):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)
        
    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        valid_msk = np.isfinite(pts.sum(axis=1))
        valid_pts = pts[valid_msk]
        valid_col = col[valid_msk]
        if all_object_pts3d is not None:
            for i, object_pts3d in enumerate(all_object_pts3d):
                pts_obj = np.concatenate([p for p in object_pts3d])
                
                random_color = np.random.rand(3)  # Random color for the object
                obj_color_mask = np.isin(valid_pts, pts_obj, assume_unique=False).all(axis=1)
                # valid_col[obj_color_mask] = [1, 0, 0]  # Red color for object points
                valid_col[obj_color_mask] = random_color  # Random color for object points

                if tracking_transformation is not None:

                    previous_transform = np.eye(4)  # Start with identity
                    for j, pts_3d_obj in enumerate(object_pts3d):
                        obj_frame = trimesh.creation.axis(origin_size=0.005, axis_length=0.1)
                        # Compute the new centroid by applying the transformation to the previous centroid
                        obj_transform = tracking_transformation[i][j]
                        
                        obj_frame.apply_transform(obj_transform)
                        scene.add_geometry(obj_frame)

                        # Update the previous transform to be used in the next iteration
                        previous_transform = obj_transform

                    # for j, pts_3d_obj in enumerate(object_pts3d):
                    #     obj_frame = trimesh.creation.axis(origin_size=0.005, axis_length=0.1)
                    #     obj_transform = np.eye(4)
                    #     # Compute the new centroid by applying the transformation to the previous centroid
                    #     if j == 0:
                    #         obj_transform = previous_transform @ tracking_transformation[i][j]
                    #         obj_frame.apply_transform(obj_transform)
                    #         scene.add_geometry(obj_frame)

                    #         # Update the previous transform to be used in the next iteration
                    #         previous_transform = obj_transform
                    #     else:
                    #         obj_transform = previous_transform @ tracking_transformation[i][j]
                    #         obj_frame.apply_transform(obj_transform)
                    #         scene.add_geometry(obj_frame)
                        
                        
            pct_updated = trimesh.PointCloud(valid_pts, colors=valid_col)
            scene.add_geometry(pct_updated)
        else:
            pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
            scene.add_geometry(pct)
        # scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    if show_cam:
        for i, pose_c2w in enumerate(cams2world):
            if isinstance(cam_color, list):
                camera_edge_color = cam_color[i]
            else:
                camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
            add_scene_cam(scene, pose_c2w, camera_edge_color,
                        None if transparent_cams else imgs[i], focals[i],
                        imsize=imgs[i].shape[1::-1], screen_width=cam_size)
            
            # Add camera reference frame
            if i == 0:
                cam_axis = trimesh.creation.axis(origin_size=0.01, axis_length=0.1)
                cam_axis.apply_transform(pose_c2w)
                scene.add_geometry(cam_axis)

    # Add world reference frame
    # world_axis = trimesh.creation.axis(origin_size=0.01, axis_length=0.1)
    # scene.add_geometry(world_axis)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    
    if save_name is None: save_name='scene'
    outfile = os.path.join(outdir, save_name+'.glb')
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile

def get_dynamic_mask_from_pairviewer(scene, flow_net=None, both_directions=False, output_dir='./demo_tmp', motion_mask_thre=0.35):
    """
    get the dynamic mask from the pairviewer
    """
    if flow_net is None:
        # flow_net = load_RAFT(model_path="third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth").to('cuda').eval() # sea-raft
        flow_net = load_RAFT(model_path="third_party/RAFT/models/raft-things.pth").to('cuda').eval()

    imgs = scene.imgs
    img1 = torch.from_numpy(imgs[0]*255).permute(2,0,1)[None]                       # (B, 3, H, W)
    img2 = torch.from_numpy(imgs[1]*255).permute(2,0,1)[None]
    with torch.no_grad():
        forward_flow = flow_net(img1.cuda(), img2.cuda(), iters=20, test_mode=True)[1]  # (B, 2, H, W)
        if both_directions:
            backward_flow = flow_net(img2.cuda(), img1.cuda(), iters=20, test_mode=True)[1]
            
    B, _, H, W = forward_flow.shape

    depth_map1 = scene.get_depthmaps()[0] # (H, W)
    depth_map2 = scene.get_depthmaps()[1]

    im_poses = scene.get_im_poses()
    cam1 = im_poses[0]                  # (4, 4)   cam2world
    cam2 = im_poses[1]
    extrinsics1 = torch.linalg.inv(cam1) # (4, 4)   world2cam
    extrinsics2 = torch.linalg.inv(cam2)

    intrinsics = scene.get_intrinsics()
    intrinsics_1 = intrinsics[0]        # (3, 3)
    intrinsics_2 = intrinsics[1]

    ego_flow_1_2 = compute_optical_flow(depth_map1, depth_map2, extrinsics1, extrinsics2, intrinsics_1, intrinsics_2) # (H*W, 2)
    ego_flow_1_2 = ego_flow_1_2.reshape(H, W, 2).transpose(2, 0, 1) # (2, H, W)

    error_map = np.linalg.norm(ego_flow_1_2 - forward_flow[0].cpu().numpy(), axis=0) # (H, W)

    error_map_normalized = (error_map - error_map.min()) / (error_map.max() - error_map.min())
    error_map_normalized_int = (error_map_normalized * 255).astype(np.uint8)
    if both_directions:
        ego_flow_2_1 = compute_optical_flow(depth_map2, depth_map1, extrinsics2, extrinsics1, intrinsics_2, intrinsics_1)
        ego_flow_2_1 = ego_flow_2_1.reshape(H, W, 2).transpose(2, 0, 1)
        error_map_2 = np.linalg.norm(ego_flow_2_1 - backward_flow[0].cpu().numpy(), axis=0)
        error_map_2_normalized = (error_map_2 - error_map_2.min()) / (error_map_2.max() - error_map_2.min())
        error_map_2_normalized = (error_map_2_normalized * 255).astype(np.uint8)
        cv2.imwrite(f'{output_dir}/dynamic_mask_bw.png', cv2.applyColorMap(error_map_2_normalized, cv2.COLORMAP_JET))
        np.save(f'{output_dir}/dynamic_mask_bw.npy', error_map_2)

        backward_flow = backward_flow[0].cpu().numpy().transpose(1, 2, 0)
        np.save(f'{output_dir}/backward_flow.npy', backward_flow)
        flow_img = flow_to_image(backward_flow)
        cv2.imwrite(f'{output_dir}/backward_flow.png', flow_img)

    cv2.imwrite(f'{output_dir}/dynamic_mask.png', cv2.applyColorMap(error_map_normalized_int, cv2.COLORMAP_JET))
    error_map_normalized_bin = (error_map_normalized > motion_mask_thre).astype(np.uint8)
    # save the binary mask
    cv2.imwrite(f'{output_dir}/dynamic_mask_binary.png', error_map_normalized_bin*255)
    # save the original one as npy file
    np.save(f'{output_dir}/dynamic_mask.npy', error_map)

    # also save the flow
    forward_flow = forward_flow[0].cpu().numpy().transpose(1, 2, 0)
    np.save(f'{output_dir}/forward_flow.npy', forward_flow)
    # save flow as image
    flow_img = flow_to_image(forward_flow)
    cv2.imwrite(f'{output_dir}/forward_flow.png', flow_img)

    return error_map