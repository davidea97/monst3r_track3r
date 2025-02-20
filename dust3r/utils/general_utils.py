import glob
import os
import numpy as np
import yaml
import torch
from pytorch3d.loss import chamfer_distance
from typing import Optional
import cv2

SUPPORTED_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']


def generate_image_list(folder_path):
    """
    Generates a list of image paths organized by subfolders (e.g., camera1, camera2, etc.).

    Args:
        folder_path (str): The root folder containing subfolders named camera1, camera2, etc.

    Returns:
        list of lists: A list where each sublist contains paths to images from one camera folder.
    """
    # Get all subdirectories within the folder_path
    subfolders = sorted([
        os.path.join(folder_path, d) for d in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, d)) and d.startswith('camera')
    ])

    # Initialize the matrix (list of lists) to store images from each subfolder
    image_list = [[] for _ in range(len(subfolders))]

    for i, subfolder in enumerate(subfolders):
        # Find all .png files in the subfolder (adjust for other extensions if needed)
        image_paths = []
        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            image_paths.extend(glob.glob(os.path.join(os.path.join(subfolder, "image"), f"*{ext}")))

        # Normalize paths to use forward slashes for compatibility
        image_paths = [path.replace("\\", "/") for path in image_paths]

        # Sort the paths to ensure consistent ordering
        image_paths.sort()

        # Add the list of image paths from this subfolder to the matrix
        image_list[i] = image_paths

    return image_list, subfolders


def generate_mask_list(folder_path, image_list, image_ext=None):
    """
    Generates a list of mask image paths organized by subfolders (e.g., camera1, camera2, etc.),
    but only includes folders that contain a "mask" subfolder.

    Args:
        folder_path (str): The root folder containing subfolders named camera1, camera2, etc.

    Returns:
        list of lists: A list where each sublist contains paths to mask images from one camera folder.
    """
    
    # Get all subdirectories within the folder_path
    subfolders = sorted([
        os.path.join(folder_path, d) for d in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, d)) and d.startswith('camera')
    ])

    mask_list = [[] for _ in range(len(subfolders))]

    for i, subfolder in enumerate(subfolders):
        # Check if the "mask" subfolder exists
        mask_folder = os.path.join(subfolder, "masks")
        if not os.path.exists(mask_folder) or not os.path.isdir(mask_folder):
            print(f"Skipping: {mask_folder} (does not exist)")
            continue
        
        # Find all .png files in the "mask" subfolder (adjust for other extensions if needed)
        mask_paths = glob.glob(os.path.join(mask_folder, f"*{image_ext}"))

        mask_paths = [path.replace("\\", "/") for path in mask_paths]
        # Sort the paths to ensure consistent ordering
        mask_paths.sort()
        mask_paths = mask_paths[:len(image_list[i])]

        mask_list[i] = mask_paths

    return mask_list

def chamfer_loss(src_pts, tgt_pts):
    """
    Calcola la Chamfer Distance tra due nuvole di punti di dimensioni diverse.
    
    Args:
        src_pts (torch.Tensor): Nuvola di punti sorgente (N, 3)
        tgt_pts (torch.Tensor): Nuvola di punti target (M, 3)
    
    Returns:
        loss (torch.Tensor): Chamfer distance loss
    """
    loss, _ = chamfer_distance(src_pts.unsqueeze(0), tgt_pts.unsqueeze(0))  # Aggiunge batch dimension
    return loss

def to_torch(tensor, device):
    """Convert a TensorFlow tensor to a PyTorch tensor and move it to the correct device."""
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)  # Already a PyTorch tensor, just move to device
    elif isinstance(tensor, (list, tuple)):  
        return torch.tensor(tensor, dtype=torch.float32, device=device)  # Convert lists
    else:  
        return torch.tensor(tensor.numpy(), dtype=torch.float32, device=device)  # Convert TF tensor
    

def read_intrinsics(camera_folders, config, intrinsic_file="intrinsic_pars_file.yaml"):
    intrinsics = []
    dist_coeffs = []
    for camera_folder in camera_folders:
        with open(os.path.join(camera_folder, intrinsic_file), "r") as file:
            data = yaml.safe_load(file)

        target_image_size = config['image_size']
        img_width = data['img_width']
        img_height = data['img_height']
        original_size = max(img_width, img_height)
        scale_factor = target_image_size / original_size

        fx = data['fx']*scale_factor
        fy = data['fy']*scale_factor
        cx = data['cx']*scale_factor
        cy = data['cy']*scale_factor
        intrinsics.append({'focal': fx,       
                            'pp': (cx, cy)}
        )

        k0 = data['dist_k0']
        k1 = data['dist_k1']
        k2 = data['dist_k2']
        px = data['dist_px']
        py = data['dist_py']

        dist_coeffs.append(np.array([k0, k1, px, py, k2]))
    
    return intrinsics, dist_coeffs



def visualize_matches(
        image0: np.ndarray,
        image1: np.ndarray,
        kp0: np.ndarray,
        kp1: np.ndarray,
        match_matrix: np.ndarray,
        match_labels: Optional[np.ndarray] = None,
        show_keypoints: bool = False,
        highlight_unmatched: bool = False,
        title: Optional[str] = None,
        line_width: int = 1,
        circle_radius: int = 4,
        circle_thickness: int = 2,
        rng: Optional['np.random.Generator'] = None,
    ):
    """Generates visualization of keypoints and matches for two images.

    Stacks image0 and image1 horizontally. In case the two images have different
    heights, scales image1 (and its keypoints) to match image0's height. Note
    that keypoints must be in (x, y) format, NOT (row, col). If match_matrix
    includes unmatched dustbins, the dustbins will be removed before visualizing
    matches.

    Args:
        image0: (H, W, 3) array containing image0 contents.
        image1: (H, W, 3) array containing image1 contents.
        kp0: (N, 2) array where each row represents (x, y) coordinates of keypoints
        in image0.
        kp1: (M, 2) array, where each row represents (x, y) coordinates of keypoints
        in image1.
        match_matrix: (N, M) binary array, where values are non-zero for keypoint
        indices making up a match.
        match_labels: (N, M) binary array, where values are non-zero for keypoint
        indices making up a ground-truth match. When None, matches from
        'match_matrix' are colored randomly. Otherwise, matches from
        'match_matrix' are colored according to accuracy (compared to labels).
        show_keypoints: if True, all image0 and image1 keypoints (including
        unmatched ones) are visualized.
        highlight_unmatched: if True, highlights unmatched keypoints in blue.
        title: if not None, adds title text to top left of visualization.
        line_width: width of correspondence line, in pixels.
        circle_radius: radius of keypoint circles, if visualized.
        circle_thickness: thickness of keypoint circles, if visualized.
        rng: np random number generator to generate the line colors.

    Returns:
        Numpy array of image0 and image1 side-by-side, with lines between matches
        according to match_matrix. If show_keypoints is True, keypoints from both
        images are also visualized.
    """
    # initialize RNG
    if rng is None:
        rng = np.random.default_rng()

    # Make copy of input param that may be modified in this function.
    kp1 = np.copy(kp1)

    # Detect unmatched dustbins.
    has_unmatched_dustbins = (match_matrix.shape[0] == kp0.shape[0] + 1) and (
        match_matrix.shape[1] == kp1.shape[0] + 1
    )

    # If necessary, resize image1 so that the pair can be stacked horizontally.
    height0 = image0.shape[0]
    height1 = image1.shape[0]
    if height0 != height1:
        scale_factor = height0 / height1
        if scale_factor <= 1.0:
            interp_method = cv2.INTER_AREA
        else:
            interp_method = cv2.INTER_LINEAR
        new_dim1 = (int(image1.shape[1] * scale_factor), height0)
        image1 = cv2.resize(image1, new_dim1, interpolation=interp_method)
        kp1 *= scale_factor

    # Create side-by-side image and add lines for all matches.
    viz = cv2.hconcat([image0, image1])

    w0 = image0.shape[1]
    matches = np.argwhere(
        match_matrix[:-1, :-1] if has_unmatched_dustbins else match_matrix
    )
    for match in matches:
        pt0 = (int(kp0[match[0], 0]), int(kp0[match[0], 1]))
        pt1 = (int(kp1[match[1], 0] + w0), int(kp1[match[1], 1]))
        if match_labels is None:
            color = tuple(rng.integers(0, 255, size=3).tolist())
        else:
            if match_labels[match[0], match[1]]:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
        cv2.line(viz, pt0, pt1, color, line_width)

    # Optionally, add circles to output image to represent each keypoint.
    if show_keypoints:
        for i in range(np.shape(kp0)[0]):
            kp = kp0[i, :]
            if highlight_unmatched and has_unmatched_dustbins and match_matrix[i, -1]:
                cv2.circle(
                    viz,
                    tuple(kp.astype(np.int32).tolist()),
                    circle_radius,
                    (255, 0, 0),
                    circle_thickness,
                )
            else:
                cv2.circle(
                    viz,
                    tuple(kp.astype(np.int32).tolist()),
                    circle_radius,
                    (0, 0, 255),
                    circle_thickness,
                )
        for j in range(np.shape(kp1)[0]):
            kp = kp1[j, :]
            kp[0] += w0
            if highlight_unmatched and has_unmatched_dustbins and match_matrix[-1, j]:
                cv2.circle(
                    viz,
                    tuple(kp.astype(np.int32).tolist()),
                    circle_radius,
                    (255, 0, 0),
                    circle_thickness,
                )
            else:
                cv2.circle(
                    viz,
                    tuple(kp.astype(np.int32).tolist()),
                    circle_radius,
                    (0, 0, 255),
                    circle_thickness,
                )
    if title is not None:
        viz = cv2.putText(
            viz,
            title,
            (5, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return viz