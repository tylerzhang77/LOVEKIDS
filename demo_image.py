"""
LOVEKIDS - Infant 3D Pose Estimation Demo
Process images and generate 3D mesh visualization with 2D keypoints overlay
"""

import torch
import cv2
import numpy as np
import os
import argparse
from collections import OrderedDict
from torchvision import transforms as T
from models.model_3dpose_image import ImageHRNet
from utils.transform import CamTransform
from utils.render_pytorch3d import render_mesh
from utils.config import update_config
from utils.vis import get_one_box
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm


def xyxy2xywh(bbox):
    """Convert bbox from [x1, y1, x2, y2] to [cx, cy, w, h] format"""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]


def process_single_image(
    img_path,
    output_dir,
    det_model,
    pose_model,
    transform,
    device
):
    """
    Process a single image for 3D pose estimation
    
    Args:
        img_path: Path to input image
        output_dir: Directory to save results
        det_model: Person detection model
        pose_model: 3D pose estimation model
        transform: Image transformation pipeline
        device: torch.device for computation
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load and preprocess image
        input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        if input_image is None:
            print(f"Warning: Could not read image {img_path}")
            return False
            
        # Run person detection
        det_transform = T.Compose([T.ToTensor()])
        det_input = det_transform(input_image).to(device)
        det_output = det_model([det_input])[0]

        # Get bounding box
        tight_bbox = get_one_box(det_output)
        if tight_bbox is None:
            print(f"Warning: No person detected in {img_path}")
            return False

        # Transform image for pose estimation
        pose_input, bbox, img_center = transform.test_transform(input_image, tight_bbox)
        pose_input = pose_input.to(device)[None, :, :, :]
        img_center = np.array(img_center).reshape(1, 2)

        # Run pose estimation
        with torch.no_grad():
            pose_output = pose_model(
                pose_input,
                bboxes=torch.from_numpy(np.array(bbox)).to(device).unsqueeze(0).float(),
                img_center=torch.from_numpy(img_center).to(device).float(),
                if_video=False
            )

        # Prepare for mesh rendering
        vertices = pose_output.pred_vertices.detach()
        transl = pose_output.transl.detach()
        smil_faces = torch.from_numpy(pose_model.smil_process.faces.astype(np.int32)).to(device)

        # Calculate focal length
        bbox_xywh = xyxy2xywh(bbox)
        focal = 1000.0
        focal = focal / 256 * bbox_xywh[2]

        # Render 3D mesh
        color_batch = render_mesh(
            vertices=vertices,
            faces=smil_faces,
            translation=transl,
            focal_length=focal,
            height=input_image.shape[0],
            width=input_image.shape[1]
        )

        # Post-process rendering (move all tensors to CPU)
        color_batch = color_batch.cpu()
        valid_mask_batch = (color_batch[:, :, :, [-1]] > 0)
        image_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
        image_vis_batch = (image_vis_batch * 255).numpy()

        # Overlay mesh on original image
        color = image_vis_batch[0]
        valid_mask = valid_mask_batch[0].numpy()
        alpha = 0.9
        
        image_vis = alpha * color[:, :, :3] * valid_mask + \
                    (1 - alpha) * input_image * valid_mask + \
                    (1 - valid_mask) * input_image

        # Calculate 2D keypoint projections
        Pc = 2.2 * pose_output.pred_xyz_24_recon.clone().reshape(-1, 24, 3).cpu() + \
             transl.unsqueeze(1).expand(-1, 24, 3).cpu()
        u = Pc[:, :, 0] / Pc[:, :, 2] * focal + img_center[:, 0]
        v = Pc[:, :, 1] / Pc[:, :, 2] * focal + img_center[:, 1]
        uv = torch.stack([u, v], dim=-1)

        # Draw 2D keypoints on mesh
        for i in range(uv.shape[1]):
            x, y = int(uv[0, i, 0].item()), int(uv[0, i, 1].item())
            cv2.circle(image_vis, (x, y), 4, (0, 255, 0), -1)
            cv2.circle(image_vis, (x, y), 4, (0, 0, 0), 1)

        # Save visualization result
        image_vis = image_vis.astype(np.uint8)
        image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)
        basename = os.path.basename(img_path)
        res_path = os.path.join(output_dir, f"res_{basename}")
        cv2.imwrite(res_path, image_vis)

        # Save metadata to text file
        info_path = os.path.join(output_dir, f"info_{basename[:-4]}.txt")
        with open(info_path, 'w') as f:
            f.write(f"2D Keypoint Coordinates (UV):\n{uv[0].cpu().numpy()}\n\n")
            f.write(f"3D Keypoint Coordinates (Camera Space):\n{Pc[0].cpu().numpy()}\n\n")
            f.write(f"Translation Vector:\n{transl.cpu().numpy()}\n\n")
            f.write(f"Focal Length: {focal}\n")
            f.write(f"Image Center: {img_center}\n")

        return True

    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def inference_image_folder(
    img_dir, 
    output_dir,
    ckpt_path,
    cfg_path,
    gpu=0
):
    """
    Process all images in a folder
    
    Args:
        img_dir: Input image directory
        output_dir: Output directory for results
        ckpt_path: Path to model checkpoint
        cfg_path: Path to configuration file
        gpu: GPU device ID
    """
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    
    # Load configuration
    cfg = update_config(cfg_path)

    # Initialize preprocessing pipeline
    bbox_3d_shape = [item * 1e-3 for item in getattr(cfg.TRANSFORM, 'BBOX_3D_SHAPE', (2000, 2000, 2000))]
    transform = CamTransform(
        joint_pairs_17=None,
        joint_pairs_24=None,
        joint_pairs_29=None,
        bbox_shape=bbox_3d_shape,
        scale_factor=cfg.TRANSFORM.SCALE_FACTOR,
        color_factor=cfg.TRANSFORM.COLOR_FACTOR,
        occlusion=cfg.TRANSFORM.OCCLUSION,
        input_size=cfg.TRANSFORM.IMAGE_SIZE,
        output_size=cfg.TRANSFORM.HEATMAP_SIZE,
        depth_dim=cfg.TRANSFORM.DEPTH_DIM,
        bbox_3d_shape=bbox_3d_shape,
        rot=cfg.TRANSFORM.ROT_FACTOR,
        sigma=cfg.TRANSFORM.SIGMA,
        train=False,
        add_dpg=False,
        loss_type=cfg.TRANSFORM.LOSS_TYPE
    )

    # Load detection model
    print("Loading person detection model...")
    det_model = fasterrcnn_resnet50_fpn(pretrained=True)
    det_model.to(device)
    det_model.eval()

    # Load 3D pose estimation model
    print(f"Loading 3D pose model from {ckpt_path}...")
    model = ImageHRNet(cfg)
    
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files
    image_files = [f for f in os.listdir(img_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_files:
        print(f"No image files found in {img_dir}")
        return

    print(f"Found {len(image_files)} images in {img_dir}")
    
    # Process each image
    successful = 0
    failed = 0
    
    for img_file in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(img_dir, img_file)
        if process_single_image(img_path, output_dir, det_model, model, transform, device):
            successful += 1
        else:
            failed += 1

    # Print summary
    print(f"\n{'='*60}")
    print(f"Processing completed:")
    print(f"  Successfully processed: {successful} images")
    print(f"  Failed to process: {failed} images")
    print(f"  Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LOVEKIDS - Infant 3D Pose Estimation Demo')
    parser.add_argument('--gpu', type=int, default=1,
                       help='GPU device ID (default: 0)')
    parser.add_argument('--img-dir', type=str, default='./examples/syrip',
                       help='Input image directory')
    parser.add_argument('--out-dir', type=str, default='./output/results',
                       help='Output directory for results')
    parser.add_argument('--ckpt-path', type=str, default='./saved_models/init_for_img_369_syrip.pth',
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--cfg-path', type=str, default='./config/config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    inference_image_folder(
        img_dir=args.img_dir,
        output_dir=args.out_dir,
        ckpt_path=args.ckpt_path,
        cfg_path=args.cfg_path,
        gpu=args.gpu
    )