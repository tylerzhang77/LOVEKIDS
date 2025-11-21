"""
LOVEKIDS - 3D Pose Estimation Model (Image-based)
HRNet backbone + SMIL parametric model for infant pose estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from easydict import EasyDict as edict
from .hrnet.hrnet import hrnet_load
from .smil.SMIL import SMIL_layer


def norm_heatmap(norm_type, heatmap, tau=5, sample_num=1):
    """
    Normalize heatmap using different strategies
    
    Args:
        norm_type: Normalization type ('softmax', 'sampling', 'multiple_sampling')
        heatmap: Input heatmap tensor [N, C, ...]
        tau: Temperature for Gumbel-Softmax
        sample_num: Number of samples for multiple sampling
    
    Returns:
        Normalized heatmap
    """
    shape = heatmap.shape
    
    if norm_type == 'softmax':
        heatmap = heatmap.reshape(*shape[:2], -1)
        heatmap = F.softmax(heatmap, 2)
        return heatmap.reshape(*shape)
    
    elif norm_type == 'sampling':
        heatmap = heatmap.reshape(*shape[:2], -1)
        eps = torch.rand_like(heatmap)
        log_eps = torch.log(-torch.log(eps))
        gumbel_heatmap = heatmap - log_eps / tau
        gumbel_heatmap = F.softmax(gumbel_heatmap, 2)
        return gumbel_heatmap.reshape(*shape)
    
    elif norm_type == 'multiple_sampling':
        heatmap = heatmap.reshape(*shape[:2], 1, -1)
        eps = torch.rand(*heatmap.shape[:2], sample_num, heatmap.shape[3], device=heatmap.device)
        log_eps = torch.log(-torch.log(eps))
        gumbel_heatmap = heatmap - log_eps / tau
        gumbel_heatmap = F.softmax(gumbel_heatmap, 3)
        gumbel_heatmap = gumbel_heatmap.reshape(shape[0], shape[1], sample_num, shape[2])
        return gumbel_heatmap.transpose(1, 2)
    
    else:
        raise NotImplementedError(f"Normalization type '{norm_type}' not implemented")


def heatmap_to_kpts(heatmaps, shift=True):
    """
    Convert 3D heatmaps to 3D keypoint coordinates using soft-argmax
    
    Args:
        heatmaps: 3D heatmap tensor [B, K, D, H, W]
        shift: Whether to shift coordinates to [-0.5, 0.5] range
    
    Returns:
        3D keypoint coordinates [B, K, 3]
    """
    B, K, D, H, W = heatmaps.shape
    
    # Precompute range tensors
    range_w = torch.arange(W, dtype=torch.float32, device=heatmaps.device).unsqueeze(-1)
    range_h = torch.arange(H, dtype=torch.float32, device=heatmaps.device).unsqueeze(-1)
    range_d = torch.arange(D, dtype=torch.float32, device=heatmaps.device).unsqueeze(-1)
    
    # Compute weighted average coordinates
    def compute_coord(hm, range_tensor):
        return hm.matmul(range_tensor).squeeze(-1)

    # Marginalize heatmaps along each dimension
    hm_x = heatmaps.sum(dim=(2, 3))  # [B, K, W]
    hm_y = heatmaps.sum(dim=(2, 4))  # [B, K, H]
    hm_z = heatmaps.sum(dim=(3, 4))  # [B, K, D]
    
    # Compute coordinates
    coord_x = compute_coord(hm_x, range_w) / W - 0.5
    coord_y = compute_coord(hm_y, range_h) / H - 0.5
    coord_z = compute_coord(hm_z, range_d) / D - 0.5

    # Stack to form 3D keypoints (U, V, D)
    uvd_kpts = torch.stack((coord_x, coord_y, coord_z), dim=2)

    return uvd_kpts


class HRNetEncoder(nn.Module):
    """
    HRNet-based encoder for extracting 2D keypoints and feature vectors
    """
    def __init__(self, config):
        super(HRNetEncoder, self).__init__()
        
        self.num_joints = config['NUM_KPTS']
        self.norm_type = config['NORM_TYPE']
        self.depth_dim = config['DEPTH_DIM']
        self.height_dim = config['HEIGHT_DIM']
        self.width_dim = config['WIDTH_DIM']
        self.smil_dtype = torch.float32
        
        self.backbone = hrnet_load(
            config['HRNET_TYPE'],
            num_joints=self.num_joints,
            depth_dim=self.depth_dim,
            is_train=True,
            generate_feat=True,
            generate_hm=True
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input image tensor [B, C, H, W]
        
        Returns:
            uvd_kpts: 3D keypoint coordinates [B, num_joints, 3]
            features: Feature vector [B, 2048]
        """
        B = x.shape[0]
        
        # Encode with HRNet
        out, features = self.backbone(x)
        
        # Reshape to 3D heatmap
        out = out.reshape(B, self.num_joints, self.depth_dim, self.height_dim, self.width_dim)
        out = out.reshape(B, self.num_joints, -1)
        
        # Normalize heatmap
        heatmaps = norm_heatmap(self.norm_type, out)
        heatmaps = heatmaps.reshape(B, self.num_joints, self.depth_dim, self.height_dim, self.width_dim)
        
        # Convert heatmap to keypoints
        uvd_kpts = heatmap_to_kpts(heatmaps)
        uvd_kpts = uvd_kpts.view(B, self.num_joints, 3)
        
        return uvd_kpts, features


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer-based models
    """
    def __init__(self, d_model, dropout, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Ensure d_model is even
        if d_model % 2 != 0:
            d_model += 1

        # Compute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """Add positional encoding to input"""
        x = x + self.pe[:, :x.size(1), :x.size(2)].to(x.device)
        return self.dropout(x)


class ImageHRNet(nn.Module):
    """
    Complete 3D pose estimation model for single images
    
    Pipeline:
        1. Extract 2D keypoints and features using HRNet
        2. Predict camera parameters, shape, and pose
        3. Reconstruct 3D pose using SMIL parametric model
    """
    def __init__(self, cfg):
        super(ImageHRNet, self).__init__()
        
        # Configuration
        self.focal_length = cfg['VIDEO']['FOCAL_LENGTH']
        self.input_size = cfg['VIDEO']['INPUT_SIZE']
        self.depth_factor = cfg['VIDEO']['DEPTH_FACTOR']
        self.smil_dtype = torch.float32
        
        # HRNet encoder
        self.hrnet_encoder = HRNetEncoder(cfg.HRNET)
        
        # Regression heads
        self.cam_projection = nn.Linear(2048, 1)      # Camera distance
        self.shape_projection = nn.Linear(2048, 20)   # Shape parameters (beta)
        self.phi_projection = nn.Linear(2048, 23 * 2) # Pose parameters (phi)
        
        # Initialize camera parameter
        init_cam = torch.tensor([0.9])
        self.register_buffer('init_cam', torch.Tensor(init_cam).float())
        
        # SMIL parametric model
        self.smil_process = SMIL_layer(
            cfg['VIDEO']['SMIL_FILE'],
            h36m_jregressor=np.load(cfg['VIDEO']['J_REGRESSOR']),
            dtype=self.smil_dtype
        )
        
    def forward(self, x, **kwargs):
        """
        Forward pass
        
        Args:
            x: Input image tensor [B, C, H, W]
            **kwargs: Additional inputs
                - bboxes: Bounding boxes [B, 4]
                - img_center: Image center [B, 2]
                - if_video: Whether processing video (default: False)
        
        Returns:
            edict containing:
                - pred_uvd_29: Predicted 2D+depth keypoints
                - pred_xyz_29: Predicted 3D keypoints (root-relative)
                - pred_shape: Shape parameters
                - pred_phi: Pose parameters
                - pred_theta_mats: Rotation matrices
                - pred_vertices: SMIL mesh vertices
                - cam_scale: Camera scale
                - cam_root: Camera space root position
                - transl: Translation vector
        """
        B, C, H, W = x.shape

        # Extract 2D keypoints and features
        uvd_29, features_2048 = self.hrnet_encoder(x)
    
        # Predict camera parameters
        init_cam = self.init_cam.expand(B, -1)
        pred_camera = self.cam_projection(features_2048).reshape(B, -1) + init_cam
        
        # Predict shape and pose
        pred_shape = self.shape_projection(features_2048).reshape(B, -1)
        pred_phi = self.phi_projection(features_2048).reshape(B, 23, 2)
        
        # Calculate camera depth
        cam_scale = pred_camera[:, :1].unsqueeze(1)
        cam_depth = self.focal_length / (self.input_size * cam_scale + 1e-9)

        # Convert 2D+depth to 3D
        xyz_29 = torch.zeros_like(uvd_29)
        
        # Process bounding box information
        if 'bboxes' in kwargs.keys():
            bboxes = kwargs['bboxes']
            img_center = kwargs['img_center']
            
            # Calculate bbox center and size
            cx = (bboxes[:, 0] + bboxes[:, 2]) * 0.5 - img_center[:, 0]
            cy = (bboxes[:, 1] + bboxes[:, 3]) * 0.5 - img_center[:, 1]
            w = (bboxes[:, 2] - bboxes[:, 0])
            h = (bboxes[:, 3] - bboxes[:, 1])

            # Normalize bbox center
            cx = cx / w
            cy = cy / h
            bbox_center = torch.stack((cx, cy), dim=1).unsqueeze(dim=1)
        else:
            bbox_center = torch.zeros_like(uvd_29[:, :, :2])

        # Depth coordinate
        xyz_29[:, :, 2:] = uvd_29[:, :, 2:].clone()
        
        # Reconstruct XY coordinates in camera space
        xy_29_meter = ((uvd_29[:, :, :2] + bbox_center) * self.input_size / self.focal_length) * \
                      (xyz_29[:, :, 2:] * self.depth_factor + cam_depth) 
        
        xyz_29[:, :, :2] = xy_29_meter / self.depth_factor

        # Calculate camera space root position
        camera_root = xyz_29[:, 0, :] * self.depth_factor
        camera_root[:, 2] += cam_depth[:, 0, 0]
        
        # Convert to root-relative coordinates
        xyz_29 = xyz_29 - xyz_29[:, [0]]
        
        # SMIL inverse kinematics
        SMIL_output = self.smil_process.Inverse_Kinematics(
            pose_skeleton=xyz_29.type(self.smil_dtype) * self.depth_factor,
            betas=pred_shape.type(self.smil_dtype),
            phis=pred_phi.type(self.smil_dtype),
            global_orient=None,
            return_verts=True
        )
        
        # Extract SMIL outputs
        pred_vertices = SMIL_output.vertices.float()
        xyz_24_recon = SMIL_output.joints.float() / self.depth_factor
        xyz_17_recon = SMIL_output.joints_from_verts.float() / self.depth_factor
        pred_theta = SMIL_output.rot_mats.float().reshape(B, 24, 3, 3)

        # Calculate translation vector
        transl = camera_root - SMIL_output.joints.float().reshape(-1, 24, 3)[:, 0, :]

        # Prepare output
        output = edict(
            pred_uvd_29=uvd_29.reshape(B, -1),
            pred_xyz_29=xyz_29.reshape(B, -1),
            pred_uvd_ori=uvd_29.reshape(B, -1),
            pred_phi=pred_phi,
            pred_shape=pred_shape,
            pred_theta_mats=pred_theta,
            pred_xyz_24_recon=xyz_24_recon.reshape(B, -1),
            pred_xyz_17_recon=xyz_17_recon.reshape(B, -1),
            pred_vertices=pred_vertices,
            cam_scale=cam_scale[:, 0],
            cam_root=camera_root,
            transl=transl,
            pred_camera=pred_camera
        )
        
        return output