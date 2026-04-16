"""
Optical Flow Extraction for VideoJAM Framework
===============================================

This module implements optical flow extraction using RAFT and conversion to
RGB motion representation as described in the VideoJAM paper.

Motion Representation:
- Magnitude: m = min(1, sqrt(u^2 + v^2) / (0.15 * sqrt(H^2 + W^2)))
- Direction: a = arctan2(v, u)
- RGB encoding follows HSV color wheel where hue=direction, saturation=1, value=magnitude
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List
import os
from pathlib import Path

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: cv2 not available. Video I/O functions will not work.")


try:
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    from torchvision.utils import flow_to_image
    RAFT_AVAILABLE = True
    FLOW_TO_IMAGE_AVAILABLE = True
except ImportError:
    RAFT_AVAILABLE = False
    FLOW_TO_IMAGE_AVAILABLE = False
    print("Warning: torchvision.models.optical_flow not available. Install torchvision >= 0.13.0")


class RAFTFlowExtractor:
    """
    Extracts dense optical flow from video frames using RAFT.
    Converts flow fields to RGB motion representation for VideoJAM.
    """

    def __init__(self, device='cuda', model_weights=None):
        """
        Initialize RAFT flow extractor.

        Args:
            device: Device to run RAFT on ('cuda' or 'cpu')
            model_weights: Optional path to custom RAFT weights, otherwise uses pretrained
        """
        if not RAFT_AVAILABLE:
            raise RuntimeError("RAFT not available. Install torchvision >= 0.13.0")

        self.device = device

        # Load RAFT model
        if model_weights is None:
            weights = Raft_Large_Weights.DEFAULT
            self.model = raft_large(weights=weights, progress=True)
        else:
            self.model = raft_large(weights=None)
            self.model.load_state_dict(torch.load(model_weights))

        self.model = self.model.to(device)
        self.model.eval()

        # RAFT preprocessing transforms
        self.transforms = weights.transforms() if model_weights is None else None

    @torch.no_grad()
    def extract_flow(self, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
        """
        Extract optical flow between two consecutive frames.

        Args:
            frame1: First frame [B, C, H, W] in range [0, 1] or [0, 255]
            frame2: Second frame [B, C, H, W] in range [0, 1] or [0, 255]

        Returns:
            flow: Optical flow [B, 2, H, W] where flow[:, 0] is u and flow[:, 1] is v
        """
        # Ensure frames are float32
        if frame1.dtype != torch.float32 and frame1.dtype != torch.float16:
            frame1 = frame1.float()
            frame2 = frame2.float()

        # If frames are in [0, 255], convert to [0, 1]
        # transforms() expects [0, 1] input
        if frame1.max() > 1.5:
            frame1 = frame1 / 255.0
            frame2 = frame2 / 255.0

        # RAFT requires dimensions divisible by 8
        # Pad if necessary
        B, C, H, W = frame1.shape
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8

        if pad_h > 0 or pad_w > 0:
            frame1 = torch.nn.functional.pad(frame1, (0, pad_w, 0, pad_h), mode='replicate')
            frame2 = torch.nn.functional.pad(frame2, (0, pad_w, 0, pad_h), mode='replicate')

        # Apply RAFT preprocessing (expects [0, 1] input, normalizes internally)
        if self.transforms is not None:
            frame1, frame2 = self.transforms(frame1, frame2)

        # Extract flow
        flow_predictions = self.model(frame1.to(self.device), frame2.to(self.device))

        # RAFT returns a list of flow predictions at different iterations
        # We use the final (most refined) prediction
        flow = flow_predictions[-1]

        # Remove padding from flow if we added any
        if pad_h > 0 or pad_w > 0:
            flow = flow[:, :, :H, :W]

        return flow

    def extract_video_flow(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        Extract optical flow for an entire video sequence.

        Args:
            video_frames: Video tensor [B, T, C, H, W] or [T, C, H, W]

        Returns:
            flows: Flow tensor [B, T-1, 2, H, W] or [T-1, 2, H, W]
                  Note: T-1 because flow is between consecutive frames
        """
        if video_frames.ndim == 4:
            # Add batch dimension if not present
            video_frames = video_frames.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        B, T, C, H, W = video_frames.shape
        flows = []

        for t in range(T - 1):
            flow = self.extract_flow(video_frames[:, t], video_frames[:, t + 1])
            flows.append(flow)

        flows = torch.stack(flows, dim=1)  # [B, T-1, 2, H, W]

        if squeeze_output:
            flows = flows.squeeze(0)

        return flows

    def extract_videojam_motion(self,
                                 video_frames: torch.Tensor,
                                 sigma: float = 0.15,
                                 deadzone_px: float = 0.0,
                                 target_resolution: int = None) -> torch.Tensor:
        """
        Extract VideoJAM motion representation for an entire video sequence.

        This performs:
        1. RAFT flow extraction between consecutive frames (T-1 flows)
        2. VideoJAM normalization (resolution-aware)
        3. HSV-to-RGB encoding
        4. Temporal alignment (duplicate first frame to get T frames)

        Args:
            video_frames: Video tensor [B, T, C, H, W] or [T, C, H, W] in range [0, 1]
            sigma: VideoJAM normalization constant (default: 0.15)
            deadzone_px: Magnitude threshold to suppress noise (default: 0.05 px)
            target_resolution: Target training resolution (e.g., 256). If set, scales flow
                              vectors to match target resolution brightness.

        Returns:
            motion_rgb: RGB motion tensor [B, T, 3, H, W] or [T, 3, H, W]
                       Aligned with input video (same temporal length T)
        """
        if video_frames.ndim == 4:
            # Add batch dimension if not present
            video_frames = video_frames.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        B, T, C, H, W = video_frames.shape

        # 1. Extract flow: [B, T-1, 2, H, W]
        flows = self.extract_video_flow(video_frames)

        # 2. Convert to VideoJAM RGB motion: [B, T-1, 3, H, W]
        motion_rgb = flow_to_motion_rgb_videojam(
            flows,
            sigma=sigma,
            deadzone_px=deadzone_px,
            target_resolution=target_resolution
        )

        # 3. Temporal alignment: duplicate first frame to match length T
        # This aligns motion[t] with video[t]
        first_frame = motion_rgb[:, 0:1, :, :, :]  # [B, 1, 3, H, W]
        motion_rgb = torch.cat([first_frame, motion_rgb], dim=1)  # [B, T, 3, H, W]

        if squeeze_output:
            motion_rgb = motion_rgb.squeeze(0)

        return motion_rgb


def flow_to_motion_rgb_videojam(flow: torch.Tensor,
                                 sigma: float = 0.15,
                                 deadzone_px: float = 0.0,
                                 target_resolution: int = None,
                                 return_magnitude_angle: bool = False) -> torch.Tensor:
    """
    Convert optical flow to RGB motion representation following VideoJAM paper specification.

    This is NOT a visualization - it's a weak, normalized motion prior for diffusion training.
    Static scenes will produce BLACK output (this is CORRECT behavior).

    VideoJAM normalization (Eq. 5 from paper):
    - Magnitude: m = min(1, sqrt(u² + v²) / (σ · sqrt(H² + W²)))
      where σ = 0.15 (fixed constant from paper)
    - Direction: α = arctan2(v, u)
    - HSV encoding: Hue=direction, Saturation=1, Value=magnitude

    IMPORTANT: Use target_resolution to match training resolution!
    If you extract flow at 1600×900 but train at 256×256, set target_resolution=256.
    This scales the flow vectors so brightness matches what you'd get from 256×256 extraction.

    Args:
        flow: Optical flow tensor [..., 2, H, W] where flow[..., 0] is u, flow[..., 1] is v
        sigma: Normalization constant (default: 0.15 from VideoJAM paper)
        deadzone_px: Magnitude threshold in pixels to suppress RAFT noise (default: 0.0)
                     IMPORTANT: 0.0 is paper-faithful. Only use 0.05-0.1 if you see
                     static noise artifacts. Can kill subtle motion in low-res or
                     small-motion videos.
        target_resolution: Target training resolution (e.g., 256). If set, scales flow
                          vectors and uses target diagonal for normalization.
                          Use this to extract at high-res but get correct brightness for training!
        return_magnitude_angle: If True, also returns (magnitude, angle) tensors

    Returns:
        motion_rgb: RGB motion tensor [..., 3, H, W] in range [0, 1]
                   Will be DARK for static scenes (this is correct!)
    """
    # Extract u, v components
    u = flow[..., 0:1, :, :]  # [..., 1, H, W]
    v = flow[..., 1:2, :, :]  # [..., 1, H, W]

    H, W = flow.shape[-2:]

    # 1. Compute raw magnitude: ||d|| = sqrt(u² + v²)
    magnitude_raw = torch.sqrt(u * u + v * v)

    # Print flow statistics for debugging
    if magnitude_raw.numel() > 0:
        mag_flat = magnitude_raw.flatten()
        # Sample to avoid memory issues
        if mag_flat.numel() > 10000:
            indices = torch.randperm(mag_flat.numel())[:10000]
            mag_sample = mag_flat[indices]
        else:
            mag_sample = mag_flat
        print(f"[VideoJAM Flow] Raw magnitude stats: "
              f"mean={mag_sample.mean().item():.4f} px, "
              f"median={mag_sample.median().item():.4f} px, "
              f"max={mag_sample.max().item():.4f} px, "
              f"p95={torch.quantile(mag_sample, 0.95).item():.4f} px")

    # 2. Optional: Apply deadzone to suppress RAFT noise in static regions
    # Zero out u,v when magnitude is below threshold (so it affects final result)
    if deadzone_px > 0:
        mask = magnitude_raw < deadzone_px
        u = torch.where(mask, torch.zeros_like(u), u)
        v = torch.where(mask, torch.zeros_like(v), v)
        # Recompute magnitude after deadzone
        magnitude_raw = torch.sqrt(u * u + v * v)

    # 2.5. Optional: Scale flow vectors to match target training resolution
    # This allows extracting flow at high-res but getting correct brightness for training
    if target_resolution is not None:
        # Scale flow vectors proportionally
        # If video is 1600x900 and target is 256, scale factor is 256/900 = 0.284
        # This makes a 100px motion at 1600x900 become 28.4px, matching native 256x256
        scale_factor = target_resolution / min(H, W)
        u = u * scale_factor
        v = v * scale_factor

        # Recompute magnitude after scaling
        magnitude_raw = torch.sqrt(u * u + v * v)

        # Use target resolution's diagonal for normalization (assumes square)
        diagonal = target_resolution * (2 ** 0.5)

        print(f"[VideoJAM Flow] Scaling flow: {H}x{W} → {target_resolution}x{target_resolution}")
        print(f"[VideoJAM Flow] Scale factor: {scale_factor:.4f}")
        print(f"[VideoJAM Flow] Target diagonal: {diagonal:.2f} px")
    else:
        # Use actual resolution's diagonal
        diagonal = (H * H + W * W) ** 0.5

    # 3. VideoJAM normalization (Eq. 5): m = min(1, ||d|| / (σ · sqrt(H² + W²)))
    normalization_factor = sigma * diagonal
    magnitude_normalized = torch.clamp(magnitude_raw / (normalization_factor + 1e-8), 0.0, 1.0)

    print(f"[VideoJAM Flow] Resolution: {H}x{W}, Diagonal={diagonal:.2f}")
    print(f"[VideoJAM Flow] Norm factor (σ·diagonal): {normalization_factor:.2f} px, σ={sigma}")
    print(f"[VideoJAM Flow] Normalized magnitude: "
          f"mean={magnitude_normalized.mean().item():.4f}, "
          f"max={magnitude_normalized.max().item():.4f}")

    # 4. Compute direction: α = arctan2(v, u)
    angle = torch.atan2(v, u)  # Range: [-π, π]

    # 5. Convert to HSV:
    # - Hue = (α + π) / (2π)  [maps -π,π to 0,1]
    # - Saturation = 1 (constant, full saturation)
    # - Value = m (normalized magnitude)
    hue = (angle + np.pi) / (2 * np.pi)  # [..., 1, H, W] in [0, 1]
    saturation = torch.ones_like(magnitude_normalized)  # S = 1 (constant)
    value = magnitude_normalized  # V = normalized magnitude

    # Stack to HSV: [..., 3, H, W]
    hsv = torch.cat([hue, saturation, value], dim=-3)

    # 6. Convert HSV to RGB
    motion_rgb = hsv_to_rgb_torch(hsv)

    if return_magnitude_angle:
        return motion_rgb, magnitude_normalized, angle
    else:
        return motion_rgb


def flow_to_motion_rgb(flow: torch.Tensor,
                       fixed_clip_px: float = 10.0,
                       deadzone: float = 0.0,
                       return_magnitude_angle: bool = False) -> torch.Tensor:
    """
    Stable flow->RGB with fixed pixel clip for consistent visualization.

    NOTE: This is for VISUALIZATION, not VideoJAM training!
    For VideoJAM training, use flow_to_motion_rgb_videojam() instead.

    HSV mapping:
    - Hue = angle (direction)
    - Saturation = 1 (constant, vivid colors)
    - Value = normalized magnitude (brightness)

    Uses fixed clip in pixels/frame for stable, visible flow across videos.

    Args:
        flow: Optical flow tensor [..., 2, H, W] where flow[..., 0] is u, flow[..., 1] is v
        fixed_clip_px: Fixed magnitude clip in pixels/frame (default: 10.0)
                      0 px → black, fixed_clip_px → full brightness
        deadzone: Magnitude threshold below which flow is set to zero (default: 0.0)
        return_magnitude_angle: If True, also returns (magnitude, angle) tensors

    Returns:
        motion_rgb: RGB motion tensor [..., 3, H, W] in range [0, 1]
    """
    # Extract u, v components
    u = flow[..., 0:1, :, :]  # [..., 1, H, W]
    v = flow[..., 1:2, :, :]  # [..., 1, H, W]

    H, W = flow.shape[-2:]

    # Compute magnitude
    magnitude_raw = torch.sqrt(u * u + v * v)

    # Print flow statistics for debugging (sample to avoid memory issues)
    if magnitude_raw.numel() > 0:
        mag_flat = magnitude_raw.flatten()
        # Sample 10k elements if tensor is large
        if mag_flat.numel() > 10000:
            indices = torch.randperm(mag_flat.numel())[:10000]
            mag_sample = mag_flat[indices]
        else:
            mag_sample = mag_flat
        print(f"Flow magnitude stats: mean={mag_sample.mean().item():.3f}, "
              f"median={mag_sample.median().item():.3f}, "
              f"max={mag_sample.max().item():.3f}, "
              f"p95={torch.quantile(mag_sample, 0.95).item():.3f} px/frame")

    # Apply deadzone to RAW magnitude (in pixels): suppress tiny noise
    if deadzone > 0:
        magnitude_raw = torch.where(magnitude_raw < deadzone, torch.zeros_like(magnitude_raw), magnitude_raw)

    # Fixed clip normalization: stable across videos
    # 0 px → 0, fixed_clip_px → 1.0
    magnitude = torch.clamp(magnitude_raw / fixed_clip_px, 0.0, 1.0)

    # Compute angle
    angle = torch.atan2(v, u)  # Range: [-pi, pi]

    # Convert to HSV (stable mapping):
    # H = angle [0, 1]
    # S = 1 (constant saturation for vivid colors)
    # V = magnitude (normalized magnitude)
    hue = (angle + np.pi) / (2 * np.pi)  # [..., 1, H, W] in [0, 1]
    saturation = torch.ones_like(magnitude)  # S = 1 (constant)
    value = magnitude  # V = magnitude

    # Stack to HSV
    hsv = torch.cat([hue, saturation, value], dim=-3)  # [..., 3, H, W]

    # Convert HSV to RGB using torch implementation
    motion_rgb = hsv_to_rgb_torch(hsv)

    if return_magnitude_angle:
        return motion_rgb, magnitude, angle
    else:
        return motion_rgb


def _get_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    """Create a 2D Gaussian kernel for spatial smoothing."""
    x = torch.arange(kernel_size).float() - kernel_size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    kernel = gauss[:, None] * gauss[None, :]
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)


def hsv_to_rgb_torch(hsv: torch.Tensor) -> torch.Tensor:
    """
    Convert HSV color space to RGB using PyTorch operations.

    Args:
        hsv: HSV tensor [..., 3, H, W] with values in [0, 1]
             hsv[..., 0, :, :] = Hue
             hsv[..., 1, :, :] = Saturation
             hsv[..., 2, :, :] = Value

    Returns:
        rgb: RGB tensor [..., 3, H, W] with values in [0, 1]
    """
    h = hsv[..., 0:1, :, :]  # Hue
    s = hsv[..., 1:2, :, :]  # Saturation
    v = hsv[..., 2:3, :, :]  # Value

    h = h * 6.0  # Scale hue to [0, 6]

    # Compute RGB components based on hue sector
    c = v * s
    x = c * (1 - torch.abs((h % 2) - 1))
    m = v - c

    # Initialize RGB
    rgb = torch.zeros_like(hsv)

    # Determine RGB values based on hue sector
    mask = (h >= 0) & (h < 1)
    rgb[..., 0:1, :, :] = torch.where(mask, c, rgb[..., 0:1, :, :])
    rgb[..., 1:2, :, :] = torch.where(mask, x, rgb[..., 1:2, :, :])

    mask = (h >= 1) & (h < 2)
    rgb[..., 0:1, :, :] = torch.where(mask, x, rgb[..., 0:1, :, :])
    rgb[..., 1:2, :, :] = torch.where(mask, c, rgb[..., 1:2, :, :])

    mask = (h >= 2) & (h < 3)
    rgb[..., 1:2, :, :] = torch.where(mask, c, rgb[..., 1:2, :, :])
    rgb[..., 2:3, :, :] = torch.where(mask, x, rgb[..., 2:3, :, :])

    mask = (h >= 3) & (h < 4)
    rgb[..., 1:2, :, :] = torch.where(mask, x, rgb[..., 1:2, :, :])
    rgb[..., 2:3, :, :] = torch.where(mask, c, rgb[..., 2:3, :, :])

    mask = (h >= 4) & (h < 5)
    rgb[..., 0:1, :, :] = torch.where(mask, x, rgb[..., 0:1, :, :])
    rgb[..., 2:3, :, :] = torch.where(mask, c, rgb[..., 2:3, :, :])

    mask = (h >= 5) & (h < 6)
    rgb[..., 0:1, :, :] = torch.where(mask, c, rgb[..., 0:1, :, :])
    rgb[..., 2:3, :, :] = torch.where(mask, x, rgb[..., 2:3, :, :])

    # Add m to all components
    rgb = rgb + m

    return rgb


def rgb_to_flow(motion_rgb: torch.Tensor,
                height: int,
                width: int,
                normalization_factor: float = 0.15) -> torch.Tensor:
    """
    Convert RGB motion representation back to optical flow (u, v).
    Inverse of flow_to_motion_rgb.

    Args:
        motion_rgb: RGB motion tensor [..., 3, H, W] in range [0, 1]
        height: Original video height
        width: Original video width
        normalization_factor: Scaling factor used in forward conversion

    Returns:
        flow: Optical flow tensor [..., 2, H, W]
    """
    # Convert RGB to HSV
    hsv = rgb_to_hsv_torch(motion_rgb)

    hue = hsv[..., 0:1, :, :]  # [..., 1, H, W] in [0, 1]
    magnitude = hsv[..., 2:3, :, :]  # [..., 1, H, W] in [0, 1]

    # Convert hue back to angle
    angle = hue * 2 * np.pi - np.pi  # [..., 1, H, W] in [-pi, pi]

    # Denormalize magnitude
    diagonal = np.sqrt(height ** 2 + width ** 2)
    magnitude_raw = magnitude * normalization_factor * diagonal

    # Convert polar to cartesian
    u = magnitude_raw * torch.cos(angle)
    v = magnitude_raw * torch.sin(angle)

    flow = torch.cat([u, v], dim=-3)

    return flow


def rgb_to_hsv_torch(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB to HSV color space using PyTorch operations.

    Args:
        rgb: RGB tensor [..., 3, H, W] with values in [0, 1]

    Returns:
        hsv: HSV tensor [..., 3, H, W] with values in [0, 1]
    """
    r = rgb[..., 0:1, :, :]
    g = rgb[..., 1:2, :, :]
    b = rgb[..., 2:3, :, :]

    max_rgb, max_idx = rgb.max(dim=-3, keepdim=True)
    min_rgb, _ = rgb.min(dim=-3, keepdim=True)

    delta = max_rgb - min_rgb

    # Hue calculation
    hue = torch.zeros_like(max_rgb)

    mask = (max_idx == 0) & (delta != 0)
    hue = torch.where(mask, ((g - b) / delta) % 6, hue)

    mask = (max_idx == 1) & (delta != 0)
    hue = torch.where(mask, ((b - r) / delta) + 2, hue)

    mask = (max_idx == 2) & (delta != 0)
    hue = torch.where(mask, ((r - g) / delta) + 4, hue)

    hue = hue / 6.0  # Normalize to [0, 1]

    # Saturation calculation
    saturation = torch.where(max_rgb != 0, delta / max_rgb, torch.zeros_like(max_rgb))

    # Value calculation
    value = max_rgb

    hsv = torch.cat([hue, saturation, value], dim=-3)

    return hsv


def save_motion_video(motion_rgb: torch.Tensor,
                     output_path: str,
                     fps: int = 30):
    """
    Save RGB motion representation as a video file using high-quality encoding.

    Args:
        motion_rgb: Motion RGB tensor [T, 3, H, W] or [B, T, 3, H, W]
        output_path: Output video file path
        fps: Frames per second
    """
    if motion_rgb.ndim == 5:
        # Take first batch element
        motion_rgb = motion_rgb[0]

    T, C, H, W = motion_rgb.shape

    # Convert to numpy and scale to [0, 255]
    motion_np = (motion_rgb.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)

    # Save frames to temporary directory and use ffmpeg for high-quality encoding
    import tempfile
    import subprocess

    temp_dir = tempfile.mkdtemp()

    try:
        # Save individual frames as PNG (lossless)
        for t in range(T):
            frame = cv2.cvtColor(motion_np[t], cv2.COLOR_RGB2BGR)
            frame_path = os.path.join(temp_dir, f'frame_{t:06d}.png')
            cv2.imwrite(frame_path, frame)

        # Use ffmpeg to create high-quality video
        # -pix_fmt yuv420p: compatible with most players
        # -crf 17: high quality (0=lossless, 23=default, 51=worst)
        # -preset slow: better compression
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(temp_dir, 'frame_%06d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '17',
            '-preset', 'medium',
            output_path
        ]

        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        print(f"Motion video saved to {output_path}")

    finally:
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir)



def precompute_motion_dataset(video_dir: str,
                              output_dir: str,
                              device: str = 'cuda',
                              video_ext: str = '.mp4',
                              use_videojam: bool = True,
                              sigma: float = 0.15,
                              deadzone_px: float = 0.0):
    """
    Precompute motion RGB videos for an entire dataset.

    IMPORTANT: Extract flow from GROUND TRUTH videos (person removed, correct physics).
    NOT from input videos with the person still present!

    The GT videos show the desired outcome (e.g., guitar falling after person removed).
    The optical flow from these GT videos teaches the model what realistic motion looks like.

    Args:
        video_dir: Directory containing GT videos (person removed, correct physics)
        output_dir: Directory to save motion RGB videos
        device: Device to run RAFT on
        video_ext: Video file extension
        use_videojam: If True, use VideoJAM normalization (recommended for training)
                     If False, use fixed-pixel visualization normalization
        sigma: VideoJAM normalization constant (default: 0.15)
        deadzone_px: Magnitude threshold to suppress noise (default: 0.05 px)
    """
    os.makedirs(output_dir, exist_ok=True)

    flow_extractor = RAFTFlowExtractor(device=device)
    video_files = sorted(Path(video_dir).glob(f'*{video_ext}'))

    print(f"Found {len(video_files)} videos to process")
    print(f"Using VideoJAM normalization: {use_videojam}")

    for video_path in video_files:
        print(f"\nProcessing {video_path.name}...")

        # Load video
        cap = cv2.VideoCapture(str(video_path))

        # Get FPS from input video
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print(f"Warning: Could not read FPS from {video_path.name}, defaulting to 30")
            fps = 30
        else:
            print(f"Input video FPS: {fps:.2f}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        # Convert to tensor [T, H, W, C] -> [T, C, H, W]
        frames = np.stack(frames)
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

        if use_videojam:
            # Use VideoJAM motion extraction (handles flow + normalization + temporal alignment)
            motion_rgb = flow_extractor.extract_videojam_motion(
                frames_tensor,
                sigma=sigma,
                deadzone_px=deadzone_px
            )  # [T, 3, H, W]
        else:
            # Legacy visualization mode
            flows = flow_extractor.extract_video_flow(frames_tensor)  # [T-1, 2, H, W]
            motion_rgb = flow_to_motion_rgb(flows)  # [T-1, 3, H, W]
            motion_rgb = torch.cat([motion_rgb[0:1], motion_rgb], dim=0)  # [T, 3, H, W]

        # Save motion video with matching FPS
        output_path = os.path.join(output_dir, video_path.name)
        save_motion_video(motion_rgb, output_path, fps=int(fps))

    print(f"\nMotion dataset saved to {output_dir}")


def extract_videojam_motion_from_video(video_path: str,
                                        output_path: str,
                                        device: str = 'cuda',
                                        sigma: float = 0.15,
                                        deadzone_px: float = 0.0,
                                        fps: int = None,
                                        target_size: int = None,
                                        target_resolution: int = None) -> torch.Tensor:
    """
    Standalone function to extract VideoJAM motion from a single video file.

    IMPORTANT: For training, set target_resolution to match your training resolution!
    This scales flow vectors to match the brightness you'd get at target resolution.

    Args:
        video_path: Path to input video
        output_path: Path to save motion RGB video
        device: Device to run RAFT on
        sigma: VideoJAM normalization constant (default: 0.15)
        deadzone_px: Magnitude threshold to suppress noise (default: 0.0 px)
        fps: Frame rate for output video (if None, matches input video FPS)
        target_size: DEPRECATED - use target_resolution instead
        target_resolution: Training resolution (e.g., 256). Scales flow vectors to match
                          brightness at this resolution while extracting at native resolution.
                          RECOMMENDED: Set to 256 for 256×256 training

    Returns:
        motion_rgb: RGB motion tensor [T, 3, H, W]
    """
    # Backwards compatibility: if target_size is provided but not target_resolution, use it
    if target_size is not None and target_resolution is None:
        print(f"⚠️  WARNING: target_size is deprecated. Use target_resolution instead.")
        print(f"   Setting target_resolution={target_size} for flow vector scaling.")
        target_resolution = target_size

    # Load video
    cap = cv2.VideoCapture(video_path)

    # Get FPS from input video if not specified
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print(f"Warning: Could not read FPS from video, defaulting to 30")
            fps = 30
        else:
            print(f"Detected input video FPS: {fps:.2f}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames loaded from {video_path}")

    print(f"Loaded {len(frames)} frames from {video_path}")

    # Convert to tensor [T, H, W, C] -> [T, C, H, W]
    frames = np.stack(frames)
    frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

    h, w = frames_tensor.shape[2:]
    print(f"Video resolution: {w}x{h}")
    print(f"Extracting flow at native resolution for maximum accuracy...")

    if target_resolution is not None:
        print(f"✓ Flow vectors will be scaled for target_resolution={target_resolution}")
        print(f"  This matches the brightness you'd get from native {target_resolution}x{target_resolution} extraction")

    # Extract VideoJAM motion at native resolution
    flow_extractor = RAFTFlowExtractor(device=device)
    motion_rgb = flow_extractor.extract_videojam_motion(
        frames_tensor,
        sigma=sigma,
        deadzone_px=deadzone_px,
        target_resolution=target_resolution
    )  # [T, 3, H, W]

    # Save motion video
    save_motion_video(motion_rgb, output_path, fps=fps)

    return motion_rgb


if __name__ == "__main__":
    """
    Example usage for precomputing motion dataset.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Precompute motion RGB videos from GT videos')
    parser.add_argument('--video_dir', type=str, required=True,
                       help='Directory containing GT videos')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save motion RGB videos')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run RAFT on (cuda or cpu)')
    parser.add_argument('--video_ext', type=str, default='.mp4',
                       help='Video file extension')

    args = parser.parse_args()

    precompute_motion_dataset(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        device=args.device,
        video_ext=args.video_ext
    )
