#!/usr/bin/env python3
"""
Stage 3a: Generate Grey Masks - Combine VLM Logic + User Trajectories

Generates grey masks (127=affected regions) by combining:
1. VLM-identified affected objects (segmented + gridified)
2. User-drawn trajectories (from Stage 3b)
3. Proximity filtering (only mask near primary object)

Input:  - vlm_analysis.json (Stage 2)
        - black_mask.mp4 (Stage 1)
        - trajectories.json (Stage 3b, optional)
Output: - grey_mask.mp4 (127=affected, 255=background)

Usage:
    python stage3a_generate_grey_masks.py --config more_dyn_2_config_points_absolute.json
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import subprocess

# Segmentation model
try:
    from ..sam3.model_builder import build_sam3_image_model
    from ..sam3.model.sam3_image_processor import Sam3Processor
    SAM3_AVAILABLE = True
except ImportError:
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        SAM3_AVAILABLE = True
    except ImportError:
        SAM3_AVAILABLE = False

try:
    from lang_sam import LangSAM
    LANGSAM_AVAILABLE = True
except ImportError:
    LANGSAM_AVAILABLE = False


class SegmentationModel:
    """Wrapper for segmentation"""

    def __init__(self, model_type: str = "sam3"):
        self.model_type = model_type.lower()

        if self.model_type == "sam3":
            if not SAM3_AVAILABLE:
                raise ImportError("SAM3 not available")
            print(f"   Loading SAM3...")
            model = build_sam3_image_model()
            self.processor = Sam3Processor(model)
            self.model = model
        elif self.model_type == "langsam":
            if not LANGSAM_AVAILABLE:
                raise ImportError("LangSAM not available")
            print(f"   Loading LangSAM...")
            self.model = LangSAM()
        else:
            raise ValueError(f"Unknown model: {model_type}")

    def segment(self, image_pil: Image.Image, prompt: str) -> np.ndarray:
        """Segment object using text prompt"""
        if self.model_type == "sam3":
            return self._segment_sam3(image_pil, prompt)
        else:
            return self._segment_langsam(image_pil, prompt)

    def _segment_sam3(self, image_pil: Image.Image, prompt: str) -> np.ndarray:
        from contextlib import nullcontext
        import torch
        h, w = image_pil.height, image_pil.width
        union = np.zeros((h, w), dtype=bool)

        try:
            use_cuda_amp = (
                torch.cuda.is_available()
                and str(getattr(self.processor, "device", "cuda")).startswith("cuda")
            )
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_cuda_amp
                else nullcontext()
            )

            with amp_ctx:
                inference_state = self.processor.set_image(image_pil)
                output = self.processor.set_text_prompt(state=inference_state, prompt=prompt)
            masks = output.get("masks")

            if masks is None or len(masks) == 0:
                return union

            if torch.is_tensor(masks):
                masks = masks.cpu().numpy()

            if masks.ndim == 2:
                union = masks.astype(bool)
            elif masks.ndim == 3:
                union = masks.any(axis=0).astype(bool)
            elif masks.ndim == 4:
                union = masks.any(axis=(0, 1)).astype(bool)

        except Exception as e:
            print(f"    Warning: SAM3 segmentation failed for '{prompt}': {e}")

        return union

    def _segment_langsam(self, image_pil: Image.Image, prompt: str) -> np.ndarray:
        h, w = image_pil.height, image_pil.width
        union = np.zeros((h, w), dtype=bool)

        try:
            results = self.model.predict([image_pil], [prompt])
            if not results:
                return union

            r0 = results[0]
            if isinstance(r0, dict) and "masks" in r0:
                masks = r0["masks"]
                if masks.ndim == 4 and masks.shape[0] == 1:
                    masks = masks[0]
                if masks.ndim == 3:
                    union = masks.any(axis=0).astype(bool)
                elif masks.ndim == 2:
                    union = masks.astype(bool)

        except Exception as e:
            print(f"    Warning: LangSAM segmentation failed for '{prompt}': {e}")

        return union


def calculate_square_grid(width: int, height: int, min_grid: int = 8) -> Tuple[int, int]:
    """Calculate grid dimensions for square cells"""
    aspect_ratio = width / height
    if width >= height:
        grid_rows = min_grid
        grid_cols = max(min_grid, round(min_grid * aspect_ratio))
    else:
        grid_cols = min_grid
        grid_rows = max(min_grid, round(min_grid / aspect_ratio))
    return grid_rows, grid_cols


def gridify_mask(mask: np.ndarray, grid_rows: int, grid_cols: int) -> np.ndarray:
    """Convert pixel mask to gridified mask"""
    h, w = mask.shape
    gridified = np.zeros((h, w), dtype=bool)

    cell_width = w / grid_cols
    cell_height = h / grid_rows

    for row in range(grid_rows):
        for col in range(grid_cols):
            y1 = int(row * cell_height)
            y2 = int((row + 1) * cell_height)
            x1 = int(col * cell_width)
            x2 = int((col + 1) * cell_width)

            cell_region = mask[y1:y2, x1:x2]
            if cell_region.any():
                gridified[y1:y2, x1:x2] = True

    return gridified


def grid_cells_to_mask(grid_cells: List[List[int]], grid_rows: int, grid_cols: int,
                       frame_width: int, frame_height: int) -> np.ndarray:
    """Convert grid cells to mask"""
    mask = np.zeros((frame_height, frame_width), dtype=bool)

    cell_width = frame_width / grid_cols
    cell_height = frame_height / grid_rows

    for row, col in grid_cells:
        y1 = int(row * cell_height)
        y2 = int((row + 1) * cell_height)
        x1 = int(col * cell_width)
        x2 = int((col + 1) * cell_width)
        mask[y1:y2, x1:x2] = True

    return mask


def dilate_mask(mask: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Dilate mask to create proximity region"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def filter_by_proximity(mask: np.ndarray, primary_mask: np.ndarray, dilation: int = 15) -> np.ndarray:
    """Filter mask to only include regions near primary mask"""
    # Dilate primary mask to create proximity region
    proximity_region = dilate_mask(primary_mask, dilation)

    # Only keep mask where it overlaps with proximity region
    filtered = mask & proximity_region

    return filtered


def process_video_grey_masks(video_info: Dict, segmenter: SegmentationModel,
                              trajectory_data: Dict = None):
    """Generate grey masks for a single video"""
    video_path = video_info.get("video_path", "")
    output_dir = Path(video_info.get("output_dir", ""))

    if not output_dir.exists():
        print(f"   ⚠️  Output directory not found: {output_dir}")
        return

    # Load required files
    vlm_analysis_path = output_dir / "vlm_analysis.json"
    black_mask_path = output_dir / "black_mask.mp4"
    input_video_path = output_dir / "input_video.mp4"

    if not vlm_analysis_path.exists():
        print(f"   ⚠️  vlm_analysis.json not found")
        return

    if not black_mask_path.exists():
        print(f"   ⚠️  black_mask.mp4 not found")
        return

    if not input_video_path.exists():
        input_video_path = Path(video_path)
        if not input_video_path.exists():
            print(f"   ⚠️  Video not found")
            return

    # Load VLM analysis
    with open(vlm_analysis_path, 'r') as f:
        analysis = json.load(f)

    # Get video properties
    cap = cv2.VideoCapture(str(input_video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Calculate grid
    min_grid = video_info.get('min_grid', 8)
    grid_rows, grid_cols = calculate_square_grid(frame_width, frame_height, min_grid)

    print(f"   Video: {frame_width}x{frame_height}, {total_frames} frames, grid: {grid_rows}x{grid_cols}")

    # Load first frame
    cap = cv2.VideoCapture(str(input_video_path))
    ret, first_frame = cap.read()
    cap.release()

    if not ret:
        print(f"   ⚠️  Failed to read first frame")
        return

    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    first_frame_pil = Image.fromarray(first_frame_rgb)

    # Load black mask (first frame for proximity filtering)
    black_cap = cv2.VideoCapture(str(black_mask_path))
    ret, black_mask_frame = black_cap.read()
    black_cap.release()

    if not ret:
        print(f"   ⚠️  Failed to read black mask")
        return

    if len(black_mask_frame.shape) == 3:
        black_mask_frame = cv2.cvtColor(black_mask_frame, cv2.COLOR_BGR2GRAY)

    primary_mask = (black_mask_frame == 0)  # 0 = primary object

    # Initialize grey mask
    grey_mask_combined = np.zeros((frame_height, frame_width), dtype=bool)

    # Process affected objects from VLM
    affected_objects = analysis.get('affected_objects', [])

    print(f"   Processing {len(affected_objects)} affected object(s)...")

    for obj in affected_objects:
        noun = obj.get('noun', '')
        category = obj.get('category', 'physical')
        will_move = obj.get('will_move', False)
        needs_trajectory = obj.get('needs_trajectory', False)

        if not noun:
            continue

        print(f"      • {noun} ({category})")

        # Check if we have trajectory data for this object
        has_trajectory = False
        if needs_trajectory and trajectory_data:
            for traj in trajectory_data:
                if traj.get('object_noun', '') == noun:
                    # Use trajectory grid cells
                    print(f"         Using user-drawn trajectory ({len(traj['trajectory_grid_cells'])} cells)")
                    traj_mask = grid_cells_to_mask(
                        traj['trajectory_grid_cells'],
                        grid_rows, grid_cols,
                        frame_width, frame_height
                    )
                    grey_mask_combined |= traj_mask
                    has_trajectory = True
                    break

        # If no trajectory or doesn't need one, segment normally
        if not has_trajectory:
            # Segment object
            obj_mask = segmenter.segment(first_frame_pil, noun)

            if obj_mask.any():
                print(f"         Segmented {obj_mask.sum()} pixels")

                # Filter by proximity to primary mask
                obj_mask_filtered = filter_by_proximity(obj_mask, primary_mask, dilation=50)

                if obj_mask_filtered.any():
                    print(f"         After proximity filter: {obj_mask_filtered.sum()} pixels")

                    # Gridify
                    obj_mask_gridified = gridify_mask(obj_mask_filtered, grid_rows, grid_cols)

                    # Add to combined grey mask
                    grey_mask_combined |= obj_mask_gridified

                    print(f"         ✓ Added to grey mask")
                else:
                    print(f"         ⚠️  No pixels near primary object, skipping")
            else:
                print(f"         ⚠️  Segmentation failed")

    # Generate grey mask video
    print(f"   Generating grey mask video...")

    # For simplicity, use same mask for all frames
    # (In future, could track objects through video)
    grey_mask_uint8 = np.where(grey_mask_combined, 127, 255).astype(np.uint8)

    # Write temp AVI
    temp_avi = output_dir / "grey_mask_temp.avi"
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    out = cv2.VideoWriter(str(temp_avi), fourcc, fps, (frame_width, frame_height), isColor=False)

    for _ in range(total_frames):
        out.write(grey_mask_uint8)

    out.release()

    # Convert to MP4
    grey_mask_mp4 = output_dir / "grey_mask.mp4"
    cmd = [
        'ffmpeg', '-y', '-i', str(temp_avi),
        '-c:v', 'libx264', '-qp', '0', '-preset', 'ultrafast',
        '-pix_fmt', 'yuv444p',
        str(grey_mask_mp4)
    ]
    subprocess.run(cmd, capture_output=True)
    temp_avi.unlink()

    print(f"   ✓ Saved grey_mask.mp4")

    # Save debug visualization
    debug_vis = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    debug_vis[grey_mask_combined] = [0, 255, 0]  # Green for affected regions
    debug_vis[primary_mask] = [255, 0, 0]  # Red for primary
    debug_path = output_dir / "debug_grey_mask.jpg"
    cv2.imwrite(str(debug_path), debug_vis)
    print(f"   ✓ Saved debug visualization")


def main():
    parser = argparse.ArgumentParser(description="Stage 3a: Generate Grey Masks")
    parser.add_argument("--config", required=True, help="Config JSON")
    parser.add_argument("--segmentation-model", default="sam3", choices=["langsam", "sam3"],
                       help="Segmentation model")
    args = parser.parse_args()

    config_path = Path(args.config)

    # Load config
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    if isinstance(config_data, list):
        videos = config_data
    elif isinstance(config_data, dict) and "videos" in config_data:
        videos = config_data["videos"]
    else:
        raise ValueError("Invalid config format")

    # Load trajectory data if exists
    trajectory_path = config_path.parent / f"{config_path.stem}_trajectories.json"
    trajectory_data = None

    if trajectory_path.exists():
        print(f"Loading trajectory data from: {trajectory_path.name}")
        with open(trajectory_path, 'r') as f:
            trajectory_data = json.load(f)
        print(f"   Loaded {len(trajectory_data)} trajectory(s)")
    else:
        print(f"No trajectory data found (Stage 3b not run or no objects needed trajectories)")

    print(f"\n{'='*70}")
    print(f"Stage 3a: Generate Grey Masks")
    print(f"{'='*70}")
    print(f"Videos: {len(videos)}")
    print(f"Segmentation: {args.segmentation_model.upper()}")
    print(f"{'='*70}\n")

    # Load segmentation model
    segmenter = SegmentationModel(args.segmentation_model)

    # Process each video
    for i, video_info in enumerate(videos):
        video_path = video_info.get('video_path', '')
        print(f"\n{'─'*70}")
        print(f"Video {i+1}/{len(videos)}: {Path(video_path).parent.name}")
        print(f"{'─'*70}")

        try:
            process_video_grey_masks(video_info, segmenter, trajectory_data)
            print(f"\n✅ Video {i+1} complete!")

        except Exception as e:
            print(f"\n❌ Error processing video {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*70}")
    print(f"✅ Stage 3a Complete!")
    print(f"{'='*70}")
    print(f"Generated grey_mask.mp4 for all videos")
    print(f"Next: Run Stage 4 to combine black + grey masks")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
