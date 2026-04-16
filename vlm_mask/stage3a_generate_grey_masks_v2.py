#!/usr/bin/env python3
"""
Stage 3a: Generate Grey Masks (CORRECTED)

Correct pipeline:
1. For EACH affected object (from VLM analysis):
   a) IF user drew trajectory (Stage 3b):
      - Segment object in first_appears_frame → get SIZE
      - Apply object SIZE along trajectory path across all frames
   b) ELSE (no user trajectory):
      - Segment object through ALL frames (captures any movement/changes)
      - This handles:
        * Static objects (can, chair)
        * Objects that move during video (golf ball)
        * Dynamic effects (paint strokes, shadows)
      - Filter by proximity to primary object

2. Accumulate all masks (one combined mask per frame)

3. Gridify ALL accumulated masks:
   - If ANY pixel in grid cell → ENTIRE cell = 127

4. Write grey_mask.mp4

Key insight: will_move / needs_trajectory are ONLY for Stage 3b (user input).
In Stage 3a, we segment ALL affected objects through ALL frames.

Input:  - vlm_analysis.json (Stage 2)
        - black_mask.mp4 (Stage 1)
        - trajectories.json (Stage 3b, optional)
Output: - grey_mask.mp4 (127=affected, 255=background)

Usage:
    python stage3a_generate_grey_masks_v2.py --config more_dyn_2_config_points_absolute.json
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import subprocess

try:
    import folder_paths
except ImportError:
    folder_paths = None

# SAM2 for video tracking
try:
    from sam2.build_sam import build_sam2_video_predictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False

# SAM3 for single-frame segmentation
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

# LangSAM
try:
    from lang_sam import LangSAM
    LANGSAM_AVAILABLE = True
except ImportError:
    LANGSAM_AVAILABLE = False


def _resolve_sam3_asset_paths() -> Tuple[str, str]:
    """Resolve local SAM3 checkpoint and BPE paths."""
    model_roots = []
    if folder_paths is not None and getattr(folder_paths, "models_dir", None):
        model_roots.append(Path(folder_paths.models_dir) / "sam3")

    checkpoint_path = None
    bpe_path = None

    for root in model_roots:
        ckpt_candidate = root / "sam3.pt"
        bpe_candidate = root / "bpe_simple_vocab_16e6.txt.gz"
        if checkpoint_path is None and ckpt_candidate.is_file():
            checkpoint_path = str(ckpt_candidate)
        if bpe_path is None and bpe_candidate.is_file():
            bpe_path = str(bpe_candidate)

    if checkpoint_path is None:
        searched_ckpts = [str(root / "sam3.pt") for root in model_roots]
        raise FileNotFoundError(
            "SAM3 checkpoint not found. Expected one of: " + ", ".join(searched_ckpts)
        )

    if bpe_path is None:
        raise FileNotFoundError(
            "SAM3 BPE vocab not found. Expected 'bpe_simple_vocab_16e6.txt.gz' under "
            "/workspace/ComfyUI/models/sam3 or ComfyUI-RMBG/models/sam3/assets."
        )

    return checkpoint_path, bpe_path


class SegmentationModel:
    """Wrapper for segmentation"""

    def __init__(self, model_type: str = "sam3"):
        self.model_type = model_type.lower()

        if self.model_type == "sam3":
            if not SAM3_AVAILABLE:
                raise ImportError("SAM3 not available")
            print(f"   Loading SAM3...")
            checkpoint_path, bpe_path = _resolve_sam3_asset_paths()
            print(f"   SAM3 checkpoint: {checkpoint_path}")
            print(f"   SAM3 BPE: {bpe_path}")
            model = build_sam3_image_model(
                bpe_path=bpe_path,
                checkpoint_path=checkpoint_path,
                load_from_HF=False,
            )
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
        """Segment object using text prompt - returns boolean mask"""
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
            print(f"         Warning: SAM3 failed: {e}")

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
            print(f"         Warning: LangSAM failed: {e}")

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


def gridify_masks(masks: List[np.ndarray], grid_rows: int, grid_cols: int) -> List[np.ndarray]:
    """
    Gridify masks: if ANY pixel in grid cell → ENTIRE cell = True

    Args:
        masks: List of boolean masks (one per frame)
        grid_rows, grid_cols: Grid dimensions

    Returns:
        List of gridified boolean masks
    """
    gridified_masks = []

    for mask in masks:
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
                # If ANY pixel in cell → ENTIRE cell
                if cell_region.any():
                    gridified[y1:y2, x1:x2] = True

        gridified_masks.append(gridified)

    return gridified_masks


def get_object_size(mask: np.ndarray) -> Tuple[int, int]:
    """Get bounding box size of object"""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return 0, 0

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    width = x2 - x1 + 1
    height = y2 - y1 + 1

    return width, height


def apply_object_along_trajectory(obj_mask: np.ndarray, trajectory_points: List[Tuple[int, int]],
                                   total_frames: int, frame_shape: Tuple[int, int]) -> List[np.ndarray]:
    """
    Apply object along trajectory path across frames.

    Args:
        obj_mask: Object mask from first_appears_frame
        trajectory_points: List of (x, y) points defining path
        total_frames: Total number of frames in video
        frame_shape: (height, width)

    Returns:
        List of masks (one per frame) with object placed along trajectory
    """
    h, w = frame_shape
    masks = [np.zeros((h, w), dtype=bool) for _ in range(total_frames)]

    if len(trajectory_points) < 2:
        return masks

    # Get object size
    obj_width, obj_height = get_object_size(obj_mask)

    if obj_width == 0 or obj_height == 0:
        return masks

    # Interpolate trajectory across frames
    num_traj_points = len(trajectory_points)

    for frame_idx in range(total_frames):
        # Map frame index to trajectory point
        t = frame_idx / max(total_frames - 1, 1)  # 0.0 to 1.0
        traj_idx = int(t * (num_traj_points - 1))
        traj_idx = min(traj_idx, num_traj_points - 1)

        # Get position on trajectory
        x_center, y_center = trajectory_points[traj_idx]

        # Place object at this position
        x1 = max(0, int(x_center - obj_width // 2))
        y1 = max(0, int(y_center - obj_height // 2))
        x2 = min(w, x1 + obj_width)
        y2 = min(h, y1 + obj_height)

        # Place object mask
        masks[frame_idx][y1:y2, x1:x2] = True

    return masks


def segment_object_all_frames(video_path: str, obj_noun: str, segmenter: SegmentationModel,
                               frame_stride: int = 1) -> List[np.ndarray]:
    """
    Segment object through all frames.

    Args:
        video_path: Path to video
        obj_noun: Object to segment
        segmenter: Segmentation model
        frame_stride: Process every Nth frame (for speed)

    Returns:
        List of boolean masks (one per frame)
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    masks = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_stride == 0:
            # Segment this frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            mask = segmenter.segment(frame_pil, obj_noun)
            masks.append(mask)

            if (frame_idx + 1) % 10 == 0:
                print(f"         Frame {frame_idx + 1}/{total_frames}...", end='\r')
        else:
            # Reuse previous mask
            if masks:
                masks.append(masks[-1])
            else:
                masks.append(np.zeros((frame_height, frame_width), dtype=bool))

        frame_idx += 1

    cap.release()
    print(f"         Segmented {total_frames} frames")

    return masks


def dilate_mask(mask: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Dilate mask for proximity checking"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def load_primary_masks_by_frame(black_mask_path: str, total_frames: int) -> List[np.ndarray]:
    """Load the primary-object mask for every frame from black_mask.mp4."""
    cap = cv2.VideoCapture(str(black_mask_path))
    primary_masks: List[np.ndarray] = []

    while True:
        ret, black_mask_frame = cap.read()
        if not ret:
            break

        if len(black_mask_frame.shape) == 3:
            black_mask_frame = cv2.cvtColor(black_mask_frame, cv2.COLOR_BGR2GRAY)

        primary_masks.append(black_mask_frame == 0)

    cap.release()

    if not primary_masks:
        raise ValueError(f"Failed to load any frames from black mask: {black_mask_path}")

    # Be defensive if the decoded frame count differs from the source metadata.
    if len(primary_masks) < total_frames:
        pad_mask = primary_masks[-1]
        primary_masks.extend([pad_mask.copy() for _ in range(total_frames - len(primary_masks))])
    elif len(primary_masks) > total_frames:
        primary_masks = primary_masks[:total_frames]

    return primary_masks


def _summarize_frame_indices(frame_indices: List[int], limit: int = 12) -> str:
    """Make long frame index lists readable in logs."""
    if not frame_indices:
        return "[]"
    if len(frame_indices) <= limit:
        return str(frame_indices)
    shown = ", ".join(str(i) for i in frame_indices[:limit])
    return f"[{shown}, ...] ({len(frame_indices)} frames)"


def filter_masks_by_proximity_per_frame(
    masks: List[np.ndarray],
    primary_masks: List[np.ndarray],
    dilation: int = 50,
    noun: str = "",
) -> List[np.ndarray]:
    """Filter masks against the primary mask of the same frame and print debug stats."""
    filtered: List[np.ndarray] = []
    fully_filtered_frames: List[int] = []
    partially_filtered_frames: List[int] = []
    kept_frames: List[int] = []
    raw_total_pixels = 0
    filtered_total_pixels = 0

    for frame_idx, mask in enumerate(masks):
        primary_mask = primary_masks[min(frame_idx, len(primary_masks) - 1)]
        proximity_region = dilate_mask(primary_mask, dilation)
        filtered_mask = mask & proximity_region
        filtered.append(filtered_mask)

        raw_pixels = int(mask.sum())
        filtered_pixels = int(filtered_mask.sum())
        raw_total_pixels += raw_pixels
        filtered_total_pixels += filtered_pixels

        if raw_pixels == 0:
            continue
        if filtered_pixels == 0:
            fully_filtered_frames.append(frame_idx)
        elif filtered_pixels < raw_pixels:
            partially_filtered_frames.append(frame_idx)
        else:
            kept_frames.append(frame_idx)

    noun_label = noun or "<unknown>"
    print(
        f"         Proximity filter (per-frame, dilation={dilation}) for '{noun_label}': "
        f"raw={raw_total_pixels}, kept={filtered_total_pixels}, removed={raw_total_pixels - filtered_total_pixels}"
    )
    if fully_filtered_frames:
        print(f"         Fully filtered frames: {_summarize_frame_indices(fully_filtered_frames)}")
    if partially_filtered_frames:
        print(f"         Partially filtered frames: {_summarize_frame_indices(partially_filtered_frames)}")
    if kept_frames:
        print(f"         Unchanged frames after filter: {_summarize_frame_indices(kept_frames)}")

    return filtered


def process_video_grey_masks(video_info: Dict, segmenter: SegmentationModel,
                              trajectory_data: List[Dict] = None):
    """Generate grey masks for a single video"""
    video_path = video_info.get("video_path", "")
    output_dir = Path(video_info.get("output_dir", ""))

    if not output_dir.exists():
        print(f"   ⚠️  Output directory not found")
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

    # Keep the Stage 1 black mask only for debug visualization. Stage 3 filtering is disabled.
    primary_masks = load_primary_masks_by_frame(str(black_mask_path), total_frames)

    # Initialize accumulated masks (one per frame)
    accumulated_masks = [np.zeros((frame_height, frame_width), dtype=bool) for _ in range(total_frames)]

    # Process affected objects
    affected_objects = analysis.get('affected_objects', [])
    print(f"   Processing {len(affected_objects)} affected object(s)...")

    for obj in affected_objects:
        noun = obj.get('noun', '')

        if not noun:
            continue

        print(f"      • {noun}")

        # Check if we have USER TRAJECTORY for this object
        has_trajectory = False
        if trajectory_data:
            for traj in trajectory_data:
                if traj.get('object_noun', '') == noun and not traj.get('skipped', False):
                    has_trajectory = True
                    traj_points = traj.get('trajectory_points', [])

                    print(f"         Using user-drawn trajectory ({len(traj_points)} points)")

                    # Segment object in first_appears_frame to get SIZE
                    first_frame_idx = obj.get('first_appears_frame', 0)
                    cap = cv2.VideoCapture(str(input_video_path))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame_idx)
                    ret, frame = cap.read()
                    cap.release()

                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_pil = Image.fromarray(frame_rgb)
                        obj_mask = segmenter.segment(frame_pil, noun)

                        if obj_mask.any():
                            obj_width, obj_height = get_object_size(obj_mask)
                            print(f"         Segmented object (size: {obj_width}x{obj_height} px)")

                            # Apply object SIZE along trajectory
                            traj_masks = apply_object_along_trajectory(
                                obj_mask, traj_points, total_frames, (frame_height, frame_width)
                            )

                            # Accumulate
                            for i in range(total_frames):
                                accumulated_masks[i] |= traj_masks[i]

                            print(f"         ✓ Applied object along trajectory across {total_frames} frames")
                        else:
                            print(f"         ⚠️  Segmentation failed, using trajectory grid cells only")
                            # Fallback: just use trajectory grid cells
                            grid_cells = traj.get('trajectory_grid_cells', [])
                            for row, col in grid_cells:
                                y1 = int(row * frame_height / grid_rows)
                                y2 = int((row + 1) * frame_height / grid_rows)
                                x1 = int(col * frame_width / grid_cols)
                                x2 = int((col + 1) * frame_width / grid_cols)
                                for i in range(total_frames):
                                    accumulated_masks[i][y1:y2, x1:x2] = True

                    break

        # If NO user trajectory, segment through ALL frames
        # This captures: static objects, objects that move during video, dynamic effects
        if not has_trajectory:
            print(f"         Segmenting through ALL frames (captures any movement/changes)...")
            obj_masks = segment_object_all_frames(str(input_video_path), noun, segmenter, frame_stride=5)
            print(f"         Proximity filter disabled; using raw segmentation masks")

            # Accumulate
            for i in range(len(obj_masks)):
                if i < len(accumulated_masks):
                    accumulated_masks[i] |= obj_masks[i]

            pixel_count = sum(mask.sum() for mask in obj_masks)
            print(f"         ✓ Segmented across {len(obj_masks)} frames ({pixel_count} total raw pixels)")

    # Cumulative union: once a region becomes grey, keep it for all later frames.
    print(f"   Applying cumulative union across frames...")
    for frame_idx in range(1, len(accumulated_masks)):
        accumulated_masks[frame_idx] |= accumulated_masks[frame_idx - 1]

    # GRIDIFY all accumulated masks
    print(f"   Gridifying masks...")
    gridified_masks = gridify_masks(accumulated_masks, grid_rows, grid_cols)

    # Convert to uint8 (127 = grey, 255 = background)
    grey_masks_uint8 = [np.where(mask, 127, 255).astype(np.uint8) for mask in gridified_masks]

    # Write video
    print(f"   Writing grey_mask.mp4...")
    temp_avi = output_dir / "grey_mask_temp.avi"
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    out = cv2.VideoWriter(str(temp_avi), fourcc, fps, (frame_width, frame_height), isColor=False)

    for mask in grey_masks_uint8:
        out.write(mask)

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

    # Save debug visualization (first frame)
    debug_vis = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    debug_vis[gridified_masks[0]] = [0, 255, 0]  # Green
    debug_vis[primary_masks[0]] = [255, 0, 0]  # Red
    debug_path = output_dir / "debug_grey_mask.jpg"
    cv2.imwrite(str(debug_path), debug_vis)


def process_config(config_path: str, segmentation_model: str = "sam3"):
    """Process all videos in config and generate grey masks."""
    config_path = Path(config_path)

    # Load config
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    if isinstance(config_data, list):
        videos = config_data
    elif isinstance(config_data, dict) and "videos" in config_data:
        videos = config_data["videos"]
    else:
        raise ValueError("Invalid config format")

    # Load trajectory data
    trajectory_path = config_path.parent / f"{config_path.stem}_trajectories.json"
    trajectory_data = None

    if trajectory_path.exists():
        print(f"Loading trajectory data: {trajectory_path.name}")
        with open(trajectory_path, 'r') as f:
            trajectory_data = json.load(f)
        print(f"   Loaded {len(trajectory_data)} trajectory(s)")

    print(f"\n{'='*70}")
    print(f"Stage 3a: Generate Grey Masks (CORRECTED)")
    print(f"{'='*70}")
    print(f"Videos: {len(videos)}")
    print(f"Segmentation: {segmentation_model.upper()}")
    print(f"{'='*70}\n")

    # Load segmentation model
    segmenter = SegmentationModel(segmentation_model)

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
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*70}")
    print(f"✅ Stage 3a Complete!")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Stage 3a: Generate Grey Masks (Corrected)")
    parser.add_argument("--config", required=True, help="Config JSON")
    parser.add_argument("--segmentation-model", default="sam3", choices=["langsam", "sam3"],
                       help="Segmentation model")
    args = parser.parse_args()
    process_config(args.config, args.segmentation_model)


if __name__ == "__main__":
    main()
