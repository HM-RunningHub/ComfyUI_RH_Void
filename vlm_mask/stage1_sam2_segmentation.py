#!/usr/bin/env python3
"""
Stage 1: SAM2 Point-Prompted Segmentation

Takes user-selected points and generates pixel-perfect masks of primary objects
using SAM2 video tracking.

Input:  <config>_points.json (with primary_points)
Output: For each video:
        - black_mask.mp4: Primary object mask (0=object, 255=background)
        - first_frame.jpg: First frame for VLM analysis
        - segmentation_info.json: Metadata

Usage:
    python stage1_sam2_segmentation.py --config more_dyn_2_config_points.json
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
import torch
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
# Check SAM2 availability
try:
    from ..sam2.build_sam import build_sam2_video_predictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("⚠️  SAM2 not installed. Install with:")
    print("   pip install git+https://github.com/facebookresearch/segment-anything-2.git")
    sys.exit(1)


class SAM2PointSegmenter:
    """SAM2 video segmentation with point prompts"""

    def __init__(self, checkpoint_path: str, model_cfg: str = "sam2_hiera_l.yaml", device: str = "cuda"):
        # print(f"   Loading SAM2 video predictor...")
        # self.device = device
        # self.predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device=device)
        # print(f"   ✓ SAM2 loaded on {device}")
        # kiki
        config_filename = (
            "sam2.1_hiera_l.yaml"
            if "2.1" in str(checkpoint_path)
            else "sam2_hiera_l.yaml"
        )
        LOCAL_MODEL_CFG = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            config_filename,
        )

        print(f"   Loading SAM2 video predictor...")
        self.device = device
        config_dir = os.path.dirname(LOCAL_MODEL_CFG)
        config_name = os.path.basename(LOCAL_MODEL_CFG)
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        initialize_config_dir(config_dir=config_dir, version_base="1.2")
        self.predictor = build_sam2_video_predictor(config_name, checkpoint_path, device=device)

    def segment_video(self, video_path: str, points: List[List[int]] = None,
                      output_mask_path: str = None, temp_dir: str = None,
                      first_appears_frame: int = 0,
                      points_by_frame: Dict[int, List[List[int]]] = None) -> Dict:
        """
        Segment video using point prompts (single or multi-frame).

        Args:
            video_path: Path to input video
            points: List of [x, y] points on object (single frame, legacy)
            output_mask_path: Path to save mask video
            temp_dir: Directory for temporary frames
            first_appears_frame: Frame index where points were selected (for single frame)
            points_by_frame: Dict mapping frame_idx → [[x, y], ...] (multi-frame support)

        Returns:
            Dict with segmentation metadata
        """
        # Handle both old and new formats
        if points_by_frame is not None:
            # Multi-frame format
            if not points_by_frame or len(points_by_frame) == 0:
                raise ValueError("No points provided")
        elif points is not None:
            # Single frame format (backwards compat)
            if not points or len(points) == 0:
                raise ValueError("No points provided")
            points_by_frame = {first_appears_frame: points}
        else:
            raise ValueError("Must provide either points or points_by_frame")

        # Create temp directory for frames
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="sam2_frames_")
            cleanup = True
        else:
            Path(temp_dir).mkdir(parents=True, exist_ok=True)
            cleanup = False

        print(f"   Extracting frames to: {temp_dir}")
        frame_files = self._extract_frames(video_path, temp_dir)

        if len(frame_files) == 0:
            raise RuntimeError(f"No frames extracted from {video_path}")

        # Get video properties
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = len(frame_files)
        cap.release()

        # Count total points across all frames
        total_points = sum(len(pts) for pts in points_by_frame.values())
        print(f"   Video: {frame_width}x{frame_height}, {total_frames} frames @ {fps} fps")
        print(f"   Using {total_points} points across {len(points_by_frame)} frame(s) for segmentation")

        # Initialize SAM2
        print(f"   Initializing SAM2...")
        inference_state = self.predictor.init_state(video_path=temp_dir)

        # Add points for each frame (all with obj_id=1 to merge into single mask)
        for frame_idx in sorted(points_by_frame.keys()):
            frame_points = points_by_frame[frame_idx]

            # Convert points to numpy array
            points_np = np.array(frame_points, dtype=np.float32)
            labels_np = np.ones(len(frame_points), dtype=np.int32)  # All positive

            # Calculate bounding box from points (with 10% margin for hair/clothes)
            x_coords = points_np[:, 0]
            y_coords = points_np[:, 1]

            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()

            # Add 10% margin
            x_margin = (x_max - x_min) * 0.1
            y_margin = (y_max - y_min) * 0.1

            box = np.array([
                max(0, x_min - x_margin),
                max(0, y_min - y_margin),
                min(frame_width, x_max + x_margin),
                min(frame_height, y_max + y_margin)
            ], dtype=np.float32)

            print(f"   Adding {len(frame_points)} points + box to frame {frame_idx}")
            print(f"   Points: {frame_points[:3]}..." if len(frame_points) > 3 else f"   Points: {frame_points}")
            print(f"   Box: [{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}]")

            # Add points + box to this frame (all use obj_id=1 to merge)
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=1,
                points=points_np,
                labels=labels_np,
                box=box,
            )

        print(f"   Propagating through video...")

        # Propagate through video
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            # Get mask for object ID 1
            mask_logits = out_mask_logits[out_obj_ids.index(1)]
            mask = (mask_logits > 0.0).cpu().numpy().squeeze()
            video_segments[out_frame_idx] = mask

        print(f"   ✓ Segmented {len(video_segments)} frames")

        # Write mask video
        print(f"   Writing mask video...")
        self._write_mask_video(
            video_segments,
            output_mask_path,
            fps,
            frame_width,
            frame_height,
            total_frames,
        )

        # Cleanup
        if cleanup:
            shutil.rmtree(temp_dir)

        # Build metadata
        metadata = {
            "total_frames": total_frames,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "fps": fps,
        }

        # Add points info based on format
        if points_by_frame:
            total_points = sum(len(pts) for pts in points_by_frame.values())
            metadata["num_points"] = total_points
            metadata["points_by_frame"] = {str(k): v for k, v in points_by_frame.items()}
        else:
            metadata["num_points"] = len(points) if points else 0
            metadata["points"] = points

        return metadata

    def _extract_frames(self, video_path: str, output_dir: str) -> List[str]:
        """Extract video frames as JPG files"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        frame_files = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # SAM2 expects frames named as frame_000000.jpg, frame_000001.jpg, etc.
            frame_filename = f"{frame_idx:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_files.append(frame_path)
            frame_idx += 1

            if frame_idx % 20 == 0:
                print(f"      Extracted {frame_idx} frames...", end='\r')

        cap.release()
        print(f"      Extracted {frame_idx} frames")

        return frame_files

    def _write_mask_video(self, masks: Dict[int, np.ndarray], output_path: str,
                          fps: float, width: int, height: int, total_frames: int):
        """Write a full-length mask video aligned to the source video frames."""
        # Write temp AVI first
        temp_avi = Path(output_path).with_suffix('.avi')
        fourcc = cv2.VideoWriter_fourcc(*'FFV1')
        out = cv2.VideoWriter(str(temp_avi), fourcc, fps, (width, height), isColor=False)

        blank_mask = np.full((height, width), 255, dtype=np.uint8)

        for frame_idx in range(total_frames):
            mask = masks.get(frame_idx)
            if mask is None:
                mask_uint8 = blank_mask
            else:
                # Convert boolean mask to 0/255
                mask_uint8 = np.where(mask, 0, 255).astype(np.uint8)
            out.write(mask_uint8)

        out.release()

        # Convert to lossless MP4
        cmd = [
            'ffmpeg', '-y', '-i', str(temp_avi),
            '-c:v', 'libx264', '-qp', '0', '-preset', 'ultrafast',
            '-pix_fmt', 'yuv444p',
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True)
        temp_avi.unlink()

        print(f"   ✓ Saved mask video: {output_path}")


def process_config(config_path: str, sam2_checkpoint: str, device: str = "cuda"):
    """Process all videos in config"""
    config_path = Path(config_path)

    # Load config
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    # Handle both formats
    if isinstance(config_data, list):
        videos = config_data
    elif isinstance(config_data, dict) and "videos" in config_data:
        videos = config_data["videos"]
    else:
        raise ValueError("Config must be a list or have 'videos' key")

    print(f"\n{'='*70}")
    print(f"Stage 1: SAM2 Point-Prompted Segmentation")
    print(f"{'='*70}")
    print(f"Config: {config_path.name}")
    print(f"Videos: {len(videos)}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")

    # Initialize SAM2
    segmenter = SAM2PointSegmenter(sam2_checkpoint, device=device)

    # Process each video
    for i, video_info in enumerate(videos):
        video_path = video_info.get("video_path", "")
        instruction = video_info.get("instruction", "")
        output_dir = video_info.get("output_dir", "")

        # Read points - support both single-frame and multi-frame formats
        points_by_frame_raw = video_info.get("primary_points_by_frame", None)
        points = video_info.get("primary_points", [])
        first_appears_frame = video_info.get("first_appears_frame", 0)

        # Convert points_by_frame from string keys to int keys
        points_by_frame = None
        if points_by_frame_raw:
            points_by_frame = {int(k): v for k, v in points_by_frame_raw.items()}

        if not video_path:
            print(f"\n⚠️  Video {i+1}: No video_path, skipping")
            continue

        if not points and not points_by_frame:
            print(f"\n⚠️  Video {i+1}: No primary_points selected, skipping")
            continue

        video_path = Path(video_path)
        if not video_path.exists():
            print(f"\n⚠️  Video {i+1}: File not found: {video_path}, skipping")
            continue

        print(f"\n{'─'*70}")
        print(f"Video {i+1}/{len(videos)}: {video_path.name}")
        print(f"{'─'*70}")
        print(f"Instruction: {instruction}")

        if points_by_frame:
            total_points = sum(len(pts) for pts in points_by_frame.values())
            print(f"Points: {total_points} across {len(points_by_frame)} frame(s)")
            print(f"Frames: {sorted(points_by_frame.keys())}")
        else:
            print(f"Points: {len(points)}")
            print(f"First appears frame: {first_appears_frame}")

        # Setup output directory
        if output_dir:
            output_dir = Path(output_dir)
        else:
            # Create unique output directory per video
            video_name = video_path.stem  # Get video name without extension
            output_dir = video_path.parent / f"{video_name}_masks_output"

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output: {output_dir}")

        try:
            # Segment video - use multi-frame or single-frame format
            black_mask_path = output_dir / "black_mask.mp4"
            if points_by_frame:
                metadata = segmenter.segment_video(
                    str(video_path),
                    output_mask_path=str(black_mask_path),
                    points_by_frame=points_by_frame
                )
                # Use first frame for VLM analysis
                first_frame_for_vlm = min(points_by_frame.keys())
            else:
                metadata = segmenter.segment_video(
                    str(video_path),
                    points=points,
                    output_mask_path=str(black_mask_path),
                    first_appears_frame=first_appears_frame
                )
                first_frame_for_vlm = first_appears_frame

            # Save frame where object appears for VLM analysis
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame_for_vlm)
            ret, first_frame = cap.read()
            cap.release()

            if ret:
                first_frame_path = output_dir / "first_frame.jpg"
                cv2.imwrite(str(first_frame_path), first_frame)
                print(f"   ✓ Saved first frame (frame {first_frame_for_vlm}): {first_frame_path.name}")

            # Copy input video
            input_copy_path = output_dir / "input_video.mp4"
            if not input_copy_path.exists():
                shutil.copy2(video_path, input_copy_path)
                print(f"   ✓ Copied input video")

            # Save metadata
            metadata["video_path"] = str(video_path)
            metadata["instruction"] = instruction
            if points_by_frame:
                metadata["primary_points_by_frame"] = {str(k): v for k, v in points_by_frame.items()}
                metadata["primary_frames"] = sorted(points_by_frame.keys())
                metadata["first_appears_frame"] = min(points_by_frame.keys())
            else:
                metadata["primary_points"] = points
                metadata["first_appears_frame"] = first_appears_frame

            metadata_path = output_dir / "segmentation_info.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"   ✓ Saved metadata: {metadata_path.name}")

            print(f"\n✅ Video {i+1} complete!")

        except Exception as e:
            print(f"\n❌ Error processing video {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*70}")
    print(f"Stage 1 Complete!")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Stage 1: SAM2 Point-Prompted Segmentation")
    parser.add_argument("--config", required=True, help="Config JSON with primary_points")
    parser.add_argument("--sam2-checkpoint", default="../sam2_hiera_large.pt",
                       help="Path to SAM2 checkpoint")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    if not SAM2_AVAILABLE:
        print("❌ SAM2 not available")
        sys.exit(1)

    # Check checkpoint exists
    checkpoint_path = Path(args.sam2_checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print(f"   Download with:")
        print(f"   wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt")
        sys.exit(1)

    process_config(args.config, str(checkpoint_path), args.device)


if __name__ == "__main__":
    main()
