#!/usr/bin/env python3
"""
Stage 4: Combine Black and Grey Masks into Tri/Quad Mask

Combines the black mask (primary object) and grey masks (affected objects)
into a single tri-mask or quad-mask video.

Mask values:
- 0: Primary object (from black mask)
- 63: Overlap of primary and affected objects
- 127: Affected objects only (from grey masks)
- 255: Background (keep)
"""

import json
import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def combine_masks(black_frame, grey_frame):
    """
    Combine black and grey mask frames.

    Rules:
    - black=0, grey=255 → 0 (primary object only)
    - black=255, grey=127 → 127 (affected object only)
    - black=0, grey=127 → 63 (overlap)
    - black=255, grey=255 → 255 (background)

    Args:
        black_frame: Frame from black_mask.mp4 (0=object, 255=background)
        grey_frame: Frame from grey_mask.mp4 (127=object, 255=background)

    Returns:
        Combined mask frame
    """
    # Initialize with background (255)
    combined = np.full_like(black_frame, 255, dtype=np.uint8)

    # Primary object only (black=0, grey=255)
    primary_only = (black_frame == 0) & (grey_frame == 255)
    combined[primary_only] = 0

    # Affected object only (black=255, grey=127)
    affected_only = (black_frame == 255) & (grey_frame == 127)
    combined[affected_only] = 127

    # Overlap (black=0, grey=127)
    overlap = (black_frame == 0) & (grey_frame == 127)
    combined[overlap] = 63

    return combined


def process_video(black_mask_path: Path, grey_mask_path: Path, output_path: Path):
    """Combine black and grey mask videos into trimask/quadmask"""
    import subprocess

    print(f"   Loading black mask: {black_mask_path.name}")
    black_cap = cv2.VideoCapture(str(black_mask_path))

    print(f"   Loading grey mask: {grey_mask_path.name}")
    grey_cap = cv2.VideoCapture(str(grey_mask_path))

    # Get video properties
    fps = black_cap.get(cv2.CAP_PROP_FPS)
    width = int(black_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(black_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(black_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Check grey mask has same properties
    grey_total_frames = int(grey_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames != grey_total_frames:
        print(f"   ⚠️  Warning: Frame count mismatch (black: {total_frames}, grey: {grey_total_frames})")
        total_frames = min(total_frames, grey_total_frames)

    print(f"   Video: {width}x{height} @ {fps:.2f}fps, {total_frames} frames")
    print(f"   Combining masks...")

    # Collect all frames first
    combined_frames = []

    # Process frames
    for frame_idx in tqdm(range(total_frames), desc="   Combining"):
        ret_black, black_frame = black_cap.read()
        ret_grey, grey_frame = grey_cap.read()

        if not ret_black or not ret_grey:
            print(f"   ⚠️  Warning: Could not read frame {frame_idx}")
            break

        # Convert to grayscale if needed
        if len(black_frame.shape) == 3:
            black_frame = cv2.cvtColor(black_frame, cv2.COLOR_BGR2GRAY)
        if len(grey_frame.shape) == 3:
            grey_frame = cv2.cvtColor(grey_frame, cv2.COLOR_BGR2GRAY)

        # Combine
        combined_frame = combine_masks(black_frame, grey_frame)
        combined_frames.append(combined_frame)

    # Cleanup
    black_cap.release()
    grey_cap.release()

    # On the first frame, clamp near-grey values (100–135) to 255 (background).
    # Video codecs can introduce slight luma drift around 127; this ensures no
    # grey pixels survive into the final quadmask on frame 0.
    if combined_frames:
        f0 = combined_frames[0]
        grey_pixels = (f0 > 100) & (f0 < 135)
        f0[grey_pixels] = 255
        combined_frames[0] = f0

    # Write using LOSSLESS encoding to preserve exact mask values
    print(f"   Writing lossless video...")

    # Write temp AVI with FFV1 codec (lossless)
    temp_avi = output_path.with_suffix('.avi')
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    out = cv2.VideoWriter(str(temp_avi), fourcc, fps, (width, height), isColor=False)

    for frame in combined_frames:
        out.write(frame)
    out.release()

    # Convert to LOSSLESS H.264 (qp=0, yuv444p to preserve all luma values)
    cmd = [
        'ffmpeg', '-y', '-i', str(temp_avi),
        '-c:v', 'libx264', '-qp', '0', '-preset', 'ultrafast',
        '-pix_fmt', 'yuv444p',
        '-r', str(fps),
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"   ⚠️  Warning: ffmpeg conversion had issues")
        print(result.stderr)

    # Clean up temp file
    temp_avi.unlink()

    print(f"   ✓ Saved: {output_path.name}")


def process_config(config_path: str):
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
    print(f"Stage 4: Combine Masks into Tri/Quad Mask")
    print(f"{'='*70}")
    print(f"Config: {config_path.name}")
    print(f"Videos: {len(videos)}")
    print(f"{'='*70}\n")

    # Process each video
    success_count = 0
    for i, video_info in enumerate(videos):
        video_path = video_info.get("video_path", "")
        output_dir = video_info.get("output_dir", "")

        print(f"\n{'─'*70}")
        print(f"Video {i+1}/{len(videos)}: {Path(video_path).name}")
        print(f"{'─'*70}")

        if not output_dir:
            print(f"   ⚠️  No output_dir specified, skipping")
            continue

        output_dir = Path(output_dir)
        if not output_dir.exists():
            print(f"   ⚠️  Output directory not found: {output_dir}")
            continue

        # Check for required masks
        black_mask_path = output_dir / "black_mask.mp4"
        grey_mask_path = output_dir / "grey_mask.mp4"

        if not black_mask_path.exists():
            print(f"   ⚠️  black_mask.mp4 not found, skipping")
            continue

        if not grey_mask_path.exists():
            print(f"   ⚠️  grey_mask.mp4 not found, skipping")
            continue

        # Output path
        output_path = output_dir / "quadmask_0.mp4"

        try:
            process_video(black_mask_path, grey_mask_path, output_path)
            success_count += 1
            print(f"\n✅ Video {i+1} complete!")
        except Exception as e:
            print(f"\n❌ Error processing video {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    print(f"\n{'='*70}")
    print(f"Stage 4 Complete!")
    print(f"{'='*70}")
    print(f"Successful: {success_count}/{len(videos)}")
    print(f"Failed: {len(videos) - success_count}/{len(videos)}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 4: Combine black and grey masks into tri/quad mask"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config JSON (with output_dir for each video)"
    )

    args = parser.parse_args()
    process_config(args.config)


if __name__ == "__main__":
    main()
