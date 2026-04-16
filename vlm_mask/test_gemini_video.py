#!/usr/bin/env python3
"""
Quick Gemini API smoke test — samples a few frames from a video and asks a
simple question. Use this to verify your API key works before running the
full pipeline.

Usage:
    export GEMINI_API_KEY="your_aistudio_key"
    python test_gemini_video.py --video path/to/video.mp4
"""

import os
import sys
import base64
import argparse
import cv2
import numpy as np
from pathlib import Path

import openai


FREE_TIER_MODEL = "gemini-2.0-flash"
NUM_FRAMES = 4  # keep low for free tier rate limits


def sample_frames(video_path: str, n: int = NUM_FRAMES):
    """Sample n evenly-spaced frames from the video, return as base64 data URLs."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(i * (total - 1) / (n - 1)) for i in range(n)]

    data_urls = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64 = base64.b64encode(buf).decode("utf-8")
        data_urls.append(f"data:image/jpeg;base64,{b64}")

    cap.release()
    return data_urls


def main():
    parser = argparse.ArgumentParser(description="Gemini API smoke test with video frames")
    parser.add_argument("--video", required=True, help="Path to a video file")
    parser.add_argument("--model", default=FREE_TIER_MODEL, help="Gemini model to use")
    parser.add_argument("--frames", type=int, default=NUM_FRAMES, help="Number of frames to sample")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set")
        sys.exit(1)

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    print(f"Video:  {video_path.name}")
    print(f"Model:  {args.model}")
    print(f"Frames: {args.frames}")
    print()

    print(f"Sampling {args.frames} frames...")
    data_urls = sample_frames(str(video_path), args.frames)
    print(f"Got {len(data_urls)} frames. Sending to Gemini...")

    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    content = [
        {"type": "image_url", "image_url": {"url": url}} for url in data_urls
    ]
    content.append({
        "type": "text",
        "text": "These are evenly-spaced frames from a short video. In one sentence, describe what is happening in the video."
    })

    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": content}],
    )

    print("\n--- Gemini response ---")
    print(response.choices[0].message.content)
    print("-----------------------")
    print("\n✅ API key works!")


if __name__ == "__main__":
    main()
