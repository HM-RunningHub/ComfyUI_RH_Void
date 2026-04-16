#!/usr/bin/env python3
"""
Stage 2: VLM Analysis - Identify Affected Objects & Physics

Analyzes videos with primary masks to identify:
- Integral belongings (to add to black mask)
- Affected objects (shadows, reflections, held items)
- Physics behavior (will_move, needs_trajectory)

Input:  Config from Stage 1 (with output_dir containing black_mask.mp4, first_frame.jpg)
Output: For each video:
        - vlm_analysis.json: Identified objects and physics reasoning

Usage:
    python stage2_vlm_analysis.py --config more_dyn_2_config_points_absolute.json
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
import base64
from pathlib import Path
from typing import Dict, List
from PIL import Image, ImageDraw

import openai

DEFAULT_MODEL = "gemini-3-flash-preview"
# DEFAULT_MODEL = 'gemini-3-pro-preview'


def _print_api_exception(prefix: str, exc: Exception):
    """Print richer API error diagnostics for easier debugging."""
    print(f"{prefix}: {type(exc).__name__}: {exc}")
    print(f"{prefix} repr: {exc!r}")

    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        print(f"{prefix} status_code: {status_code}")

    response = getattr(exc, "response", None)
    if response is not None:
        resp_status = getattr(response, "status_code", None)
        if resp_status is not None:
            print(f"{prefix} response.status_code: {resp_status}")
        text = getattr(response, "text", None)
        if text:
            print(f"{prefix} response.text: {text}")
        try:
            json_body = response.json()
            print(f"{prefix} response.json: {json.dumps(json_body, ensure_ascii=False)}")
        except Exception:
            pass

    body = getattr(exc, "body", None)
    if body:
        print(f"{prefix} body: {body}")


def image_to_data_url(image_path: str) -> str:
    """Convert image file to base64 data URL"""
    with open(image_path, 'rb') as f:
        img_data = base64.b64encode(f.read()).decode('utf-8')

    # Detect format
    ext = Path(image_path).suffix.lower()
    if ext == '.png':
        mime = 'image/png'
    elif ext in ['.jpg', '.jpeg']:
        mime = 'image/jpeg'
    else:
        mime = 'image/jpeg'

    return f"data:{mime};base64,{img_data}"


def video_to_data_url(video_path: str) -> str:
    """Convert video file to base64 data URL"""
    with open(video_path, 'rb') as f:
        video_data = base64.b64encode(f.read()).decode('utf-8')
    return f"data:video/mp4;base64,{video_data}"


def calculate_square_grid(width: int, height: int, min_grid: int = 8) -> tuple:
    """Calculate grid dimensions matching stage3a logic"""
    aspect_ratio = width / height
    if width >= height:
        grid_rows = min_grid
        grid_cols = max(min_grid, round(min_grid * aspect_ratio))
    else:
        grid_cols = min_grid
        grid_rows = max(min_grid, round(min_grid / aspect_ratio))
    return grid_rows, grid_cols


def _load_reference_frame(first_frame_path: str) -> np.ndarray:
    """Load the Stage 1 reference frame saved for VLM analysis."""
    frame = cv2.imread(first_frame_path)
    if frame is None:
        raise ValueError(f"Failed to load reference frame: {first_frame_path}")
    return frame


def create_first_frame_with_mask_overlay(first_frame_path: str, black_mask_path: str,
                                          output_path: str, frame_idx: int = 0,
                                          video_path: str = None) -> str:
    """Create visualization of first frame with red overlay on primary object

    Args:
        first_frame_path: Path to first_frame.jpg
        black_mask_path: Path to black_mask.mp4
        output_path: Where to save overlay
        frame_idx: Which frame to extract from black_mask.mp4 (default: 0)
    """
    # Stage 1 already saved first_frame.jpg at the VLM reference frame.
    frame = _load_reference_frame(first_frame_path)

    # first_frame.jpg is already the Stage 1 reference frame, but black_mask.mp4 is
    # now full-length, so we need to seek to the matching source frame.
    cap = cv2.VideoCapture(black_mask_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, mask_frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Failed to load black mask frame {frame_idx}: {black_mask_path}")

    # Convert mask to binary (0 = object, 255 = background)
    if len(mask_frame.shape) == 3:
        mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)

    object_mask = (mask_frame == 0)

    # Create red overlay on object
    overlay = frame.copy()
    overlay[object_mask] = [0, 0, 255]  # Red in BGR

    # Blend: 60% original + 40% red overlay
    result = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

    # Save
    cv2.imwrite(output_path, result)
    return output_path


def create_gridded_frame_overlay(first_frame_path: str, black_mask_path: str,
                                  output_path: str, min_grid: int = 8,
                                  frame_idx: int = 0, video_path: str = None) -> tuple:
    """Create first frame with BOTH red mask overlay AND grid lines

    Returns: (output_path, grid_rows, grid_cols)
    """
    # Stage 1 already saved first_frame.jpg at the VLM reference frame.
    frame = _load_reference_frame(first_frame_path)

    h, w = frame.shape[:2]

    # Load black mask
    cap = cv2.VideoCapture(black_mask_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, mask_frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Failed to load black mask frame {frame_idx}: {black_mask_path}")

    if len(mask_frame.shape) == 3:
        mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)

    object_mask = (mask_frame == 0)

    # Create red overlay
    overlay = frame.copy()
    overlay[object_mask] = [0, 0, 255]
    result = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

    # Calculate grid
    grid_rows, grid_cols = calculate_square_grid(w, h, min_grid)

    # Draw grid lines
    cell_width = w / grid_cols
    cell_height = h / grid_rows

    # Vertical lines
    for col in range(1, grid_cols):
        x = int(col * cell_width)
        cv2.line(result, (x, 0), (x, h), (255, 255, 0), 1)  # Yellow lines

    # Horizontal lines
    for row in range(1, grid_rows):
        y = int(row * cell_height)
        cv2.line(result, (0, y), (w, y), (255, 255, 0), 1)

    # Add grid labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    thickness = 1

    # Label columns at top
    for col in range(grid_cols):
        x = int((col + 0.5) * cell_width)
        cv2.putText(result, str(col), (x-5, 15), font, font_scale, (255, 255, 0), thickness)

    # Label rows on left
    for row in range(grid_rows):
        y = int((row + 0.5) * cell_height)
        cv2.putText(result, str(row), (5, y+5), font, font_scale, (255, 255, 0), thickness)

    cv2.imwrite(output_path, result)
    return output_path, grid_rows, grid_cols


def create_multi_frame_grid_samples(video_path: str, output_dir: Path,
                                      min_grid: int = 8,
                                      sample_points: list = [0.0, 0.11, 0.22, 0.33, 0.44, 0.56, 0.67, 0.78, 0.89, 1.0]) -> tuple:
    """
    Create gridded frame samples at multiple time points in video.
    Helps VLM see objects that appear mid-video with grid reference.

    Args:
        video_path: Path to video
        output_dir: Where to save samples
        min_grid: Minimum grid size
        sample_points: List of normalized positions [0.0-1.0] to sample

    Returns: (sample_paths, grid_rows, grid_cols)
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate grid (same for all frames)
    grid_rows, grid_cols = calculate_square_grid(w, h, min_grid)
    cell_width = w / grid_cols
    cell_height = h / grid_rows

    sample_paths = []

    for i, t in enumerate(sample_points):
        frame_idx = int(t * (total_frames - 1))
        frame_idx = max(0, min(frame_idx, total_frames - 1))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Draw grid
        result = frame.copy()

        # Vertical lines
        for col in range(1, grid_cols):
            x = int(col * cell_width)
            cv2.line(result, (x, 0), (x, h), (255, 255, 0), 2)

        # Horizontal lines
        for row in range(1, grid_rows):
            y = int(row * cell_height)
            cv2.line(result, (0, y), (w, y), (255, 255, 0), 2)

        # Add grid labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1

        # Label columns
        for col in range(grid_cols):
            x = int((col + 0.5) * cell_width)
            cv2.putText(result, str(col), (x-8, 20), font, font_scale, (255, 255, 0), thickness)

        # Label rows
        for row in range(grid_rows):
            y = int((row + 0.5) * cell_height)
            cv2.putText(result, str(row), (10, y+8), font, font_scale, (255, 255, 0), thickness)

        # Add frame number and percentage
        label = f"Frame {frame_idx} ({int(t*100)}%)"
        cv2.putText(result, label, (10, h-10), font, 0.5, (255, 255, 0), 2)

        # Save
        output_path = output_dir / f"grid_sample_frame_{frame_idx:04d}.jpg"
        cv2.imwrite(str(output_path), result)
        sample_paths.append(output_path)

    cap.release()
    return sample_paths, grid_rows, grid_cols


def make_vlm_analysis_prompt(instruction: str, grid_rows: int, grid_cols: int,
                              has_multi_frame_grids: bool = False) -> str:
    """Create VLM prompt for analyzing video with primary mask"""

    grid_context = ""
    if has_multi_frame_grids:
        grid_context = f"""
1. **Multiple Grid Reference Frames**: Sampled frames at 0%, 11%, 22%, 33%, 44%, 56%, 67%, 78%, 89%, 100% of video
   - Each frame shows YELLOW GRID with {grid_rows} rows × {grid_cols} columns
   - Grid cells labeled (row, col) starting from (0, 0) at top-left
   - Frame number shown at bottom
   - Use these to locate objects that appear MID-VIDEO and track object positions across time
2. **First Frame with RED mask**: Shows what will be REMOVED (primary object)
3. **Full Video**: Complete action and interactions"""
    else:
        grid_context = f"""
1. **First Frame with Grid**: PRIMARY OBJECT highlighted in RED + GRID OVERLAY
   - The red overlay shows what will be REMOVED (already masked)
   - Yellow grid with {grid_rows} rows × {grid_cols} columns
   - Grid cells are labeled (row, col) starting from (0, 0) at top-left
2. **Full Video**: Complete scene and action"""

    return f"""
You are an expert video analyst specializing in physics and object interactions.

═══════════════════════════════════════════════════════════════════
CONTEXT
═══════════════════════════════════════════════════════════════════

You will see MULTIPLE inputs:
{grid_context}

Edit instruction: "{instruction}"

IMPORTANT: Some objects may NOT appear in first frame. They may enter later.
Watch the ENTIRE video and note when each object first appears.

═══════════════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════════════

Analyze what would happen if the PRIMARY OBJECT (shown in red) is removed.
Watch the ENTIRE video to see all interactions and movements.

STEP 1: IDENTIFY INTEGRAL BELONGINGS (0-3 items)
─────────────────────────────────────────────────
Items that should be ADDED to the primary removal mask (removed WITH primary object):

✓ INCLUDE:
  • Distinct wearable items: hat, backpack, jacket (if separate/visible)
  • Vehicles/equipment being ridden: bike, skateboard, surfboard, scooter
  • Large carried items that are part of the subject

✗ DO NOT INCLUDE:
  • Generic clothing (shirt, pants, shoes) - already captured with person
  • Held items that could be set down: guitar, cup, phone, tools
  • Objects they're interacting with but not wearing/riding

Examples:
  • Person on bike → integral: "bike"
  • Person with guitar → integral: none (guitar is affected, not integral)
  • Surfer → integral: "surfboard"
  • Boxer → integral: "boxing gloves" (wearable equipment)

STEP 2: IDENTIFY AFFECTED OBJECTS (0-5 objects)
────────────────────────────────────────────────
Objects/effects that are SEPARATE from primary but affected by its removal.

CRITICAL: Do NOT include integral belongings from Step 1.

Two categories:

A) VISUAL ARTIFACTS (disappear when primary removed):
   • shadow, reflection, wake, ripples, splash, footprints
   • These vanish completely - no physics needed

   **CRITICAL FOR VISUAL ARTIFACTS:**
   You MUST provide GRID LOCALIZATIONS across the reference frames.
   Keyword segmentation fails to isolate specific shadows/reflections.

   For each visual artifact:
   - Look at each grid reference frame you were shown
   - Identify which grid cells the artifact occupies in EACH frame
   - List all grid cells (row, col) that contain any part of it
   - Be thorough - include ALL touched cells (over-mask is better than under-mask)

   Format:
   {{
     "noun": "shadow",
     "category": "visual_artifact",
     "grid_localizations": [
       {{"frame": 0, "grid_regions": [{{"row": 6, "col": 3}}, {{"row": 6, "col": 4}}, ...]}},
       {{"frame": 5, "grid_regions": [{{"row": 6, "col": 4}}, ...]}},
       // ... for each reference frame shown
     ]
   }}

B) PHYSICAL OBJECTS (may move, fall, or stay):

   CRITICAL - Understand the difference:

   **SUPPORTING vs ACTING ON:**
   • SUPPORTING = holding UP against gravity → object WILL FALL when removed
     Examples: holding guitar, carrying cup, person sitting on chair
     → will_move: TRUE

   • ACTING ON = touching/manipulating but object rests on stable surface → object STAYS
     Examples: hand crushing can (can on table), hand opening can (can on counter),
              hand pushing object (object on floor)
     → will_move: FALSE

   **Key Questions:**
   1. Is the primary object HOLDING THIS UP against gravity?
      - YES → will_move: true, needs_trajectory: true
      - NO → Check next question

   2. Is this object RESTING ON a stable surface (table, floor, counter)?
      - YES → will_move: false (stays on surface when primary removed)
      - NO → will_move: true

   3. Is the primary object DOING an action TO this object?
      - Opening can, crushing can, pushing button, turning knob
      - When primary removed → action STOPS, object stays in current state
      - will_move: false

   **SPECIAL CASE - Object Currently Moving But Should Have Stayed:**
   If primary object CAUSES another object to move (hitting, kicking, throwing):
   - The object is currently moving in the video
   - But WITHOUT primary, it would have stayed at its original position
   - You MUST provide:
     • "currently_moving": true
     • "should_have_stayed": true
     • "original_position_grid": {{"row": R, "col": C}} - Where it started

   Examples:
   - Golf club hits ball → Ball at tee, then flies (mark original tee position)
   - Person kicks soccer ball → Ball on ground, then rolls (mark original ground position)
   - Hand throws object → Object held, then flies (mark original held position)

   Format:
   {{
     "noun": "golf ball",
     "category": "physical",
     "currently_moving": true,
     "should_have_stayed": true,
     "original_position_grid": {{"row": 6, "col": 7}},
     "why": "ball was stationary until club hit it"
   }}

   For each physical object, determine:
   - **will_move**: true ONLY if object will fall/move when support removed
   - **first_appears_frame**: frame number object first appears (0 if from start)
   - **why**: Brief explanation of relationship to primary object

   IF will_move=TRUE, also provide GRID-BASED TRAJECTORY:
   - **object_size_grids**: {{"rows": R, "cols": C}} - How many grid cells object occupies
     IMPORTANT: Add 1 extra cell padding for safety (better to over-mask than under-mask)
     Example: Object looks 2×1 → report as 3×2

   - **trajectory_path**: List of keyframe positions as grid coordinates
     Format: [{{"frame": N, "grid_row": R, "grid_col": C}}, ...]
     - IMPORTANT: First keyframe should be at first_appears_frame (not frame 0 if object appears later!)
     - Provide 3-5 keyframes spanning from first appearance to end
     - (grid_row, grid_col) is the CENTER position of object at that frame
     - Use the yellow grid reference frames to determine positions
     - For objects appearing mid-video: use the grid samples to locate them
     - Example: Object appears at frame 15, falls to bottom
       [{{"frame": 15, "grid_row": 3, "grid_col": 5}},  ← First appearance
        {{"frame": 25, "grid_row": 6, "grid_col": 5}},  ← Mid-fall
        {{"frame": 35, "grid_row": 9, "grid_col": 5}}]  ← On ground

✓ Objects held/carried at ANY point in video
✓ Objects the primary supports or interacts with
✓ Visual effects visible at any time

✗ Background objects never touched
✗ Other people/animals with no contact
✗ Integral belongings (already in Step 1)

STEP 3: SCENE DESCRIPTION
──────────────────────────
Describe scene WITHOUT the primary object (1-2 sentences).
Focus on what remains and any dynamic changes (falling objects, etc).

═══════════════════════════════════════════════════════════════════
OUTPUT FORMAT (STRICT JSON ONLY)
═══════════════════════════════════════════════════════════════════

EXAMPLES TO LEARN FROM:

Example 1: Person holding guitar
{{
  "affected_objects": [
    {{
      "noun": "guitar",
      "will_move": true,
      "why": "person is SUPPORTING guitar against gravity by holding it",
      "object_size_grids": {{"rows": 3, "cols": 2}},
      "trajectory_path": [
        {{"frame": 0, "grid_row": 4, "grid_col": 5}},
        {{"frame": 15, "grid_row": 6, "grid_col": 5}},
        {{"frame": 30, "grid_row": 8, "grid_col": 6}}
      ]
    }}
  ]
}}

Example 2: Hand crushing can on table
{{
  "affected_objects": [
    {{
      "noun": "can",
      "will_move": false,
      "why": "can RESTS ON TABLE - hand is just acting on it. When hand removed, can stays on table (uncrushed)"
    }}
  ]
}}

Example 3: Hands opening can on counter
{{
  "affected_objects": [
    {{
      "noun": "can",
      "will_move": false,
      "why": "can RESTS ON COUNTER - hands are doing opening action. When hands removed, can stays closed on counter"
    }}
  ]
}}

Example 4: Person sitting on chair
{{
  "affected_objects": [
    {{
      "noun": "chair",
      "will_move": false,
      "why": "chair RESTS ON FLOOR - person sitting on it doesn't make it fall. Chair stays on floor when person removed"
    }}
  ]
}}

Example 5: Person throws ball (ball appears at frame 12)
{{
  "affected_objects": [
    {{
      "noun": "ball",
      "category": "physical",
      "will_move": true,
      "first_appears_frame": 12,
      "why": "ball is SUPPORTED by person's hand, then thrown",
      "object_size_grids": {{"rows": 2, "cols": 2}},
      "trajectory_path": [
        {{"frame": 12, "grid_row": 4, "grid_col": 3}},
        {{"frame": 20, "grid_row": 2, "grid_col": 6}},
        {{"frame": 28, "grid_row": 5, "grid_col": 8}}
      ]
    }}
  ]
}}

Example 6: Person with shadow (shadow needs grid localization)
{{
  "affected_objects": [
    {{
      "noun": "shadow",
      "category": "visual_artifact",
      "why": "cast by person on the floor",
      "will_move": false,
      "first_appears_frame": 0,
      "movement_description": "Disappears entirely as visual artifact",
      "grid_localizations": [
        {{"frame": 0, "grid_regions": [{{"row": 6, "col": 3}}, {{"row": 6, "col": 4}}, {{"row": 7, "col": 3}}, {{"row": 7, "col": 4}}]}},
        {{"frame": 12, "grid_regions": [{{"row": 6, "col": 4}}, {{"row": 6, "col": 5}}, {{"row": 7, "col": 4}}]}},
        {{"frame": 23, "grid_regions": [{{"row": 5, "col": 4}}, {{"row": 6, "col": 4}}, {{"row": 6, "col": 5}}]}},
        {{"frame": 35, "grid_regions": [{{"row": 6, "col": 3}}, {{"row": 6, "col": 4}}, {{"row": 7, "col": 3}}]}},
        {{"frame": 47, "grid_regions": [{{"row": 6, "col": 3}}, {{"row": 7, "col": 3}}, {{"row": 7, "col": 4}}]}}
      ]
    }}
  ]
}}

Example 7: Golf club hits ball (Case 4 - currently moving but should stay)
{{
  "affected_objects": [
    {{
      "noun": "golf ball",
      "category": "physical",
      "currently_moving": true,
      "should_have_stayed": true,
      "original_position_grid": {{"row": 6, "col": 7}},
      "first_appears_frame": 0,
      "why": "ball was stationary on tee until club hit it. Without club, ball would remain at original position."
    }}
  ]
}}

YOUR OUTPUT FORMAT:
{{
  "edit_instruction": "{instruction}",
  "integral_belongings": [
    {{
      "noun": "bike",
      "why": "person is riding the bike throughout the video"
    }}
  ],
  "affected_objects": [
    {{
      "noun": "guitar",
      "category": "physical",
      "why": "person is SUPPORTING guitar against gravity by holding it",
      "will_move": true,
      "first_appears_frame": 0,
      "movement_description": "Will fall from held position to the ground",
      "object_size_grids": {{"rows": 3, "cols": 2}},
      "trajectory_path": [
        {{"frame": 0, "grid_row": 3, "grid_col": 6}},
        {{"frame": 20, "grid_row": 6, "grid_col": 6}},
        {{"frame": 40, "grid_row": 9, "grid_col": 7}}
      ]
    }},
    {{
      "noun": "shadow",
      "category": "visual_artifact",
      "why": "cast by person on floor",
      "will_move": false,
      "first_appears_frame": 0,
      "movement_description": "Disappears entirely as visual artifact"
    }}
  ],
  "scene_description": "An acoustic guitar falling to the ground in an empty room. Natural window lighting.",
  "confidence": 0.85
}}

CRITICAL REMINDERS:
• Watch ENTIRE video before answering
• SUPPORTING vs ACTING ON:
  - Primary HOLDS UP object against gravity → will_move=TRUE (provide grid trajectory)
  - Primary ACTS ON object (crushing, opening) but object on stable surface → will_move=FALSE
  - Object RESTS ON stable surface (table, floor) → will_move=FALSE
• For visual artifacts (shadow, reflection): will_move=false (no trajectory needed)
• For held objects (guitar, cup): will_move=true (MUST provide object_size_grids + trajectory_path)
• For objects on surfaces being acted on (can being crushed, can being opened): will_move=false
• Grid trajectory: Add +1 cell padding to object size (over-mask is better than under-mask)
• Grid trajectory: Use the yellow grid overlay to determine (row, col) positions
• Be conservative - when in doubt, DON'T include
• Output MUST be valid JSON only

GRID INFO: {grid_rows} rows × {grid_cols} columns
EDIT INSTRUCTION: {instruction}
""".strip()


def call_vlm_with_images_and_video(client, model: str, image_data_urls: list,
                                    video_data_url: str, prompt: str) -> str:
    """Call VLM with multiple images and video"""
    content = []

    # Add all images first
    for img_url in image_data_urls:
        content.append({"type": "image_url", "image_url": {"url": img_url}})

    # Add video
    content.append({"type": "image_url", "image_url": {"url": video_data_url}})

    # Add prompt
    content.append({"type": "text", "text": prompt})

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert video analyst with deep understanding of physics and object interactions. Always output valid JSON only."
                },
                {
                    "role": "user",
                    "content": content
                },
            ],
        )
    except Exception as exc:
        _print_api_exception("   API request failed", exc)
        raise
    return resp.choices[0].message.content


def parse_vlm_response(raw: str) -> Dict:
    """Parse VLM JSON response"""
    # Strip markdown code blocks
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split('\n')
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = '\n'.join(lines)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON in response
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = json.loads(cleaned[start:end+1])
        else:
            raise ValueError("Failed to parse VLM response as JSON")

    # Validate structure
    result = {
        "edit_instruction": parsed.get("edit_instruction", ""),
        "integral_belongings": [],
        "affected_objects": [],
        "scene_description": parsed.get("scene_description", ""),
        "confidence": float(parsed.get("confidence", 0.0))
    }

    # Parse integral belongings
    for item in parsed.get("integral_belongings", [])[:3]:
        obj = {
            "noun": str(item.get("noun", "")).strip().lower(),
            "why": str(item.get("why", "")).strip()[:200]
        }
        if obj["noun"]:
            result["integral_belongings"].append(obj)

    # Parse affected objects
    for item in parsed.get("affected_objects", [])[:5]:
        obj = {
            "noun": str(item.get("noun", "")).strip().lower(),
            "category": str(item.get("category", "physical")).strip().lower(),
            "why": str(item.get("why", "")).strip()[:200],
            "will_move": bool(item.get("will_move", False)),
            "first_appears_frame": int(item.get("first_appears_frame", 0)),
            "movement_description": str(item.get("movement_description", "")).strip()[:300]
        }

        # Parse Case 4: currently moving but should have stayed
        if "currently_moving" in item:
            obj["currently_moving"] = bool(item.get("currently_moving", False))
        if "should_have_stayed" in item:
            obj["should_have_stayed"] = bool(item.get("should_have_stayed", False))
        if "original_position_grid" in item:
            orig_grid = item.get("original_position_grid", {})
            obj["original_position_grid"] = {
                "row": int(orig_grid.get("row", 0)),
                "col": int(orig_grid.get("col", 0))
            }

        # Parse grid localizations for visual artifacts
        if "grid_localizations" in item:
            grid_locs = []
            for loc in item.get("grid_localizations", []):
                frame_loc = {
                    "frame": int(loc.get("frame", 0)),
                    "grid_regions": []
                }
                for region in loc.get("grid_regions", []):
                    frame_loc["grid_regions"].append({
                        "row": int(region.get("row", 0)),
                        "col": int(region.get("col", 0))
                    })
                if frame_loc["grid_regions"]:  # Only add if has regions
                    grid_locs.append(frame_loc)
            if grid_locs:
                obj["grid_localizations"] = grid_locs

        # Parse grid trajectory if will_move=true
        if obj["will_move"] and "object_size_grids" in item and "trajectory_path" in item:
            size_grids = item.get("object_size_grids", {})
            obj["object_size_grids"] = {
                "rows": int(size_grids.get("rows", 2)),
                "cols": int(size_grids.get("cols", 2))
            }

            trajectory = []
            for point in item.get("trajectory_path", []):
                trajectory.append({
                    "frame": int(point.get("frame", 0)),
                    "grid_row": int(point.get("grid_row", 0)),
                    "grid_col": int(point.get("grid_col", 0))
                })

            if trajectory:  # Only add if we have valid trajectory points
                obj["trajectory_path"] = trajectory

        if obj["noun"]:
            result["affected_objects"].append(obj)

    return result


def process_video(video_info: Dict, client, model: str):
    """Process a single video with VLM analysis"""
    video_path = video_info.get("video_path", "")
    instruction = video_info.get("instruction", "")
    output_dir = video_info.get("output_dir", "")

    if not output_dir:
        print(f"   ⚠️  No output_dir specified, skipping")
        return None

    output_dir = Path(output_dir)
    if not output_dir.exists():
        print(f"   ⚠️  Output directory not found: {output_dir}")
        print(f"   Run Stage 1 first to create black masks")
        return None

    # Check required files from Stage 1
    black_mask_path = output_dir / "black_mask.mp4"
    first_frame_path = output_dir / "first_frame.jpg"
    input_video_path = output_dir / "input_video.mp4"
    segmentation_info_path = output_dir / "segmentation_info.json"

    if not black_mask_path.exists():
        print(f"   ⚠️  black_mask.mp4 not found in {output_dir}")
        print(f"   Run Stage 1 first")
        return None

    if not first_frame_path.exists():
        print(f"   ⚠️  first_frame.jpg not found in {output_dir}")
        return None

    if not input_video_path.exists():
        # Try original video path
        if Path(video_path).exists():
            input_video_path = Path(video_path)
        else:
            print(f"   ⚠️  Video not found: {video_path}")
            return None

    # Read segmentation metadata to get correct frame index
    frame_idx = 0  # Default
    if segmentation_info_path.exists():
        try:
            with open(segmentation_info_path, 'r') as f:
                seg_info = json.load(f)
                frame_idx = seg_info.get("first_appears_frame", 0)
                print(f"   Using frame {frame_idx} from segmentation metadata")
        except Exception as e:
            print(f"   Warning: Could not read segmentation_info.json: {e}")
            print(f"   Using frame 0 as fallback")

    # Get min_grid for grid calculation
    min_grid = video_info.get('min_grid', 8)
    use_multi_frame_grids = video_info.get('multi_frame_grids', True)  # Default: use multi-frame
    max_video_size_mb = video_info.get('max_video_size_for_multiframe', 25)  # Default: 25MB limit

    # Check video size and auto-disable multi-frame for large videos
    if use_multi_frame_grids:
        video_size_mb = input_video_path.stat().st_size / (1024 * 1024)
        if video_size_mb > max_video_size_mb:
            print(f"   ⚠️  Video size ({video_size_mb:.1f} MB) exceeds {max_video_size_mb} MB")
            print(f"   Auto-disabling multi-frame grids to avoid API errors")
            use_multi_frame_grids = False

    print(f"   Creating frame overlays and grids...")
    overlay_path = output_dir / "first_frame_with_mask.jpg"
    gridded_path = output_dir / "first_frame_with_grid.jpg"

    # Create regular overlay (for backwards compatibility)
    create_first_frame_with_mask_overlay(
        str(first_frame_path),
        str(black_mask_path),
        str(overlay_path),
        frame_idx=frame_idx,
    )

    image_data_urls = []

    if use_multi_frame_grids:
        # Create multi-frame grid samples for objects appearing mid-video
        print(f"   Creating multi-frame grid samples (0%, 25%, 50%, 75%, 100%)...")
        sample_paths, grid_rows, grid_cols = create_multi_frame_grid_samples(
            str(input_video_path),
            output_dir,
            min_grid=min_grid
        )

        # Encode all grid samples
        for sample_path in sample_paths:
            image_data_urls.append(image_to_data_url(str(sample_path)))

        # Also add the first frame with mask overlay
        _, _, _ = create_gridded_frame_overlay(
            str(first_frame_path),
            str(black_mask_path),
            str(gridded_path),
            min_grid=min_grid,
            frame_idx=frame_idx,
        )
        image_data_urls.append(image_to_data_url(str(gridded_path)))

        print(f"   Grid: {grid_rows}x{grid_cols}, {len(sample_paths)} sample frames + masked frame")

    else:
        # Single gridded first frame (old approach)
        _, grid_rows, grid_cols = create_gridded_frame_overlay(
            str(first_frame_path),
            str(black_mask_path),
            str(gridded_path),
            min_grid=min_grid,
            frame_idx=frame_idx,
        )
        image_data_urls.append(image_to_data_url(str(gridded_path)))
        print(f"   Grid: {grid_rows}x{grid_cols} (single frame)")

    print(f"   Encoding video for VLM...")

    # Check video size
    video_size_mb = input_video_path.stat().st_size / (1024 * 1024)
    print(f"   Video size: {video_size_mb:.1f} MB")

    if video_size_mb > 20:
        print(f"   ⚠️  Warning: Large video may cause API errors")
        if use_multi_frame_grids:
            print(f"   Consider setting multi_frame_grids=false for large videos")

    video_data_url = video_to_data_url(str(input_video_path))

    print(f"   Calling {model}...")
    prompt = make_vlm_analysis_prompt(instruction, grid_rows, grid_cols,
                                       has_multi_frame_grids=use_multi_frame_grids)

    try:
        try:
            raw_response = call_vlm_with_images_and_video(
                client, model, image_data_urls, video_data_url, prompt
            )
        except Exception as e:
            _print_api_exception("   First VLM attempt failed", e)
            # If multi-frame fails (likely payload size issue), fall back to single frame
            if use_multi_frame_grids and "400" in str(e):
                print(f"   ⚠️  Multi-frame request failed (payload too large?)")
                print(f"   Falling back to single-frame grid mode...")

                # Retry with just the gridded first frame
                image_data_urls = [image_to_data_url(str(gridded_path))]
                prompt = make_vlm_analysis_prompt(instruction, grid_rows, grid_cols,
                                                   has_multi_frame_grids=False)

                try:
                    raw_response = call_vlm_with_images_and_video(
                        client, model, image_data_urls, video_data_url, prompt
                    )
                    print(f"   ✓ Single-frame fallback succeeded")
                except Exception as e2:
                    _print_api_exception("   Single-frame fallback failed", e2)
                    raise e2  # Re-raise if fallback also fails
            else:
                raise  # Re-raise if not a 400 or not multi-frame mode

        # Parse and save results (runs whether first call succeeded or fallback succeeded)
        print(f"   Parsing VLM response...")
        analysis = parse_vlm_response(raw_response)

        # Save results
        output_path = output_dir / "vlm_analysis.json"
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)

        print(f"   ✓ Saved VLM analysis: {output_path.name}")

        # Print summary
        print(f"\n   Summary:")
        print(f"   - Integral belongings: {len(analysis['integral_belongings'])}")
        for obj in analysis['integral_belongings']:
            print(f"     • {obj['noun']}: {obj['why']}")

        print(f"   - Affected objects: {len(analysis['affected_objects'])}")
        for obj in analysis['affected_objects']:
            move_str = "WILL MOVE" if obj['will_move'] else "STAYS/DISAPPEARS"
            traj_str = ""
            if obj.get('will_move') and 'trajectory_path' in obj:
                num_points = len(obj['trajectory_path'])
                size = obj.get('object_size_grids', {})
                traj_str = f" (trajectory: {num_points} keyframes, size: {size.get('rows')}×{size.get('cols')} grids)"
            print(f"     • {obj['noun']}: {move_str}{traj_str}")

        return analysis

    except Exception as e:
        _print_api_exception("   ❌ VLM analysis failed", e)
        import traceback
        traceback.print_exc()
        return None


def process_config(config_path: str, model: str = DEFAULT_MODEL, api_key: str = ""):
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
    print(f"Stage 2: VLM Analysis - Identify Affected Objects")
    print(f"{'='*70}")
    print(f"Config: {config_path.name}")
    print(f"Videos: {len(videos)}")
    print(f"Model: {model}")
    print(f"{'='*70}\n")

    # Initialize VLM client
    # api_key = os.environ.get("GEMINI_API_KEY")
    # if not api_key:
    #     raise RuntimeError("GEMINI_API_KEY environment variable not set")
    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    # Process each video
    results = []
    for i, video_info in enumerate(videos):
        video_path = video_info.get("video_path", "")
        instruction = video_info.get("instruction", "")

        print(f"\n{'─'*70}")
        print(f"Video {i+1}/{len(videos)}: {Path(video_path).name}")
        print(f"{'─'*70}")
        print(f"Instruction: {instruction}")

        try:
            analysis = process_video(video_info, client, model)
            results.append({
                "video": video_path,
                "success": analysis is not None,
                "analysis": analysis
            })

            if analysis:
                print(f"\n✅ Video {i+1} complete!")
            else:
                print(f"\n⚠️  Video {i+1} skipped")

        except Exception as e:
            print(f"\n❌ Error processing video {i+1}: {e}")
            results.append({
                "video": video_path,
                "success": False,
                "error": str(e)
            })
            continue

    # Summary
    print(f"\n{'='*70}")
    print(f"Stage 2 Complete!")
    print(f"{'='*70}")
    successful = sum(1 for r in results if r["success"])
    print(f"Successful: {successful}/{len(videos)}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Stage 2: VLM Analysis")
    parser.add_argument("--config", required=True, help="Config JSON from Stage 1")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="VLM model name")
    args = parser.parse_args()

    process_config(args.config, args.model)


if __name__ == "__main__":
    main()
