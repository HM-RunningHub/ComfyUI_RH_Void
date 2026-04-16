#!/usr/bin/env python3
"""
Stage 3b: Trajectory Drawing GUI (Simplified - No Segmentation)

For objects with needs_trajectory=true, user draws movement paths.

Input:  Config with vlm_analysis.json in output_dir
Output: trajectory_data.json with user-drawn paths as grid cells

Usage:
    python stage3b_trajectory_gui.py --config more_dyn_2_config_points_absolute.json
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
from pathlib import Path
from typing import Dict, List, Tuple


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


def points_to_grid_cells(points: List[Tuple[int, int]], grid_rows: int, grid_cols: int,
                         frame_width: int, frame_height: int) -> List[List[int]]:
    """Convert trajectory points to grid cells"""
    cell_width = frame_width / grid_cols
    cell_height = frame_height / grid_rows

    grid_cells = set()
    for x, y in points:
        col = int(x / cell_width)
        row = int(y / cell_height)
        if 0 <= row < grid_rows and 0 <= col < grid_cols:
            grid_cells.add((row, col))

    # Sort by row, then col
    return sorted([[r, c] for r, c in grid_cells])


class TrajectoryGUI:
    def __init__(self, root, objects_data: List[Dict]):
        self.root = root
        self.root.title("Stage 3b: Trajectory Drawing")

        self.objects_data = objects_data  # List of {video_info, objects_needing_trajectory}
        self.current_video_idx = 0
        self.current_object_idx = 0

        # Current state
        self.frame = None
        self.trajectory_points = []
        self.drawing = False

        # Display
        self.display_scale = 1.0
        self.photo = None

        # Results storage
        self.all_trajectories = []  # List of trajectories for all videos

        self.setup_ui()
        self.load_current_object()

    def setup_ui(self):
        """Setup GUI layout"""
        # Top info
        info_frame = ttk.Frame(self.root)
        info_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.video_label = ttk.Label(info_frame, text="Video: ", font=("Arial", 10, "bold"))
        self.video_label.pack(side=tk.LEFT, padx=5)

        self.object_label = ttk.Label(info_frame, text="Object: ", foreground="blue")
        self.object_label.pack(side=tk.LEFT, padx=10)

        # Instructions
        inst_frame = ttk.LabelFrame(self.root, text="Instructions")
        inst_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Label(inst_frame, text="1. See the frame where object is visible", foreground="blue").pack(anchor=tk.W, padx=5)
        ttk.Label(inst_frame, text="2. Click and drag to draw trajectory path (RED line)", foreground="red").pack(anchor=tk.W, padx=5)
        ttk.Label(inst_frame, text="3. Draw from object's current position to where it should end up", foreground="orange").pack(anchor=tk.W, padx=5)
        ttk.Label(inst_frame, text="4. Click 'Clear' to restart, 'Save & Next' when done", foreground="green").pack(anchor=tk.W, padx=5)

        # Canvas
        canvas_frame = ttk.LabelFrame(self.root, text="Draw Trajectory Path")
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(canvas_frame, width=800, height=600, bg='black', cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

        # Controls
        controls = ttk.Frame(self.root)
        controls.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        self.status_label = ttk.Label(controls, text="Draw trajectory path for object", foreground="blue")
        self.status_label.pack(side=tk.TOP, pady=5)

        button_frame = ttk.Frame(controls)
        button_frame.pack(side=tk.TOP)

        ttk.Button(button_frame, text="Clear Trajectory", command=self.clear_trajectory).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Skip Object", command=self.skip_object).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save & Next", command=self.save_and_next).pack(side=tk.LEFT, padx=5)

        self.progress_label = ttk.Label(controls, text="", font=("Arial", 9))
        self.progress_label.pack(side=tk.TOP, pady=5)

    def load_current_object(self):
        """Load current object for trajectory drawing"""
        if self.current_video_idx >= len(self.objects_data):
            # All done
            self.finish()
            return

        data = self.objects_data[self.current_video_idx]
        video_info = data['video_info']
        objects_needing_traj = data['objects']

        if self.current_object_idx >= len(objects_needing_traj):
            # Done with this video, move to next
            self.current_video_idx += 1
            self.current_object_idx = 0
            self.load_current_object()
            return

        obj = objects_needing_traj[self.current_object_idx]
        video_path = video_info.get('video_path', '')
        output_dir = Path(video_info.get('output_dir', ''))

        # Update labels
        self.video_label.config(text=f"Video: {Path(video_path).parent.name}/{Path(video_path).name}")
        self.object_label.config(text=f"Object: {obj['noun']} (will fall/move)")

        total_objects = sum(len(d['objects']) for d in self.objects_data)
        current_obj_num = sum(len(self.objects_data[i]['objects']) for i in range(self.current_video_idx)) + self.current_object_idx + 1
        self.progress_label.config(text=f"Object {current_obj_num}/{total_objects} across {len(self.objects_data)} video(s)")

        # Extract frame
        frame_idx = obj.get('first_appears_frame', 0)
        input_video = output_dir / "input_video.mp4"

        if not input_video.exists():
            input_video = Path(video_path)

        cap = cv2.VideoCapture(str(input_video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            messagebox.showerror("Error", f"Failed to read frame {frame_idx} from video")
            self.skip_object()
            return

        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        print(f"\n   Loaded frame {frame_idx} for '{obj['noun']}'")

        # Clear trajectory
        self.trajectory_points = []

        # Calculate grid for this video
        h, w = self.frame.shape[:2]
        min_grid = video_info.get('min_grid', 8)
        self.grid_rows, self.grid_cols = calculate_square_grid(w, h, min_grid)

        # Display
        self.status_label.config(text="Draw trajectory path (click and drag)", foreground="blue")
        self.display_frame()

    def display_frame(self):
        """Display frame with trajectory"""
        if self.frame is None:
            return

        # Create visualization
        vis = self.frame.copy()
        h, w = vis.shape[:2]

        # Draw trajectory
        if len(self.trajectory_points) > 1:
            for i in range(len(self.trajectory_points) - 1):
                pt1 = self.trajectory_points[i]
                pt2 = self.trajectory_points[i + 1]
                cv2.line(vis, pt1, pt2, (255, 0, 0), 5)  # Thicker line for visibility

            # Draw start point (green) and end point (red)
            if len(self.trajectory_points) > 0:
                start_pt = self.trajectory_points[0]
                end_pt = self.trajectory_points[-1]
                cv2.circle(vis, start_pt, 8, (0, 255, 0), -1)  # Green start
                cv2.circle(vis, end_pt, 8, (255, 0, 0), -1)    # Red end

        # Scale for display
        max_width, max_height = 800, 600
        scale_w = max_width / w
        scale_h = max_height / h
        self.display_scale = min(scale_w, scale_h, 1.0)

        new_w = int(w * self.display_scale)
        new_h = int(h * self.display_scale)
        vis_resized = cv2.resize(vis, (new_w, new_h))

        # Convert to PIL and display
        pil_img = Image.fromarray(vis_resized)
        self.photo = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def on_canvas_click(self, event):
        """Start drawing trajectory"""
        # Convert to frame coordinates
        x = int(event.x / self.display_scale)
        y = int(event.y / self.display_scale)

        self.trajectory_points = [(x, y)]
        self.drawing = True

    def on_canvas_drag(self, event):
        """Continue drawing trajectory"""
        if not self.drawing:
            return

        x = int(event.x / self.display_scale)
        y = int(event.y / self.display_scale)

        # Add point if far enough from last point
        if len(self.trajectory_points) > 0:
            last_x, last_y = self.trajectory_points[-1]
            dist = np.sqrt((x - last_x)**2 + (y - last_y)**2)
            if dist > 5:  # Minimum distance between points
                self.trajectory_points.append((x, y))
                self.display_frame()

    def on_canvas_release(self, event):
        """Finish drawing trajectory"""
        self.drawing = False
        if len(self.trajectory_points) > 0:
            x = int(event.x / self.display_scale)
            y = int(event.y / self.display_scale)
            self.trajectory_points.append((x, y))
            self.display_frame()

    def clear_trajectory(self):
        """Clear drawn trajectory"""
        self.trajectory_points = []
        self.display_frame()
        self.status_label.config(text="Trajectory cleared. Draw again.", foreground="blue")

    def skip_object(self):
        """Skip current object without saving trajectory"""
        result = messagebox.askyesno("Skip Object", "Skip this object without drawing trajectory?")
        if not result:
            return

        # Save empty trajectory
        data = self.objects_data[self.current_video_idx]
        obj = data['objects'][self.current_object_idx]

        self.all_trajectories.append({
            'video_path': data['video_info']['video_path'],
            'object_noun': obj['noun'],
            'trajectory_points': [],
            'trajectory_grid_cells': [],
            'skipped': True
        })

        self.current_object_idx += 1
        self.load_current_object()

    def save_and_next(self):
        """Save trajectory and move to next object"""
        if len(self.trajectory_points) < 2:
            messagebox.showwarning("Warning", "Draw a trajectory path first (at least 2 points)")
            return

        # Convert to grid cells
        data = self.objects_data[self.current_video_idx]
        obj = data['objects'][self.current_object_idx]

        grid_cells = points_to_grid_cells(
            self.trajectory_points,
            self.grid_rows,
            self.grid_cols,
            self.frame.shape[1],
            self.frame.shape[0]
        )

        # Save
        self.all_trajectories.append({
            'video_path': data['video_info']['video_path'],
            'object_noun': obj['noun'],
            'first_appears_frame': obj.get('first_appears_frame', 0),
            'trajectory_points': self.trajectory_points,
            'trajectory_grid_cells': grid_cells,
            'grid_rows': self.grid_rows,
            'grid_cols': self.grid_cols,
            'skipped': False
        })

        print(f"   ✓ Saved trajectory for '{obj['noun']}': {len(grid_cells)} grid cells")

        self.current_object_idx += 1
        self.load_current_object()

    def finish(self):
        """All objects done"""
        self.status_label.config(text="All trajectories complete!", foreground="green")
        messagebox.showinfo("Complete", "All trajectory drawings complete!\n\nSaving results...")
        self.root.quit()


def find_objects_needing_trajectory(config_path: str) -> List[Dict]:
    """Find all objects that need trajectory input"""
    config_path = Path(config_path)

    with open(config_path, 'r') as f:
        config_data = json.load(f)

    if isinstance(config_data, list):
        videos = config_data
    elif isinstance(config_data, dict) and "videos" in config_data:
        videos = config_data["videos"]
    else:
        raise ValueError("Invalid config format")

    objects_data = []

    for video_info in videos:
        output_dir = Path(video_info.get('output_dir', ''))
        vlm_analysis_path = output_dir / "vlm_analysis.json"

        if not vlm_analysis_path.exists():
            print(f"   Skipping {output_dir.parent.name}: no vlm_analysis.json")
            continue

        with open(vlm_analysis_path, 'r') as f:
            analysis = json.load(f)

        # Find objects with needs_trajectory=true
        objects_needing_traj = [
            obj for obj in analysis.get('affected_objects', [])
            if obj.get('needs_trajectory', False)
        ]

        if objects_needing_traj:
            objects_data.append({
                'video_info': video_info,
                'objects': objects_needing_traj,
                'output_dir': output_dir
            })

    return objects_data


def main():
    parser = argparse.ArgumentParser(description="Stage 3b: Trajectory Drawing GUI")
    parser.add_argument("--config", required=True, help="Config JSON")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Stage 3b: Trajectory Drawing GUI")
    print(f"{'='*70}\n")

    # Find objects needing trajectories
    print("Finding objects that need trajectory input...")
    objects_data = find_objects_needing_trajectory(args.config)

    if not objects_data:
        print("\n✅ No objects need trajectory input!")
        print("All objects are either stationary or visual artifacts.")
        print("Proceeding to Stage 3a for mask generation...")
        return

    total_objects = sum(len(d['objects']) for d in objects_data)
    print(f"\nFound {total_objects} object(s) needing trajectories across {len(objects_data)} video(s):")
    for d in objects_data:
        video_name = Path(d['video_info']['video_path']).parent.name
        print(f"  • {video_name}: {', '.join(obj['noun'] for obj in d['objects'])}")

    # Launch GUI
    print("\nLaunching trajectory drawing GUI...")
    print("Instructions:")
    print("  1. See the frame where the object is visible")
    print("  2. Click and drag to draw trajectory path (RED line)")
    print("  3. Draw from object's current position to where it should end up")
    print("  4. Click 'Save & Next' when done with each object")
    print("")

    root = tk.Tk()
    root.geometry("900x800")
    gui = TrajectoryGUI(root, objects_data)
    root.mainloop()

    # Save trajectories
    config_path = Path(args.config)
    output_path = config_path.parent / f"{config_path.stem}_trajectories.json"

    with open(output_path, 'w') as f:
        json.dump(gui.all_trajectories, f, indent=2)

    print(f"\n{'='*70}")
    print(f"✅ Stage 3b Complete!")
    print(f"{'='*70}")
    print(f"Saved trajectories to: {output_path}")
    print(f"Total trajectories: {len(gui.all_trajectories)}")
    print(f"\nNext: Run Stage 3a to generate grey masks (includes trajectories)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
