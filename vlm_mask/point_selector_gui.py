#!/usr/bin/env python3
"""
Point Selector GUI - Multi-Frame Support

NEW: Support adding points across multiple frames for complex cases
Example: Car appears at frame 0, hand carrying it appears at frame 30
         → Add points on car at frame 0, points on hand at frame 30
         → Both get segmented together as "primary object to remove"

Usage:
    python point_selector_gui_multiframe.py --config pexel_test_config.json
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


class PointSelectorGUI:
    def __init__(self, root, config_path=None):
        self.root = root
        self.root.title("Point Selector - Multi-Frame Support")

        # Data
        self.config_path = config_path
        self.config_data = None
        self.current_video_idx = 0
        self.current_frame_idx = 0
        self.video_captures = []
        self.total_frames_list = []

        # NEW: Points organized by frame
        self.points_by_frame = {}  # {frame_idx: [(x, y), ...]}
        self.all_points_by_frame = []  # List of dicts for all videos

        # Display
        self.display_scale = 1.0
        self.photo = None
        self.point_radius = 8

        self.setup_ui()

        if config_path:
            self.load_config_direct(config_path)

    def setup_ui(self):
        """Setup the GUI layout"""
        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Config", command=self.load_config)
        file_menu.add_command(label="Save Points", command=self.save_points)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Top toolbar
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Label(toolbar, text="Config:").pack(side=tk.LEFT)
        self.config_label = ttk.Label(toolbar, text="None", foreground="gray")
        self.config_label.pack(side=tk.LEFT, padx=5)

        ttk.Button(toolbar, text="Load Config", command=self.load_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Save All Points", command=self.save_points).pack(side=tk.LEFT, padx=5)

        # Video info
        info_frame = ttk.Frame(self.root)
        info_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.video_label = ttk.Label(info_frame, text="Video: None", font=("Arial", 10, "bold"))
        self.video_label.pack(side=tk.LEFT, padx=5)

        self.instruction_label = ttk.Label(info_frame, text="", foreground="blue")
        self.instruction_label.pack(side=tk.LEFT, padx=10)

        # Frame navigation controls - COMPACT (Ctrl+←/→ shortcuts)
        frame_nav = ttk.LabelFrame(self.root, text="Frame Navigation")
        frame_nav.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        btn_frame = ttk.Frame(frame_nav)
        btn_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        # Compact buttons
        ttk.Button(btn_frame, text="<<", command=self.first_frame, width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_frame, text="<10", command=lambda: self.prev_frame(10), width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_frame, text="<", command=lambda: self.prev_frame(1), width=3).pack(side=tk.LEFT, padx=1)

        self.frame_label = ttk.Label(btn_frame, text="F: 0/0", font=("Arial", 9))
        self.frame_label.pack(side=tk.LEFT, padx=8)

        ttk.Button(btn_frame, text=">", command=lambda: self.next_frame(1), width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_frame, text="10>", command=lambda: self.next_frame(10), width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_frame, text=">>", command=self.last_frame, width=3).pack(side=tk.LEFT, padx=1)

        # Slider inline
        self.frame_slider = ttk.Scale(btn_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_slider_change, length=250)
        self.frame_slider.pack(side=tk.LEFT, padx=5)

        # Frames with points inline
        ttk.Label(btn_frame, text="Points:", font=("Arial", 8)).pack(side=tk.LEFT, padx=3)
        self.frames_with_points_label = ttk.Label(btn_frame, text="None", foreground="blue", font=("Arial", 8))
        self.frames_with_points_label.pack(side=tk.LEFT)

        # Main canvas - SMALLER to fit everything
        canvas_frame = ttk.LabelFrame(self.root, text="Click to add points")
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=2)

        self.canvas = tk.Canvas(canvas_frame, width=800, height=450, bg='black', cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Bottom controls - COMPACT
        controls = ttk.Frame(self.root)
        controls.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=2)

        # Point info - compact
        point_info = ttk.Frame(controls)
        point_info.pack(side=tk.TOP, fill=tk.X, pady=2)

        self.point_count_label = ttk.Label(point_info, text="Pts: 0", font=("Arial", 9))
        self.point_count_label.pack(side=tk.LEFT, padx=5)

        ttk.Button(point_info, text="Clear Frame", command=self.clear_current_frame, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(point_info, text="Clear ALL", command=self.clear_all_frames, width=9).pack(side=tk.LEFT, padx=2)
        ttk.Button(point_info, text="Undo", command=self.undo_last_point, width=6).pack(side=tk.LEFT, padx=2)

        # Video navigation - compact
        nav_frame = ttk.Frame(controls)
        nav_frame.pack(side=tk.TOP, fill=tk.X, pady=2)

        ttk.Button(nav_frame, text="<< First", command=self.first_video, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="< Prev", command=self.prev_video, width=8).pack(side=tk.LEFT, padx=2)

        self.nav_label = ttk.Label(nav_frame, text="Video: 0/0", font=("Arial", 10, "bold"))
        self.nav_label.pack(side=tk.LEFT, padx=15)

        ttk.Button(nav_frame, text="Save & Next >", command=self.save_and_next, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="Last >>", command=self.last_video, width=8).pack(side=tk.LEFT, padx=2)

        # Status - compact
        self.status_label = ttk.Label(controls, text="Load config", foreground="blue", font=("Arial", 8))
        self.status_label.pack(side=tk.TOP, pady=2)

        # Keyboard shortcuts
        self.root.bind("<space>", lambda e: self.save_and_next())
        self.root.bind("<Left>", lambda e: self.prev_video())
        self.root.bind("<Right>", lambda e: self.save_and_next())
        self.root.bind("<Control-z>", lambda e: self.undo_last_point())
        self.root.bind("<Control-Left>", lambda e: self.prev_frame(1))
        self.root.bind("<Control-Right>", lambda e: self.next_frame(1))
        self.root.bind("<Control-Shift-Left>", lambda e: self.prev_frame(10))
        self.root.bind("<Control-Shift-Right>", lambda e: self.next_frame(10))

    def load_config_direct(self, config_path):
        """Load config from path (for command line usage)"""
        self.config_path = Path(config_path)

        try:
            with open(self.config_path, 'r') as f:
                self.config_data = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config: {e}")
            return

        self.process_config()

    def load_config(self):
        """Load JSON config file via dialog"""
        filepath = filedialog.askopenfilename(
            title="Select Config JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if not filepath:
            return

        self.config_path = Path(filepath)

        try:
            with open(self.config_path, 'r') as f:
                self.config_data = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config: {e}")
            return

        self.process_config()

    def process_config(self):
        """Process loaded config"""
        # Validate config
        if isinstance(self.config_data, list):
            videos = self.config_data
        elif isinstance(self.config_data, dict) and "videos" in self.config_data:
            videos = self.config_data["videos"]
        else:
            messagebox.showerror("Error", "Config must be a list or have 'videos' key")
            return

        if not isinstance(videos, list) or len(videos) == 0:
            messagebox.showerror("Error", "No videos in config")
            return

        self.videos = videos

        # Open video captures
        self.status_label.config(text="Opening video files...", foreground="blue")
        self.root.update()

        self.open_videos()

        # Initialize storage - now dict per video
        self.all_points_by_frame = [{} for _ in range(len(self.videos))]

        # Load existing points if available
        self.load_existing_points()

        # Update UI
        self.config_label.config(text=self.config_path.name, foreground="black")
        self.current_video_idx = 0
        self.current_frame_idx = 0
        self.display_current_video()

        self.status_label.config(
            text=f"Loaded {len(self.videos)} videos. Navigate frames and click points. Can add points on multiple frames!",
            foreground="green"
        )

    def open_videos(self):
        """Open all videos for frame navigation"""
        self.video_captures = []
        self.total_frames_list = []

        for i, video_info in enumerate(self.videos):
            video_path = video_info.get("video_path", "")

            if not video_path:
                self.video_captures.append(None)
                self.total_frames_list.append(0)
                continue

            video_path = Path(video_path)
            if not video_path.is_absolute():
                video_path = self.config_path.parent / video_path

            if not video_path.exists():
                messagebox.showwarning("Warning", f"Video not found: {video_path}")
                self.video_captures.append(None)
                self.total_frames_list.append(0)
                continue

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                messagebox.showwarning("Warning", f"Failed to open video: {video_path}")
                self.video_captures.append(None)
                self.total_frames_list.append(0)
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_captures.append(cap)
            self.total_frames_list.append(total_frames)

            self.status_label.config(text=f"Opened video {i+1}/{len(self.videos)}", foreground="blue")
            self.root.update()

    def load_existing_points(self):
        """Load existing points from output file if it exists"""
        output_path = self.config_path.parent / f"{self.config_path.stem}_points.json"

        if not output_path.exists():
            return

        try:
            with open(output_path, 'r') as f:
                existing_data = json.load(f)

            if isinstance(existing_data, list):
                existing_videos = existing_data
            elif isinstance(existing_data, dict) and "videos" in existing_data:
                existing_videos = existing_data["videos"]
            else:
                return

            for i, video_data in enumerate(existing_videos):
                if i < len(self.all_points_by_frame):
                    # Load multi-frame format
                    points_by_frame = video_data.get("primary_points_by_frame", {})
                    # Convert string keys to int
                    self.all_points_by_frame[i] = {int(k): v for k, v in points_by_frame.items()}

            self.status_label.config(text="Loaded existing points", foreground="green")
        except Exception as e:
            print(f"Warning: Could not load existing points: {e}")

    def get_current_frame(self):
        """Get frame at current_frame_idx from current video"""
        if self.current_video_idx >= len(self.video_captures):
            return None

        cap = self.video_captures[self.current_video_idx]
        if cap is None:
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = cap.read()

        if not ret:
            return None

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def display_current_video(self):
        """Display current video frame"""
        if not self.video_captures:
            return

        video_info = self.videos[self.current_video_idx]
        video_path = video_info.get("video_path", "")

        # Update labels
        self.video_label.config(text=f"Video: {Path(video_path).name}")
        instruction = video_info.get("instruction", "")
        if instruction:
            self.instruction_label.config(text=f"Instruction: {instruction}")

        self.nav_label.config(text=f"Video: {self.current_video_idx + 1}/{len(self.videos)}")

        # Load points for this video
        self.points_by_frame = self.all_points_by_frame[self.current_video_idx].copy()

        # Update frame controls
        total_frames = self.total_frames_list[self.current_video_idx]
        self.frame_slider.config(to=max(1, total_frames - 1))
        self.frame_slider.set(self.current_frame_idx)
        self.frame_label.config(text=f"F: {self.current_frame_idx}/{total_frames - 1}")

        # Update frames with points display
        self.update_frames_display()

        self.display_frame()

    def update_frames_display(self):
        """Update display showing which frames have points"""
        if not self.points_by_frame:
            self.frames_with_points_label.config(text="None", foreground="gray")
        else:
            frames = sorted(self.points_by_frame.keys())
            frames_str = ", ".join(f"F{f}" for f in frames)
            total_points = sum(len(pts) for pts in self.points_by_frame.values())
            self.frames_with_points_label.config(
                text=f"{frames_str} ({total_points} total points)",
                foreground="green"
            )

    def display_frame(self):
        """Display current frame with points"""
        frame = self.get_current_frame()
        if frame is None:
            return

        # Draw points for CURRENT frame
        vis = frame.copy()
        current_points = self.points_by_frame.get(self.current_frame_idx, [])

        for i, (x, y) in enumerate(current_points):
            cv2.circle(vis, (x, y), self.point_radius, (255, 0, 0), -1)
            cv2.circle(vis, (x, y), self.point_radius + 2, (255, 255, 255), 2)
            cv2.putText(vis, str(i + 1), (x + 12, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show indicator if other frames have points
        if len(self.points_by_frame) > 0:
            other_frames = [f for f in self.points_by_frame.keys() if f != self.current_frame_idx]
            if other_frames:
                text = f"Other frames with points: {', '.join(map(str, sorted(other_frames)))}"
                cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Scale for display
        h, w = vis.shape[:2]
        max_width, max_height = 800, 450
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

        self.point_count_label.config(text=f"Pts on F{self.current_frame_idx}: {len(current_points)}")

    def on_canvas_click(self, event):
        """Handle click on canvas - add point to CURRENT frame"""
        # Convert to frame coordinates
        x = int(event.x / self.display_scale)
        y = int(event.y / self.display_scale)

        # Add to current frame
        if self.current_frame_idx not in self.points_by_frame:
            self.points_by_frame[self.current_frame_idx] = []

        self.points_by_frame[self.current_frame_idx].append((x, y))
        self.update_frames_display()
        self.display_frame()

    def clear_current_frame(self):
        """Clear points for current frame only"""
        if self.current_frame_idx in self.points_by_frame:
            del self.points_by_frame[self.current_frame_idx]
            self.update_frames_display()
            self.display_frame()

    def clear_all_frames(self):
        """Clear all points for current video"""
        result = messagebox.askyesno("Clear All", "Clear points from ALL frames?")
        if result:
            self.points_by_frame = {}
            self.update_frames_display()
            self.display_frame()

    def undo_last_point(self):
        """Remove last point from current frame"""
        if self.current_frame_idx in self.points_by_frame and self.points_by_frame[self.current_frame_idx]:
            self.points_by_frame[self.current_frame_idx].pop()
            if not self.points_by_frame[self.current_frame_idx]:
                del self.points_by_frame[self.current_frame_idx]
            self.update_frames_display()
            self.display_frame()

    # Frame navigation methods
    def first_frame(self):
        """Jump to first frame"""
        self.current_frame_idx = 0
        self.frame_slider.set(self.current_frame_idx)
        self.update_frame_display()

    def last_frame(self):
        """Jump to last frame"""
        total_frames = self.total_frames_list[self.current_video_idx]
        self.current_frame_idx = max(0, total_frames - 1)
        self.frame_slider.set(self.current_frame_idx)
        self.update_frame_display()

    def prev_frame(self, step=1):
        """Go to previous frame"""
        self.current_frame_idx = max(0, self.current_frame_idx - step)
        self.frame_slider.set(self.current_frame_idx)
        self.update_frame_display()

    def next_frame(self, step=1):
        """Go to next frame"""
        total_frames = self.total_frames_list[self.current_video_idx]
        self.current_frame_idx = min(total_frames - 1, self.current_frame_idx + step)
        self.frame_slider.set(self.current_frame_idx)
        self.update_frame_display()

    def on_slider_change(self, value):
        """Handle slider change"""
        self.current_frame_idx = int(float(value))
        self.update_frame_display()

    def update_frame_display(self):
        """Update frame label and display"""
        total_frames = self.total_frames_list[self.current_video_idx]
        self.frame_label.config(text=f"F: {self.current_frame_idx}/{total_frames - 1}")
        self.display_frame()

    # Video navigation
    def first_video(self):
        """Jump to first video"""
        self.save_current_points()
        self.current_video_idx = 0
        self.current_frame_idx = 0
        self.display_current_video()

    def last_video(self):
        """Jump to last video"""
        self.save_current_points()
        self.current_video_idx = len(self.videos) - 1
        self.current_frame_idx = 0
        self.display_current_video()

    def prev_video(self):
        """Go to previous video"""
        if self.current_video_idx > 0:
            self.save_current_points()
            self.current_video_idx -= 1
            self.current_frame_idx = 0
            self.display_current_video()

    def save_and_next(self):
        """Save current points and move to next video"""
        if len(self.points_by_frame) == 0:
            result = messagebox.askyesno("No Points", "No points selected for any frame. Continue to next video?")
            if not result:
                return

        self.save_current_points()

        if self.current_video_idx < len(self.videos) - 1:
            self.current_video_idx += 1
            self.current_frame_idx = 0
            self.display_current_video()
        else:
            messagebox.showinfo("Complete", "All videos processed!")

    def save_current_points(self):
        """Save current video's points to storage"""
        self.all_points_by_frame[self.current_video_idx] = self.points_by_frame.copy()

    def save_points(self):
        """Save all points to JSON file"""
        if not self.config_path:
            messagebox.showerror("Error", "No config loaded")
            return

        # Save current video first
        self.save_current_points()

        # Build output
        output_videos = []
        for i, video_info in enumerate(self.videos):
            video_data = video_info.copy()

            points_by_frame = self.all_points_by_frame[i]

            # Convert to serializable format (int keys → string keys for JSON)
            video_data["primary_points_by_frame"] = {
                str(frame_idx): points for frame_idx, points in points_by_frame.items()
            }

            # Also save list of frames for easy access
            video_data["primary_frames"] = sorted(points_by_frame.keys())

            # Backwards compatibility: if only one frame, save as before
            if len(points_by_frame) == 1:
                frame_idx = list(points_by_frame.keys())[0]
                video_data["first_appears_frame"] = frame_idx
                video_data["primary_points"] = points_by_frame[frame_idx]
            elif len(points_by_frame) > 1:
                # Multiple frames - use first frame as "first_appears_frame"
                video_data["first_appears_frame"] = min(points_by_frame.keys())
                # Flatten all points for backwards compat (not ideal but helps)
                all_points = []
                for frame_idx in sorted(points_by_frame.keys()):
                    all_points.extend(points_by_frame[frame_idx])
                video_data["primary_points"] = all_points

            output_videos.append(video_data)

        # Match input format
        if isinstance(self.config_data, list):
            output_data = output_videos
        else:
            output_data = {"videos": output_videos}

        # Save
        output_path = self.config_path.parent / f"{self.config_path.stem}_points.json"

        try:
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)

            self.status_label.config(text=f"Saved to {output_path.name}", foreground="green")
            messagebox.showinfo("Success", f"Points saved to:\n{output_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")

    def __del__(self):
        """Clean up video captures"""
        for cap in self.video_captures:
            if cap is not None:
                cap.release()


def main():
    parser = argparse.ArgumentParser(description="Point Selector GUI - Multi-Frame Support")
    parser.add_argument("--config", help="Config JSON file to load")
    args = parser.parse_args()

    root = tk.Tk()
    root.geometry("900x750")  # Compact height to fit on screen
    gui = PointSelectorGUI(root, config_path=args.config)
    root.mainloop()


if __name__ == "__main__":
    main()
