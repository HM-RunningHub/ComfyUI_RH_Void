#!/usr/bin/env python3
"""
Mask Editor GUI - Edit gridified video masks with grid toggling and brush tools
"""
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import subprocess
from pathlib import Path
import copy
import time

class MaskEditorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Mask Editor")

        # Video data
        self.rgb_frames = []
        self.mask_frames = []
        self.current_frame = 0
        self.grid_rows = 0
        self.grid_cols = 0
        self.min_grid = 8

        # Edit state
        self.undo_stack = []
        self.redo_stack = []
        self.current_tool = "grid"  # "grid" or "brush"
        self.brush_size = 20
        self.brush_mode = "add"  # "add" or "erase"

        # Display state
        self.display_scale = 1.0
        self.rgb_photo = None
        self.mask_photo = None
        self.dragging = False
        self.last_brush_pos = None
        self.last_update_time = 0
        self.update_interval = 0.2  # Update every 200ms during dragging (5 FPS - less choppy)
        self.cached_rgb_frame = None  # Cache current RGB frame
        self.cached_frame_idx = -1  # Track which frame is cached
        self.pending_update = False  # Track if update is needed after drag
        self.brush_repeat_id = None  # Timer for continuous brush application

        # Paths
        self.folder_path = None
        self.mask_path = None
        self.rgb_path = None

        self.setup_ui()

    def setup_ui(self):
        """Setup the GUI layout"""
        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Folder", command=self.load_folder)
        file_menu.add_command(label="Save Mask", command=self.save_mask)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self.redo, accelerator="Ctrl+Y")

        # Keyboard shortcuts
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-y>", lambda e: self.redo())
        self.root.bind("<Left>", lambda e: self.prev_frame())
        self.root.bind("<Right>", lambda e: self.next_frame())

        # Top toolbar
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Label(toolbar, text="Folder:").pack(side=tk.LEFT)
        self.folder_label = ttk.Label(toolbar, text="None", foreground="gray")
        self.folder_label.pack(side=tk.LEFT, padx=5)

        ttk.Button(toolbar, text="Open Folder", command=self.load_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Save Mask", command=self.save_mask).pack(side=tk.LEFT, padx=5)

        # Main content area
        content = ttk.Frame(self.root)
        content.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel - Original video
        left_panel = ttk.LabelFrame(content, text="Original Video")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.rgb_canvas = tk.Canvas(left_panel, width=640, height=480, bg='black')
        self.rgb_canvas.pack(fill=tk.BOTH, expand=True)

        # Right panel - Mask
        right_panel = ttk.LabelFrame(content, text="Mask (Editable)")
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.mask_canvas = tk.Canvas(right_panel, width=640, height=480, bg='black')
        self.mask_canvas.pack(fill=tk.BOTH, expand=True)
        self.mask_canvas.bind("<Button-1>", self.on_mask_click)
        self.mask_canvas.bind("<B1-Motion>", self.on_mask_drag)
        self.mask_canvas.bind("<ButtonRelease-1>", self.on_mask_release)

        # Bottom controls
        controls = ttk.Frame(self.root)
        controls.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        # Frame navigation
        nav_frame = ttk.LabelFrame(controls, text="Frame Navigation")
        nav_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        ttk.Button(nav_frame, text="<<", command=self.first_frame, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="<", command=self.prev_frame, width=5).pack(side=tk.LEFT, padx=2)

        self.frame_label = ttk.Label(nav_frame, text="Frame: 0 / 0")
        self.frame_label.pack(side=tk.LEFT, padx=10)

        ttk.Button(nav_frame, text=">", command=self.next_frame, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text=">>", command=self.last_frame, width=5).pack(side=tk.LEFT, padx=2)

        self.frame_slider = ttk.Scale(nav_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                      command=self.on_slider_change)
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        # Tool selection
        tool_frame = ttk.LabelFrame(controls, text="Tools")
        tool_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        self.tool_var = tk.StringVar(value="grid")
        ttk.Radiobutton(tool_frame, text="Grid Toggle", variable=self.tool_var,
                       value="grid", command=self.on_tool_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(tool_frame, text="Grid Black Toggle", variable=self.tool_var,
                       value="grid_black", command=self.on_tool_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(tool_frame, text="Brush (Add Black)", variable=self.tool_var,
                       value="brush_add", command=self.on_tool_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(tool_frame, text="Brush (Erase Black)", variable=self.tool_var,
                       value="brush_erase", command=self.on_tool_change).pack(side=tk.LEFT, padx=5)

        ttk.Label(tool_frame, text="Brush Size:").pack(side=tk.LEFT, padx=10)
        self.brush_slider = ttk.Scale(tool_frame, from_=5, to=100, orient=tk.HORIZONTAL,
                                     command=self.on_brush_size_change)
        self.brush_slider.set(20)
        self.brush_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.brush_size_label = ttk.Label(tool_frame, text="20")
        self.brush_size_label.pack(side=tk.LEFT, padx=5)

        # Copy from previous frame
        copy_frame = ttk.LabelFrame(controls, text="Copy from Previous Frame")
        copy_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        ttk.Button(copy_frame, text="Copy Black Mask",
                  command=self.copy_black_from_previous).pack(side=tk.LEFT, padx=5)
        ttk.Button(copy_frame, text="Copy Grey Mask",
                  command=self.copy_grey_from_previous).pack(side=tk.LEFT, padx=5)

        # Info panel
        info_frame = ttk.Frame(controls)
        info_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        self.info_label = ttk.Label(info_frame, text="Load a folder to begin", foreground="blue")
        self.info_label.pack(side=tk.LEFT, padx=5)

        self.grid_info_label = ttk.Label(info_frame, text="Grid: N/A")
        self.grid_info_label.pack(side=tk.RIGHT, padx=5)

    def calculate_square_grid(self, width, height, min_grid=8):
        """Calculate grid dimensions to make square cells"""
        aspect_ratio = width / height

        if width >= height:
            grid_rows = min_grid
            grid_cols = max(min_grid, round(min_grid * aspect_ratio))
        else:
            grid_cols = min_grid
            grid_rows = max(min_grid, round(min_grid / aspect_ratio))

        return grid_rows, grid_cols

    def load_folder(self):
        """Load a folder containing rgb_full.mp4/input_video.mp4 and quadmask_0.mp4"""
        folder = filedialog.askdirectory(title="Select Folder")
        if not folder:
            return

        folder_path = Path(folder)

        # Find RGB video
        rgb_path = None
        for name in ["rgb_full.mp4", "input_video.mp4"]:
            candidate = folder_path / name
            if candidate.exists():
                rgb_path = candidate
                break

        mask_path = folder_path / "quadmask_0.mp4"

        if not rgb_path or not mask_path.exists():
            messagebox.showerror("Error", "Folder must contain quadmask_0.mp4 and rgb_full.mp4 or input_video.mp4")
            return

        self.folder_path = folder_path
        self.rgb_path = rgb_path
        self.mask_path = mask_path

        # Load videos
        self.load_videos()

    def load_videos(self):
        """Load RGB and mask videos into memory"""
        self.info_label.config(text="Loading videos...")
        self.root.update()

        # Load RGB frames
        self.rgb_frames = self.read_video_frames(self.rgb_path)

        # Load mask frames
        self.mask_frames = self.read_video_frames(self.mask_path)

        if len(self.rgb_frames) != len(self.mask_frames):
            messagebox.showwarning("Warning",
                f"Frame count mismatch: RGB={len(self.rgb_frames)}, Mask={len(self.mask_frames)}")

        if len(self.mask_frames) == 0:
            messagebox.showerror("Error", "No frames loaded")
            return

        # Calculate grid dimensions
        height, width = self.mask_frames[0].shape[:2]
        self.grid_rows, self.grid_cols = self.calculate_square_grid(width, height, self.min_grid)

        # Calculate display scale
        max_width = 600
        max_height = 450
        scale_w = max_width / width
        scale_h = max_height / height
        self.display_scale = min(scale_w, scale_h, 1.0)

        # Update UI
        self.folder_label.config(text=self.folder_path.name, foreground="black")
        self.grid_info_label.config(text=f"Grid: {self.grid_rows}x{self.grid_cols}")
        self.frame_slider.config(to=len(self.mask_frames)-1)
        self.current_frame = 0
        self.undo_stack = []
        self.redo_stack = []

        self.update_display()
        self.info_label.config(text=f"Loaded {len(self.mask_frames)} frames", foreground="green")

    def read_video_frames(self, video_path):
        """Read all frames from a video"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
        cap.release()
        return frames

    def write_video_frames(self, frames, output_path, fps=12):
        """Write frames to a video file using lossless H.264"""
        if not frames:
            return

        height, width = frames[0].shape[:2]

        # Write temp AVI first
        temp_avi = output_path.with_suffix('.avi')
        fourcc = cv2.VideoWriter_fourcc(*'FFV1')
        out = cv2.VideoWriter(str(temp_avi), fourcc, fps, (width, height), isColor=False)

        for frame in frames:
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            out.write(frame)

        out.release()

        # Convert to LOSSLESS H.264 (qp=0)
        cmd = [
            'ffmpeg', '-y', '-i', str(temp_avi),
            '-c:v', 'libx264', '-qp', '0', '-preset', 'ultrafast',
            '-pix_fmt', 'yuv444p', '-r', '12',
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True)
        temp_avi.unlink()

    def update_display(self, fast_mode=False):
        """Update both canvas displays (or just mask in fast mode)"""
        if not self.mask_frames:
            return

        # Cache RGB frame if needed (only in full mode)
        if not fast_mode and self.cached_frame_idx != self.current_frame:
            if self.current_frame < len(self.rgb_frames):
                rgb_frame = self.rgb_frames[self.current_frame]
                self.cached_rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_GRAY2RGB) if len(rgb_frame.shape) == 2 else rgb_frame.copy()
                self.cached_frame_idx = self.current_frame
            else:
                self.cached_rgb_frame = None

        if not fast_mode:
            # Update frame label
            self.frame_label.config(text=f"Frame: {self.current_frame + 1} / {len(self.mask_frames)}")
            self.frame_slider.set(self.current_frame)

            # Display RGB frame
            if self.cached_rgb_frame is not None:
                rgb_display = cv2.resize(self.cached_rgb_frame, None, fx=self.display_scale, fy=self.display_scale)
                rgb_image = Image.fromarray(rgb_display)
                self.rgb_photo = ImageTk.PhotoImage(rgb_image)
                self.rgb_canvas.delete("all")
                self.rgb_canvas.create_image(0, 0, anchor=tk.NW, image=self.rgb_photo)

        # Display mask frame with grid overlay
        mask_frame = self.mask_frames[self.current_frame]
        # Use simple mode (no RGB blending) during fast updates for speed
        mask_display = self.create_mask_visualization(mask_frame, self.cached_rgb_frame, simple_mode=fast_mode)
        mask_display = cv2.resize(mask_display, None, fx=self.display_scale, fy=self.display_scale)
        mask_image = Image.fromarray(mask_display)
        self.mask_photo = ImageTk.PhotoImage(mask_image)
        self.mask_canvas.delete("all")
        self.mask_canvas.create_image(0, 0, anchor=tk.NW, image=self.mask_photo)

        # Draw grid overlay (skip in fast mode for performance)
        if not fast_mode:
            self.draw_grid_overlay()

    def create_mask_visualization(self, mask_frame, rgb_frame=None, simple_mode=False):
        """Create RGB visualization of mask with color coding and RGB background"""
        height, width = mask_frame.shape
        vis = np.zeros((height, width, 3), dtype=np.uint8)

        if simple_mode:
            # Fast simple mode - no blending, just solid colors
            vis[mask_frame == 255] = [150, 150, 150]  # Background - gray
            vis[mask_frame == 127] = [0, 200, 0]      # Gridified - green
            vis[mask_frame == 63] = [200, 200, 0]     # Overlap - yellow
            vis[mask_frame == 0] = [200, 0, 0]        # Black - red
            return vis

        # If RGB frame is provided, use it as background
        if rgb_frame is not None:
            if len(rgb_frame.shape) == 2:
                # Convert grayscale to RGB
                rgb_background = cv2.cvtColor(rgb_frame, cv2.COLOR_GRAY2RGB)
            else:
                rgb_background = rgb_frame.copy()

            # Use it at 50% opacity as base
            vis = (rgb_background * 0.5).astype(np.uint8)

        # Color coding with transparency to show background:
        # 0 (black) -> Red tint (to indicate removal area)
        # 63 (overlap) -> Yellow tint
        # 127 (gridified) -> Green tint
        # 255 (background) -> Keep RGB background visible

        # Background areas - show RGB at 60% brightness
        bg_mask = mask_frame == 255
        if rgb_frame is not None:
            vis[bg_mask] = (rgb_background[bg_mask] * 0.6).astype(np.uint8)
        else:
            vis[bg_mask] = [150, 150, 150]

        # Green overlay for gridified areas - blend 40% background + 60% green tint
        green_mask = mask_frame == 127
        if rgb_frame is not None:
            vis[green_mask] = np.clip(rgb_background[green_mask] * 0.4 + np.array([0, 180, 0]) * 0.6, 0, 255).astype(np.uint8)
        else:
            vis[green_mask] = [0, 200, 0]

        # Yellow overlay for overlap areas - blend 40% background + 60% yellow tint
        yellow_mask = mask_frame == 63
        if rgb_frame is not None:
            vis[yellow_mask] = np.clip(rgb_background[yellow_mask] * 0.4 + np.array([180, 180, 0]) * 0.6, 0, 255).astype(np.uint8)
        else:
            vis[yellow_mask] = [200, 200, 0]

        # Red tint for black areas (removal) - blend 30% background + 70% red tint
        black_mask = mask_frame == 0
        if rgb_frame is not None:
            vis[black_mask] = np.clip(rgb_background[black_mask] * 0.3 + np.array([200, 0, 0]) * 0.7, 0, 255).astype(np.uint8)
        else:
            vis[black_mask] = [200, 0, 0]

        return vis

    def draw_grid_overlay(self):
        """Draw grid lines on mask canvas"""
        if not self.mask_frames:
            return

        height, width = self.mask_frames[0].shape

        scaled_width = int(width * self.display_scale)
        scaled_height = int(height * self.display_scale)

        cell_width = scaled_width / self.grid_cols
        cell_height = scaled_height / self.grid_rows

        # Draw vertical lines
        for col in range(self.grid_cols + 1):
            x = int(col * cell_width)
            self.mask_canvas.create_line(x, 0, x, scaled_height, fill='red', width=1, tags='grid')

        # Draw horizontal lines
        for row in range(self.grid_rows + 1):
            y = int(row * cell_height)
            self.mask_canvas.create_line(0, y, scaled_width, y, fill='red', width=1, tags='grid')

    def get_grid_from_pos(self, x, y):
        """Get grid row, col from canvas position"""
        if not self.mask_frames:
            return None, None

        height, width = self.mask_frames[0].shape

        # Convert to frame coordinates
        frame_x = int(x / self.display_scale)
        frame_y = int(y / self.display_scale)

        if frame_x < 0 or frame_x >= width or frame_y < 0 or frame_y >= height:
            return None, None

        cell_width = width / self.grid_cols
        cell_height = height / self.grid_rows

        col = int(frame_x / cell_width)
        row = int(frame_y / cell_height)

        return row, col

    def toggle_grid(self, row, col):
        """Toggle a grid cell between 127 and 255, handling 63 overlaps"""
        if row is None or col is None:
            return

        if row < 0 or row >= self.grid_rows or col < 0 or col >= self.grid_cols:
            return

        # Save state for undo
        self.save_state()

        mask = self.mask_frames[self.current_frame]
        height, width = mask.shape

        cell_width = width / self.grid_cols
        cell_height = height / self.grid_rows

        y1 = int(row * cell_height)
        y2 = int((row + 1) * cell_height)
        x1 = int(col * cell_width)
        x2 = int((col + 1) * cell_width)

        grid_region = mask[y1:y2, x1:x2]

        # Check if grid has any 127 or 63 values
        has_active = np.any((grid_region == 127) | (grid_region == 63))

        if has_active:
            # Turn OFF: 127->255, 63->0, keep 0 and 255 as is
            mask[y1:y2, x1:x2] = np.where(grid_region == 127, 255,
                                         np.where(grid_region == 63, 0, grid_region))
        else:
            # Turn ON: 255->127, 0->63, keep others as is
            mask[y1:y2, x1:x2] = np.where(grid_region == 255, 127,
                                         np.where(grid_region == 0, 63, grid_region))

        self.update_display()

    def toggle_grid_black(self, row, col):
        """Toggle black mask in a grid cell"""
        if row is None or col is None:
            return

        if row < 0 or row >= self.grid_rows or col < 0 or col >= self.grid_cols:
            return

        # Save state for undo
        self.save_state()

        mask = self.mask_frames[self.current_frame]
        height, width = mask.shape

        cell_width = width / self.grid_cols
        cell_height = height / self.grid_rows

        y1 = int(row * cell_height)
        y2 = int((row + 1) * cell_height)
        x1 = int(col * cell_width)
        x2 = int((col + 1) * cell_width)

        grid_region = mask[y1:y2, x1:x2]

        # Check if grid has any black (0 or 63 values)
        has_black = np.any((grid_region == 0) | (grid_region == 63))

        if has_black:
            # Turn OFF black: 0->255, 63->127, keep 127 and 255 as is
            mask[y1:y2, x1:x2] = np.where(grid_region == 0, 255,
                                         np.where(grid_region == 63, 127, grid_region))
        else:
            # Turn ON black: 255->0, 127->63, keep others as is
            mask[y1:y2, x1:x2] = np.where(grid_region == 255, 0,
                                         np.where(grid_region == 127, 63, grid_region))

        self.update_display()

    def apply_brush(self, x, y, mode="add"):
        """Apply brush to add/erase black mask (vectorized for speed)"""
        if not self.mask_frames:
            return

        mask = self.mask_frames[self.current_frame]
        height, width = mask.shape

        # Convert to frame coordinates
        frame_x = int(x / self.display_scale)
        frame_y = int(y / self.display_scale)

        if frame_x < 0 or frame_x >= width or frame_y < 0 or frame_y >= height:
            return

        # Create circular brush using vectorized operations
        radius = int(self.brush_size / 2)

        y1 = max(0, frame_y - radius)
        y2 = min(height, frame_y + radius + 1)
        x1 = max(0, frame_x - radius)
        x2 = min(width, frame_x + radius + 1)

        # Get the region
        region = mask[y1:y2, x1:x2]

        # Create coordinate grids for distance calculation
        yy, xx = np.ogrid[y1:y2, x1:x2]
        dist = np.sqrt((xx - frame_x)**2 + (yy - frame_y)**2)
        brush_mask = dist <= radius

        if mode == "add":
            # Add black: 255->0, 127->63
            region[brush_mask & (region == 255)] = 0
            region[brush_mask & (region == 127)] = 63
        else:  # erase
            # Erase black: 0->255, 63->127
            region[brush_mask & (region == 0)] = 255
            region[brush_mask & (region == 63)] = 127

    def on_mask_click(self, event):
        """Handle click on mask canvas"""
        if not self.mask_frames:
            return

        tool = self.tool_var.get()

        if tool == "grid":
            row, col = self.get_grid_from_pos(event.x, event.y)
            self.toggle_grid(row, col)
        elif tool == "grid_black":
            row, col = self.get_grid_from_pos(event.x, event.y)
            self.toggle_grid_black(row, col)
        elif tool in ["brush_add", "brush_erase"]:
            self.save_state()
            mode = "add" if tool == "brush_add" else "erase"
            self.apply_brush(event.x, event.y, mode)
            self.dragging = True
            self.last_brush_pos = (event.x, event.y)
            self.last_update_time = time.time()
            self.update_display(fast_mode=True)
            # Start continuous brush application
            self.schedule_brush_repeat()

    def on_mask_drag(self, event):
        """Handle drag on mask canvas with throttled updates"""
        if not self.dragging:
            return

        tool = self.tool_var.get()
        if tool in ["brush_add", "brush_erase"]:
            # Update brush position when moving
            self.last_brush_pos = (event.x, event.y)
            mode = "add" if tool == "brush_add" else "erase"
            self.apply_brush(event.x, event.y, mode)

            # Only update display if enough time has passed (fast mode - no grid)
            current_time = time.time()
            if current_time - self.last_update_time >= self.update_interval:
                self.update_display(fast_mode=True)
                self.last_update_time = current_time

    def on_mask_release(self, event):
        """Handle release on mask canvas"""
        self.dragging = False
        self.last_brush_pos = None
        # Cancel continuous brush application
        if self.brush_repeat_id:
            self.root.after_cancel(self.brush_repeat_id)
            self.brush_repeat_id = None
        # Final full update when releasing to show the complete result with blending
        self.update_display(fast_mode=False)

    def schedule_brush_repeat(self):
        """Schedule continuous brush application while mouse is held down"""
        if self.dragging and self.last_brush_pos:
            tool = self.tool_var.get()
            if tool in ["brush_add", "brush_erase"]:
                mode = "add" if tool == "brush_add" else "erase"
                x, y = self.last_brush_pos
                self.apply_brush(x, y, mode)

                # Update display if enough time has passed
                current_time = time.time()
                if current_time - self.last_update_time >= self.update_interval:
                    self.update_display(fast_mode=True)
                    self.last_update_time = current_time

                # Schedule next application (every 30ms for smooth continuous painting)
                self.brush_repeat_id = self.root.after(30, self.schedule_brush_repeat)

    def copy_black_from_previous(self):
        """Copy ONLY black component from previous frame, preserving grey in current frame"""
        if not self.mask_frames:
            messagebox.showwarning("Warning", "No mask loaded")
            return

        if self.current_frame == 0:
            messagebox.showwarning("Warning", "Cannot copy from previous frame - already at first frame")
            return

        # Save state for undo
        self.save_state()

        prev_mask = self.mask_frames[self.current_frame - 1]
        curr_mask = self.mask_frames[self.current_frame]

        # Copy ONLY the black component from previous frame
        # Where prev has black (0 or 63): add black to curr
        # Where prev doesn't have black (127 or 255): remove black from curr

        has_black_in_prev = (prev_mask == 0) | (prev_mask == 63)
        no_black_in_prev = (prev_mask == 127) | (prev_mask == 255)

        # Remove black where prev doesn't have it (preserve grey)
        curr_mask[no_black_in_prev & (curr_mask == 0)] = 255   # 0 → 255
        curr_mask[no_black_in_prev & (curr_mask == 63)] = 127  # 63 → 127 (keep grey)

        # Add black where prev has it (preserve grey)
        curr_mask[has_black_in_prev & (curr_mask == 255)] = 0   # 255 → 0
        curr_mask[has_black_in_prev & (curr_mask == 127)] = 63  # 127 → 63 (keep grey, add black)

        self.update_display()
        self.info_label.config(text="Copied black mask from previous frame", foreground="green")

    def copy_grey_from_previous(self):
        """Copy ONLY grey component from previous frame, preserving black in current frame"""
        if not self.mask_frames:
            messagebox.showwarning("Warning", "No mask loaded")
            return

        if self.current_frame == 0:
            messagebox.showwarning("Warning", "Cannot copy from previous frame - already at first frame")
            return

        # Save state for undo
        self.save_state()

        prev_mask = self.mask_frames[self.current_frame - 1]
        curr_mask = self.mask_frames[self.current_frame]

        # Copy ONLY the grey component from previous frame
        # Where prev has grey (127 or 63): add grey to curr
        # Where prev doesn't have grey (0 or 255): remove grey from curr

        has_grey_in_prev = (prev_mask == 127) | (prev_mask == 63)
        no_grey_in_prev = (prev_mask == 0) | (prev_mask == 255)

        # Remove grey where prev doesn't have it (preserve black)
        curr_mask[no_grey_in_prev & (curr_mask == 127)] = 255  # 127 → 255
        curr_mask[no_grey_in_prev & (curr_mask == 63)] = 0     # 63 → 0 (keep black)

        # Add grey where prev has it (preserve black)
        curr_mask[has_grey_in_prev & (curr_mask == 255)] = 127  # 255 → 127
        curr_mask[has_grey_in_prev & (curr_mask == 0)] = 63     # 0 → 63 (keep black, add grey)

        self.update_display()
        self.info_label.config(text="Copied grey mask from previous frame", foreground="green")

    def save_state(self):
        """Save current state for undo"""
        if not self.mask_frames:
            return

        # Save deep copy of current frame
        state = {
            'frame': self.current_frame,
            'mask': self.mask_frames[self.current_frame].copy()
        }
        self.undo_stack.append(state)
        self.redo_stack.clear()

        # Limit undo stack size
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def undo(self):
        """Undo last edit"""
        if not self.undo_stack:
            return

        # Save current state to redo
        redo_state = {
            'frame': self.current_frame,
            'mask': self.mask_frames[self.current_frame].copy()
        }
        self.redo_stack.append(redo_state)

        # Restore previous state
        state = self.undo_stack.pop()
        self.current_frame = state['frame']
        self.mask_frames[self.current_frame] = state['mask']

        self.update_display()

    def redo(self):
        """Redo last undone edit"""
        if not self.redo_stack:
            return

        # Save current state to undo
        undo_state = {
            'frame': self.current_frame,
            'mask': self.mask_frames[self.current_frame].copy()
        }
        self.undo_stack.append(undo_state)

        # Restore redo state
        state = self.redo_stack.pop()
        self.current_frame = state['frame']
        self.mask_frames[self.current_frame] = state['mask']

        self.update_display()

    def save_mask(self):
        """Save edited mask back to quadmask_0.mp4"""
        if not self.mask_frames or not self.mask_path:
            messagebox.showwarning("Warning", "No mask loaded")
            return

        # Confirm save
        result = messagebox.askyesno("Confirm Save",
            f"Save mask to {self.mask_path.name}?\nThis will overwrite the existing file.")
        if not result:
            return

        self.info_label.config(text="Saving mask...", foreground="blue")
        self.root.update()

        # Write video
        self.write_video_frames(self.mask_frames, self.mask_path)

        self.info_label.config(text="Mask saved successfully!", foreground="green")
        messagebox.showinfo("Success", f"Mask saved to {self.mask_path.name}!")

    def first_frame(self):
        """Go to first frame"""
        self.current_frame = 0
        self.update_display()

    def last_frame(self):
        """Go to last frame"""
        if self.mask_frames:
            self.current_frame = len(self.mask_frames) - 1
            self.update_display()

    def prev_frame(self):
        """Go to previous frame"""
        if self.current_frame > 0:
            self.current_frame -= 1
            self.update_display()

    def next_frame(self):
        """Go to next frame"""
        if self.mask_frames and self.current_frame < len(self.mask_frames) - 1:
            self.current_frame += 1
            self.update_display()

    def on_slider_change(self, value):
        """Handle slider change"""
        if not self.mask_frames:
            return

        new_frame = int(float(value))
        if new_frame != self.current_frame:
            self.current_frame = new_frame
            self.update_display()

    def on_tool_change(self):
        """Handle tool selection change"""
        tool = self.tool_var.get()
        if tool == "grid":
            self.info_label.config(text="Grid Toggle: Click grids to toggle 127↔255", foreground="blue")
        elif tool == "grid_black":
            self.info_label.config(text="Grid Black Toggle: Click grids to toggle black mask (0/63)", foreground="blue")
        elif tool == "brush_add":
            self.info_label.config(text="Brush (Add): Paint black mask areas", foreground="blue")
        else:  # brush_erase
            self.info_label.config(text="Brush (Erase): Erase black mask areas", foreground="blue")

    def on_brush_size_change(self, value):
        """Handle brush size change"""
        self.brush_size = int(float(value))
        self.brush_size_label.config(text=str(self.brush_size))

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1400x800")
    app = MaskEditorGUI(root)
    root.mainloop()
