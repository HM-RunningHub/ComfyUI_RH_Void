import base64
import gc
import io
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from aiohttp import web
from PIL import Image
from server import PromptServer
import uuid
import torch
import comfy.utils

import folder_paths
try:
    from comfy_api.input_impl.video_types import VideoFromFile
except ImportError:
    VideoFromFile = None

from .passv2v import load_pipeline, USE_VAE_MASK, STACK_MASK
from .videox_fun.utils.utils import get_video_mask_input_from_paths, save_videos_grid, save_inout_row

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".gif"}
PREVIEW_FRAME_STRIDE = 4

MAX_VIDEO_LENGTH = 197


def _clear_cuda_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _pil_image_to_tensor(image, mode="RGB"):
    """Convert a PIL image to a ComfyUI IMAGE tensor: [1, H, W, C], float32, 0..1."""
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got {type(image)!r}")

    image = image.convert(mode)
    image_array = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(image_array).unsqueeze(0)

def _parse_points_by_frame(raw_points):
    if isinstance(raw_points, dict):
        data = raw_points
    else:
        try:
            data = json.loads(raw_points) if raw_points and raw_points.strip() else {}
        except (json.JSONDecodeError, AttributeError):
            data = {}

    normalized = {}
    if not isinstance(data, dict):
        return normalized

    for frame_key, points in data.items():
        try:
            normalized_key = str(int(frame_key))
        except (TypeError, ValueError):
            continue

        frame_points = []
        if isinstance(points, list):
            for point in points:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    continue
                try:
                    x = int(round(float(point[0])))
                    y = int(round(float(point[1])))
                except (TypeError, ValueError):
                    continue
                frame_points.append([x, y])

        if frame_points:
            normalized[normalized_key] = frame_points

    return normalized


def _resolve_uploaded_video_path(filename):
    if not filename:
        return ""

    safe_name = os.path.basename(filename)
    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in VIDEO_EXTENSIONS:
        return ""

    input_dir = folder_paths.get_input_directory()
    full_path = os.path.join(input_dir, safe_name)
    if not os.path.exists(full_path):
        return ""

    return full_path


def _video_to_base64_frames(video_path, frame_stride=PREVIEW_FRAME_STRIDE):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    frames = []
    source_frame_indices = []
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_stride = max(1, int(frame_stride))
    frame_index = 0

    try:
        while True:
            if frame_index > MAX_VIDEO_LENGTH:
                break

            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % frame_stride == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb_frame)
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                frames.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
                source_frame_indices.append(frame_index)

            frame_index += 1
    finally:
        capture.release()

    return {
        "frames": frames,
        "source_frame_indices": source_frame_indices,
        "frame_stride": frame_stride,
        "width": width,
        "height": height,
        "total_frames": total_frames or frame_index,
        "fps": fps,
    }


def _save_video_bytesio_to_temp_file(bytesio_obj):
    temp_dir = folder_paths.get_temp_directory()
    bytesio_obj.seek(0)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".mp4",
        prefix="rh_void_input_",
        dir=temp_dir,
        delete=False,
    ) as temp_file:
        temp_file.write(bytesio_obj.read())
        return temp_file.name


def _resolve_video_input_path(video_input):
    """Extract a filesystem path from a ComfyUI VIDEO input."""
    if video_input is None:
        raise ValueError("Video input is None")

    if isinstance(video_input, str):
        return video_input

    if isinstance(video_input, Path):
        return str(video_input)

    if isinstance(video_input, io.BytesIO):
        return _save_video_bytesio_to_temp_file(video_input)

    if hasattr(video_input, "get_stream_source"):
        source = video_input.get_stream_source()
        if isinstance(source, str):
            return source
        if isinstance(source, Path):
            return str(source)
        if isinstance(source, io.BytesIO):
            return _save_video_bytesio_to_temp_file(source)

    private_attr = "_VideoFromFile__file"
    if hasattr(video_input, private_attr):
        file_obj = getattr(video_input, private_attr)
        if isinstance(file_obj, str):
            return file_obj
        if isinstance(file_obj, Path):
            return str(file_obj)
        if isinstance(file_obj, io.BytesIO):
            return _save_video_bytesio_to_temp_file(file_obj)

    for attr_name in ["path", "file_path", "filename", "source", "_path", "_file_path"]:
        if hasattr(video_input, attr_name):
            path = getattr(video_input, attr_name)
            if isinstance(path, str) and path:
                return path
            if isinstance(path, Path):
                return str(path)

    raise ValueError(f"Cannot extract video path from input type: {type(video_input)!r}")


def _write_stage1_config(video_path, output_dir, prompt, primary_points_by_frame):
    points_by_frame = _parse_points_by_frame(primary_points_by_frame)
    if not points_by_frame:
        raise ValueError("`primary_points_by_frame` is empty or invalid.")

    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "stage1_config.json")
    first_appears_frame = min(int(frame_idx) for frame_idx in points_by_frame.keys())
    config_data = {
        "videos": [
            {
                "video_path": video_path,
                "instruction": prompt,
                "output_dir": output_dir,
                "primary_points_by_frame": points_by_frame,
                "first_appears_frame": first_appears_frame,
            }
        ]
    }

    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(config_data, config_file, ensure_ascii=False, indent=2)

    return config_path


@PromptServer.instance.routes.post("/rh/void/preview_frames")
async def rh_void_preview_frames(request):
    try:
        payload = await request.json()
        filename = payload.get("filename", "")
        frame_stride = payload.get("frame_stride", PREVIEW_FRAME_STRIDE)
        video_path = _resolve_uploaded_video_path(filename)
        if not video_path:
            return web.json_response({"error": "Invalid or missing uploaded video."}, status=400)

        preview = _video_to_base64_frames(video_path, frame_stride=frame_stride)
        return web.json_response(preview)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


class RunningHub_Void_PointEditor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preview_stride": ("INT", {"default": PREVIEW_FRAME_STRIDE, "min": 1, "max": 16}),
                "upload": ("STRING", {"default": ""}),
                "upload_status": ("STRING", {"default": "idle"}),
                "points_store": ("STRING", {"default": "{}", "multiline": True}),
                "coordinates": ("STRING", {"default": "{}", "multiline": True}),
            }
        }

    RETURN_TYPES = ("VIDEO", "RHVoidPoints")
    RETURN_NAMES = ("video", "primary_points_by_frame")
    FUNCTION = "collect_points"
    CATEGORY = "RunningHub/Void"

    def collect_points(self, preview_stride, upload, upload_status, points_store, coordinates):
        confirmed_points = _parse_points_by_frame(coordinates)
        video_path = _resolve_uploaded_video_path(upload) if upload_status == "success" else ""
        video_obj = self._create_video_object(video_path) if video_path else None

        return {
            "ui": {
                "confirmed_coordinates": [json.dumps(confirmed_points)],
                "selected_video": [upload or ""],
            },
            "result": (video_obj, json.dumps(confirmed_points)),
        }

    def _create_video_object(self, video_path):
        if VideoFromFile is not None:
            return VideoFromFile(video_path)
        return video_path

class RunningHub_Void_MaskReasoner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "primary_points_by_frame": ("RHVoidPoints", ),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "analysis_model": (["RunningHub", "gemini-3-flash-preview", "gemini-3-pro-preview"], {"default": "RunningHub"}),
                "api_key": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("VIDEO", "IMAGE", "IMAGE")
    RETURN_NAMES = ("quadmask", "debug_image(with mask)", "debug_image(with grid)")
    FUNCTION = "run"
    CATEGORY = "RunningHub/Void"

    def run(self, **kwargs):
        video = kwargs.get("video", None)
        primary_points_by_frame = str(kwargs.get("primary_points_by_frame", None))
        prompt = kwargs.get("prompt", "")
        analysis_model = kwargs.get("analysis_model")
        print('sample config:', primary_points_by_frame)

        video_path = _resolve_video_input_path(video)
        output_dir = os.path.join(folder_paths.get_temp_directory(), f"void_mask_reasoner_{uuid.uuid4()}")
        config_path = _write_stage1_config(video_path, output_dir, prompt, primary_points_by_frame)

        from .vlm_mask.stage1_sam2_segmentation import process_config as process_config_stage1
        sam2_path = os.path.join(folder_paths.models_dir, "sam2", "sam2.1_hiera_large.pt")
        # sam2_path = os.path.join(folder_paths.models_dir, "sam2", "sam2_hiera_large.pt")
        print(f"[RunningHub Void] Stage 1 config saved to: {config_path}")
        process_config_stage1(config_path, sam2_path)
        _clear_cuda_cache()
        api_key = kwargs.get("api_key", "")

        if analysis_model == "gemini-3-flash-preview" or analysis_model == "gemini-3-pro-preview":
            from .vlm_mask.stage2_vlm_analysis import process_config as process_config_stage2_gemini
            process_config_stage2_gemini(config_path, analysis_model, api_key)
        elif analysis_model == "RunningHub":
            from .vlm_mask.stage2_vlm_analysis_rh import process_config as process_config_stage2_rh
            process_config_stage2_rh(config_path)

        _clear_cuda_cache()

        from .vlm_mask.stage3a_generate_grey_masks_v2 import process_config as process_config_stage3a
        process_config_stage3a(config_path)
        _clear_cuda_cache()

        from .vlm_mask.stage4_combine_masks import process_config as process_config_stage4
        process_config_stage4(config_path)
        _clear_cuda_cache()

        output_path = os.path.join(output_dir, "quadmask_0.mp4")
        debug_image_with_mask_path = os.path.join(output_dir, "first_frame_with_mask.jpg")
        debug_image_with_mask = _pil_image_to_tensor(Image.open(debug_image_with_mask_path))
        debug_image_with_grid_path = os.path.join(output_dir, "first_frame_with_grid.jpg")
        debug_image_with_grid = _pil_image_to_tensor(Image.open(debug_image_with_grid_path))

        return (VideoFromFile(output_path), debug_image_with_mask, debug_image_with_grid)

class RunningHub_Void_PassLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                
            },
        }

    RETURN_TYPES = ("RH_Void_PIPELINE", )
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load"
    CATEGORY = "RunningHub/Void"

    OUTPUT_NODE = True

    def load(self, **kwargs):
        model_path = os.path.join(folder_paths.models_dir, "diffusers", "CogVideoX-Fun-V1.5-5b-InP")
        transformer_path = os.path.join(folder_paths.models_dir, "void-model", "void_pass1.safetensors")
        pipeline, vae = load_pipeline(model_path, transformer_path)
        return ({'pipeline': pipeline, 'vae': vae}, )

class RunningHub_Void_PassSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("RH_Void_PIPELINE", ),
                "source": ("VIDEO", ),
                "quadmask": ("VIDEO", ),
                "prompt": ("STRING", {"default": "A video of a rubber ducky.", "multiline": True}),
                "width": ("INT", {"default": 720, "min": 16, "max": 10000, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 10000, "step": 16}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 50}),
                "num_frames": ("INT", {"default": 85, "min": 1, "max": MAX_VIDEO_LENGTH, "step": 4}),
                "fps": ("INT", {"default": 12, "min": 1, "max": 30}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "sample"
    CATEGORY = "RunningHub/Void"

    def sample(self, **kwargs):
        void_pipeline = kwargs.get("pipeline", None)
        pipeline = void_pipeline['pipeline']
        vae = void_pipeline['vae']

        video = kwargs.get("source", None)
        quadmask = kwargs.get("quadmask", None)
        prompt = kwargs.get("prompt", "")
        video_path = _resolve_video_input_path(video)
        quadmask_path = _resolve_video_input_path(quadmask)
        seed = kwargs.get("seed", 42) % (2 ** 32)
        steps = kwargs.get("steps")
        width = kwargs.get("width")
        height = kwargs.get("height")
        num_frames = kwargs.get("num_frames")
        fps = kwargs.get("fps")

        # kiki: hardcode 
        neg_prompt = (
            "The video is not of a high quality, it has a low resolution. "
            "Watermark present in each frame. The background is solid. "
            "Strange body and strange trajectory. Distortion. "
        )
        guidance_scale = 1.0
        denoise_strength = 1.0

        video_length = MAX_VIDEO_LENGTH
        video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
        print(f'Video length: {video_length}')
        sample_size = height, width
        input_video, input_video_mask, prompt, _ = get_video_mask_input_from_paths(
            video_path=video_path,
            quadmask_path=quadmask_path,
            prompt=prompt,
            sample_size=sample_size,
            max_video_length=video_length,
            temporal_window_size=num_frames,
            use_trimask=False,
            use_quadmask=True,
            dilate_width=11,
        )

        _clear_cuda_cache()
        generator = torch.Generator(device="cuda").manual_seed(seed)
        self.pbar = comfy.utils.ProgressBar(total=steps)

        try:
            with torch.no_grad():
                sample = pipeline(
                    prompt,
                    num_frames = num_frames,
                    negative_prompt = neg_prompt,
                    height      = sample_size[0],
                    width       = sample_size[1],
                    generator   = generator,
                    guidance_scale = guidance_scale,
                    num_inference_steps = steps,
                    video       = input_video,
                    mask_video  = input_video_mask,
                    strength    = denoise_strength,
                    use_trimask = True,
                    zero_out_mask_region = False,
                    skip_unet = False,
                    use_vae_mask = USE_VAE_MASK,
                    stack_mask = STACK_MASK,
                    update_func = self.update,
                ).videos

            output_path = os.path.join(folder_paths.get_output_directory(), f"void_pass_sampler_{uuid.uuid4()}.mp4")

            save_videos_grid(sample, output_path, fps=fps)
            return (VideoFromFile(output_path), )
        finally:
            if "sample" in locals():
                del sample
            del input_video
            del input_video_mask
            del generator
            _clear_cuda_cache()

    def update(self):
        self.pbar.update(1)

NODE_CLASS_MAPPINGS = {
    "RunningHub Void Point Editor": RunningHub_Void_PointEditor,
    "RunningHub Void Mask Reasoner": RunningHub_Void_MaskReasoner,
    "RunningHub Void Pass Loader": RunningHub_Void_PassLoader,
    "RunningHub Void Pass Sampler": RunningHub_Void_PassSampler,
}
