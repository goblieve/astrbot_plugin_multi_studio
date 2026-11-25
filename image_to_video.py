import asyncio
import math
import uuid
from io import BytesIO
from typing import Optional

from PIL import Image as PILImage
import aiohttp
from astrbot.api import logger

from .comfyui_api import ComfyUIAPI
from .workflow import WorkflowTemplate, randomize_seeds


class ImageToVideo:
    MAX_OUTPUT_PIXELS = 1024 * 1024

    def __init__(self, api: ComfyUIAPI, workflow_path: str, max_frames: int = 240):
        self.api = api
        self.template = WorkflowTemplate(workflow_path, ("positive", "image", "width", "height"))
        self.max_frames = max_frames

    @staticmethod
    def _extract_first_frame_if_gif(image_data: bytes) -> Optional[bytes]:
        """如果输入是动图（GIF/WebP），提取第一帧"""
        try:
            with PILImage.open(BytesIO(image_data)) as img:
                is_animated = getattr(img, 'is_animated', False)
                if is_animated:
                    logger.info("[ComfyUI] 检测到动图，将使用首帧")
                    img.seek(0)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    output = BytesIO()
                    img.save(output, format='PNG')
                    return output.getvalue()
                return image_data
        except Exception as e:
            logger.error(f"[ComfyUI] 动图检测/处理失败: {e}")
            return None

    def _calc_output_size(self, image_data: bytes) -> Optional[tuple]:
        """根据输入图像计算输出分辨率

        - 输入图像像素 ≤ 1M：直接使用原始尺寸
        - 输入图像像素 > 1M：保持比例缩放至 1M 像素以内
        - 宽高对齐到 16 的倍数（视频模型常见约束）

        注意：仅计算并写入工作流的输出分辨率，不修改输入图像本身。

        Returns:
            (width, height) 或 None（解析失败）
        """
        try:
            with PILImage.open(BytesIO(image_data)) as img:
                w, h = img.size
        except Exception as e:
            logger.error(f"[ComfyUI] 读取图像尺寸失败: {e}")
            return None

        if w <= 0 or h <= 0:
            return None

        pixels = w * h
        if pixels > self.MAX_OUTPUT_PIXELS:
            scale = math.sqrt(self.MAX_OUTPUT_PIXELS / pixels)
            w = int(w * scale)
            h = int(h * scale)

        align = 16
        w = max(align, (w // align) * align)
        h = max(align, (h // align) * align)
        return w, h

    async def generate(self, image_data: bytes, prompt: str, negative: str = "",
                       fps: Optional[float] = None, length: Optional[float] = None,
                       max_wait: float = 300.0, on_wait_callback=None,
                       on_submitted_callback=None, on_progress=None,
                       on_result_reference=None) -> Optional[bytes]:
        workflow = self.template.copy()
        processed = await asyncio.to_thread(self._extract_first_frame_if_gif, image_data)
        if processed is None:
            return None
        try:
            filename = await self.api.upload_image("i2v.png", processed)
        except (aiohttp.ClientError, TimeoutError) as error:
            logger.error(f"[ComfyUI] 视频输入图片上传失败：{error}")
            return None
        if not filename:
            logger.error("[ComfyUI] 视频输入图片上传失败")
            return None
        self.template.set(workflow, "image", filename)
        size = await asyncio.to_thread(self._calc_output_size, processed)
        if size:
            self.template.set(workflow, "width", size[0])
            self.template.set(workflow, "height", size[1])
        fps_default = self.template.get(workflow, "fps")
        length_default = self.template.get(workflow, "length")
        final_fps = float(fps) if fps is not None else (float(fps_default) if fps_default is not None else None)
        final_length = float(length) if length is not None else (float(length_default) if length_default is not None else None)
        if final_fps is not None:
            final_fps = max(1.0, final_fps)
        if self.max_frames > 0 and final_fps is not None and final_length is not None:
            final_length = min(final_length, self.max_frames / final_fps)
        if fps is not None and "fps" in self.template.bindings:
            self.template.set(workflow, "fps", final_fps)
        if final_length is not None and "length" in self.template.bindings:
            self.template.set(workflow, "length", final_length)
        self.template.set(workflow, "positive", prompt)
        if negative and "negative" in self.template.bindings:
            self.template.set(workflow, "negative", negative)
        randomize_seeds(workflow)
        result = await self.api.queue_and_wait_video(
            workflow, max_wait=max_wait, on_wait_callback=on_wait_callback,
            on_submitted_callback=on_submitted_callback, on_progress=on_progress,
            on_result_reference=on_result_reference,
        )
        if not result:
            logger.error("[ComfyUI] 视频生成失败或等待结果超时")
        return result
