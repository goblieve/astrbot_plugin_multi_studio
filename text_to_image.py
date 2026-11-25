import math
from typing import Optional

from astrbot.api import logger

from .comfyui_api import ComfyUIAPI
from .workflow import WorkflowTemplate, randomize_seeds


class TextToImage:
    MAX_OUTPUT_PIXELS = 2048 * 2048
    MIN_DIMENSION = 64
    MAX_DIMENSION = 4096
    ALIGN = 8

    def __init__(self, api: ComfyUIAPI, workflow_path: str):
        self.api = api
        self.template = WorkflowTemplate(workflow_path, ("positive",))

    @classmethod
    def _clamp_dimensions(cls, width: int, height: int) -> tuple:
        """将用户传入的 width/height 钳制到合法范围内并按 ALIGN 对齐"""
        w = max(cls.MIN_DIMENSION, min(int(width), cls.MAX_DIMENSION))
        h = max(cls.MIN_DIMENSION, min(int(height), cls.MAX_DIMENSION))

        pixels = w * h
        if pixels > cls.MAX_OUTPUT_PIXELS:
            scale = math.sqrt(cls.MAX_OUTPUT_PIXELS / pixels)
            w = int(w * scale)
            h = int(h * scale)

        align = cls.ALIGN
        w = max(align, (w // align) * align)
        h = max(align, (h // align) * align)
        return w, h

    async def generate(self, prompt: str, negative: str = "bad hands", width: int = None, height: int = None,
                       scale: float = None, max_wait: float = 300.0, on_wait_callback=None,
                       on_submitted_callback=None, on_result_reference=None,
                       on_progress=None) -> Optional[bytes]:
        workflow = self.template.copy()
        self.template.set(workflow, "positive", prompt)
        if "negative" in self.template.bindings:
            self.template.set(workflow, "negative", negative)
        if (width is not None or height is not None) and "width" in self.template.bindings and "height" in self.template.bindings:
            width = self.template.get(workflow, "width") if width is None else width
            height = self.template.get(workflow, "height") if height is None else height
            try:
                width, height = self._clamp_dimensions(width, height)
            except (TypeError, ValueError) as error:
                logger.warning(f"[ComfyUI] 分辨率参数无效，已忽略：{error}")
                width = height = None
            if width is not None:
                self.template.set(workflow, "width", width)
                self.template.set(workflow, "height", height)
        if scale is not None and "scale" in self.template.bindings:
            self.template.set(workflow, "scale", scale)
        randomize_seeds(workflow)
        result = await self.api.queue_and_wait_image(
            workflow, max_wait=max_wait, on_wait_callback=on_wait_callback,
            on_submitted_callback=on_submitted_callback,
            on_result_reference=on_result_reference,
            output_node=self.template.node_id("output_image") if "output_image" in self.template.bindings else (
                self.template.node_id("output") if "output" in self.template.bindings else None
            ),
            on_progress=on_progress,
        )
        if not result:
            logger.error("[ComfyUI] SD 生成失败或等待结果超时")
        return result
