import uuid
from typing import Optional

import aiohttp
from astrbot.api import logger

from .comfyui_api import ComfyUIAPI
from .workflow import WorkflowTemplate


class ImageToText:
    def __init__(self, api: ComfyUIAPI, workflow_path: str):
        self.api = api
        self.template = WorkflowTemplate(workflow_path, ("image", "output"))

    async def generate(self, image_data: bytes, max_wait: float = 300.0, on_wait_callback=None, on_submitted_callback=None) -> Optional[str]:
        workflow = self.template.copy()
        try:
            filename = await self.api.upload_image("tagger.png", image_data)
        except (aiohttp.ClientError, TimeoutError) as error:
            logger.error(f"[ComfyUI] Tagger图片上传失败：{error}")
            return None
        if not filename:
            logger.error("[ComfyUI] Tagger 图片上传失败")
            return None
        self.template.set(workflow, "image", filename)
        result = await self.api.queue_and_wait_text(
            workflow, self.template.node_id("output"), max_wait=max_wait,
            on_wait_callback=on_wait_callback, on_submitted_callback=on_submitted_callback,
        )
        if not result:
            logger.error("[ComfyUI] Tagger 生成失败或等待结果超时")
        return result
