import asyncio
import json
import math
import uuid
from io import BytesIO
from fractions import Fraction
from typing import Optional

from PIL import Image as PILImage, ImageOps as PILImageOps
from astrbot.api import logger

from .comfyui_api import ComfyUIAPI
from .workflow import WorkflowTemplate, randomize_seeds


class VisualLoRAControls:
    @staticmethod
    def _load_lora_aliases(rows):
        aliases = {}
        if isinstance(rows, str):
            rows = [rows]
        for row in rows or []:
            if not isinstance(row, str) or "=" not in row:
                continue
            alias, filename = row.split("=", 1)
            alias = alias.strip()
            filename = filename.strip().replace("\\", "/").rsplit("/", 1)[-1]
            if (
                not alias
                or not filename
                or any(char.isspace() for char in alias)
                or "=" in alias
                or alias.casefold() in {"off", "none"}
            ):
                continue
            key = alias.casefold()
            if key in aliases:
                continue
            aliases[key] = filename
        return aliases

    @staticmethod
    def _lora_basename(value):
        return str(value or "").replace("\\", "/").rsplit("/", 1)[-1].casefold()

    def _apply_lora_selection(self, workflow, selection):
        if selection is None:
            return
        loaders = [
            node for node in workflow.values()
            if isinstance(node, dict)
            and node.get("class_type") == "VisualLoRALoader"
            and isinstance(node.get("inputs"), dict)
            and isinstance(node["inputs"].get("lora_data"), str)
        ]
        if not loaders and not selection:
            return
        if len(loaders) != 1:
            raise ValueError(
                "当前 /draw 工作流必须包含且仅包含一个 VisualLoRALoader 节点，才能按别名选择 LoRA"
            )
        inputs = loaders[0]["inputs"]
        try:
            entries = json.loads(inputs["lora_data"])
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("当前工作流的 VisualLoRALoader.lora_data 不是有效 JSON") from error
        if not isinstance(entries, list):
            raise ValueError("当前工作流的 VisualLoRALoader.lora_data 必须是列表")

        matches_by_filename = {}
        for entry in entries:
            if isinstance(entry, dict) and entry.get("name"):
                key = self._lora_basename(entry["name"])
                matches_by_filename.setdefault(key, []).append(entry)

        selected_entries = []
        selected_files = set()
        for alias, strength in selection.items():
            filename = self.lora_aliases.get(str(alias).casefold())
            if filename is None:
                available = ", ".join(sorted(self.lora_aliases))
                raise ValueError(
                    f"未配置 LoRA 别名 {alias}；请在插件配置 draw_lora_aliases 中添加。"
                    f"当前可用别名：{available or '无'}"
                )
            filename_key = self._lora_basename(filename)
            if filename_key in selected_files:
                continue
            matches = matches_by_filename.get(filename_key, [])
            if len(matches) != 1:
                state = "不存在" if not matches else "出现多次"
                raise ValueError(
                    f"别名 {alias} 对应的文件 {filename} 在当前 /draw 工作流中{state}，请核对配置"
                )
            selected_entries.append((matches[0], strength))
            selected_files.add(filename_key)

        for entry in entries:
            if isinstance(entry, dict):
                entry["enabled"] = False
        for entry, strength in selected_entries:
            entry["enabled"] = True
            if strength is not None:
                entry["strength_model"] = strength
                entry["strength_clip"] = strength

        inputs["lora_data"] = json.dumps(entries, ensure_ascii=False, separators=(",", ":"))


class QwenImage(VisualLoRAControls):
    # ComfyUI 0.37 的 TextEncodeQwenImage21 提供 16 个 Autogrow 输入；官方公开支持 10 张。
    MAX_REFERENCE_IMAGES = 16
    # Cap Qwen edit references at 2,000,000 pixels to limit sampling cost.
    MAX_REFERENCE_PIXELS = 2_000_000

    def __init__(self, api: ComfyUIAPI, workflow_path: str | None, lora_aliases=None, *, template=None):
        self.api = api
        self.lora_aliases = self._load_lora_aliases(lora_aliases)
        self.template = template if template is not None else WorkflowTemplate(
            workflow_path, ("positive", "negative", "width", "height", "latent", "output_image")
        )
        self.encode_node = self.template.node_id("positive")
        if self.template.workflow[self.encode_node]["class_type"] != "TextEncodeQwenImage21":
            raise ValueError("Qwen 工作流需要 TextEncodeQwenImage21 节点")
        if self.template.node_id("negative") != self.encode_node:
            raise ValueError("Qwen 的正负提示词必须绑定同一个 TextEncodeQwenImage21 节点")

    def dimensions_for_aspect_ratio(self, ratio, width=None, height=None):
        """Resolve a ratio to exact 32-aligned dimensions near the workflow pixel budget."""
        try:
            ratio = ratio if isinstance(ratio, Fraction) else Fraction(str(ratio))
        except (TypeError, ValueError, ZeroDivisionError, OverflowError) as error:
            raise ValueError("长宽比无效") from error
        if ratio <= 0 or not Fraction(1, 4) <= ratio <= Fraction(4, 1):
            raise ValueError("长宽比范围为 1:4 到 4:1")
        ratio = ratio.limit_denominator(16)
        width_unit = 32 * ratio.numerator
        height_unit = 32 * ratio.denominator

        if width is not None and height is not None:
            try:
                requested_width = int(width)
                requested_height = int(height)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("宽度和高度必须是正整数") from error
            if requested_width <= 0 or requested_height <= 0:
                raise ValueError("宽度和高度必须大于 0")
            requested_ratio = Fraction(requested_width, requested_height)
            if abs(float(requested_ratio / ratio) - 1) > 0.025:
                raise ValueError("比例参数与宽度/高度不匹配；请调整尺寸，或只指定比例")
            pixel_budget = requested_width * requested_height
        elif width is not None:
            try:
                requested_width = int(width)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("宽度必须是正整数") from error
            if requested_width <= 0:
                raise ValueError("宽度必须大于 0")
            multiplier = max(1, int(round(requested_width / width_unit)))
            return width_unit * multiplier, height_unit * multiplier
        elif height is not None:
            try:
                requested_height = int(height)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("高度必须是正整数") from error
            if requested_height <= 0:
                raise ValueError("高度必须大于 0")
            multiplier = max(1, int(round(requested_height / height_unit)))
            return width_unit * multiplier, height_unit * multiplier
        else:
            try:
                base_width = int(self.template.get(self.template.workflow, "width"))
                base_height = int(self.template.get(self.template.workflow, "height"))
            except (TypeError, ValueError, OverflowError):
                base_width = base_height = 1024
            if base_width <= 0 or base_height <= 0:
                base_width = base_height = 1024
            pixel_budget = base_width * base_height

        target_multiplier = math.sqrt(pixel_budget / (width_unit * height_unit))
        candidates = {max(1, math.floor(target_multiplier)), max(1, math.ceil(target_multiplier))}
        multiplier = min(
            candidates,
            key=lambda value: abs(width_unit * height_unit * value * value - pixel_budget),
        )
        return width_unit * multiplier, height_unit * multiplier

    @staticmethod
    def _prepare_image(data: bytes, preserve_original_size=False) -> bytes:
        with PILImage.open(BytesIO(data)) as source:
            width, height = source.size
            is_animated = bool(getattr(source, "is_animated", False))
            has_alpha = "A" in source.getbands() or "transparency" in source.info
            needs_resize = (
                not preserve_original_size
                and width * height > QwenImage.MAX_REFERENCE_PIXELS
            )
            if not is_animated and not needs_resize and not preserve_original_size:
                return data
            if is_animated:
                source.seek(0)
            oriented = PILImageOps.exif_transpose(source) if preserve_original_size else source
            image = oriented.convert("RGBA" if has_alpha else "RGB")
            if needs_resize:
                ratio = math.sqrt(QwenImage.MAX_REFERENCE_PIXELS / (width * height))
                scaled_width = int(width * ratio)
                scaled_height = int(height * ratio)
                if scaled_width < 32 or scaled_height < 32:
                    raise ValueError("缩至 2MP 上限后短边不足 32 像素，无法可靠保持长宽比")
                target_width = (scaled_width // 32) * 32
                target_height = (scaled_height // 32) * 32
                image = image.resize((target_width, target_height), PILImage.Resampling.LANCZOS)
            output = BytesIO()
            image.save(output, format="PNG", compress_level=1)
            return output.getvalue()
    @staticmethod
    def _linked_node_ids(value, workflow):
        if isinstance(value, list):
            if (
                len(value) == 2
                and str(value[0]) in workflow
                and isinstance(value[1], int)
            ):
                yield str(value[0])
                return
            for item in value:
                yield from QwenImage._linked_node_ids(item, workflow)
        elif isinstance(value, dict):
            for item in value.values():
                yield from QwenImage._linked_node_ids(item, workflow)

    def _reachable_output_nodes(self, workflow):
        pending = list(self._linked_node_ids(self.template.get(workflow, "output_image"), workflow))
        reachable = set()
        while pending:
            node_id = pending.pop()
            if node_id in reachable:
                continue
            reachable.add(node_id)
            node = workflow.get(node_id, {})
            for value in node.get("inputs", {}).values():
                pending.extend(self._linked_node_ids(value, workflow))
        return reachable

    def _skip_workflow_upscale(self, workflow):
        """Route the final image around static upscaling and its post-processing branch."""
        reachable = self._reachable_output_nodes(workflow)
        has_upscale = any(
            "upscale" in str(workflow.get(node_id, {}).get("class_type", "")).casefold()
            for node_id in reachable
        )
        if not has_upscale:
            return False

        decoded_nodes = [
            node_id for node_id in reachable
            if str(workflow.get(node_id, {}).get("class_type", "")).casefold()
            in {"vaedecode", "vaedecodetiled"}
        ]
        if len(decoded_nodes) != 1:
            raise ValueError(
                "当前 Qwen 工作流包含放大节点，但无法唯一定位放大前的 VAE 解码图像；"
                "为避免输出错误，未跳过放大。"
            )

        self.template.set(workflow, "output_image", [decoded_nodes[0], 0])
        logger.info("[ComfyUI] 改图选项未选择“放大”，已跳过工作流放大节点及其后处理")
        return True

    def _route_workflow_upscale_from_original(self, workflow, image_node_id):
        """Feed the uploaded original into the workflow upscaler, bypassing Qwen sampling."""
        reachable = self._reachable_output_nodes(workflow)
        upscalers = [
            node_id for node_id in reachable
            if "upscale" in str(workflow.get(node_id, {}).get("class_type", "")).casefold()
            and isinstance(workflow.get(node_id, {}).get("inputs"), dict)
            and "image" in workflow[node_id]["inputs"]
        ]
        if len(upscalers) != 1:
            raise ValueError(
                "单独放大需要当前 Qwen 工作流有且仅有一个可用的放大图像节点。"
            )

        upscaler = workflow[upscalers[0]]
        source_ids = list(self._linked_node_ids(upscaler["inputs"].get("image"), workflow))
        source_node = workflow.get(source_ids[0], {}) if len(source_ids) == 1 else {}
        upscaler["inputs"]["image"] = [image_node_id, 0]
        if str(source_node.get("class_type", "")).casefold() == "splitimagewithalpha":
            split_node_id = source_ids[0]
            alpha_links_rewired = 0
            for node_id in reachable:
                node = workflow.get(node_id, {})
                for input_name, value in node.get("inputs", {}).items():
                    if (
                        isinstance(value, list)
                        and len(value) == 2
                        and str(value[0]) == split_node_id
                        and value[1] == 1
                    ):
                        node["inputs"][input_name] = [image_node_id, 1]
                        alpha_links_rewired += 1
            if not alpha_links_rewired:
                raise ValueError("当前 Qwen 工作流的 alpha 通道无法接到原图，未执行单独放大")
        logger.info("[ComfyUI] 仅选择“放大”，已将原图直接接入放大节点，跳过 Qwen 编辑")

    def build_workflow(self, filenames, prompt, negative="", width=None, height=None, scale=None,
                       lora_selection=None, workflow_upscale=None, workflow_upscale_only=False):
        # AstrBot tool calls may use 0 for omitted optional image parameters.
        width = None if width == 0 else width
        height = None if height == 0 else height
        scale = None if scale == 0 else scale
        if len(filenames) > self.MAX_REFERENCE_IMAGES:
            raise ValueError(f"当前 ComfyUI 节点最多接收 {self.MAX_REFERENCE_IMAGES} 张参考图")
        workflow = self.template.copy()
        encoder_inputs = workflow[self.encode_node]["inputs"]
        if "resolution" in encoder_inputs:
            encoder_inputs["resolution"] = 0
        self._apply_lora_selection(workflow, lora_selection)
        self.template.set(workflow, "positive", prompt)
        self.template.set(workflow, "negative", negative)
        inputs = workflow[self.encode_node]["inputs"]
        for key in list(inputs):
            if key.startswith("images.image_"):
                del inputs[key]
        image_node_ids = []
        for index, filename in enumerate(filenames, 1):
            node_id = f"hub_image_{index}"
            while node_id in workflow:
                node_id += "_"
            workflow[node_id] = {"class_type": "LoadImage", "inputs": {"image": filename}}
            image_node_ids.append(node_id)
            inputs[f"images.image_{index}"] = [node_id, 0]

        if workflow_upscale_only:
            if workflow_upscale is not True:
                raise ValueError("单独放大必须启用工作流放大节点")
            if len(image_node_ids) != 1:
                raise ValueError("单独放大只支持一张原图")
            self._route_workflow_upscale_from_original(workflow, image_node_ids[0])

        if not workflow_upscale_only:
            if filenames and width is None and height is None:
                empty_node = self.template.get(workflow, "latent")[0]
                self.template.set(workflow, "latent", [self.encode_node, 2])
                if workflow[empty_node]["class_type"] == "EmptyLatentImage" and not any(
                    isinstance(value, list) and len(value) == 2 and value[0] == empty_node
                    for node in workflow.values() for value in node["inputs"].values()
                ):
                    del workflow[empty_node]
            else:
                for role, value in (("width", width), ("height", height)):
                    if value is not None:
                        self.template.set(workflow, role, max(32, (int(value) + 31) // 32 * 32))

        if workflow_upscale is False:
            self._skip_workflow_upscale(workflow)
            scale = None

        if scale is not None:
            node_id = "hub_upscale"
            while node_id in workflow:
                node_id += "_"
            workflow[node_id] = {"class_type": "ImageScaleBy", "inputs": {
                "image": self.template.get(workflow, "output_image"),
                "upscale_method": "lanczos", "scale_by": scale,
            }}
            self.template.set(workflow, "output_image", [node_id, 0])
        randomize_seeds(workflow)
        return workflow

    async def generate(self, prompt: str, negative: str = "", image_data_list=None,
                       width=None, height=None, scale=None, max_wait: float = 300.0,
                       on_wait_callback=None, on_submitted_callback=None, lora_selection=None,
                       workflow_upscale=None, workflow_upscale_only=False, on_result_reference=None,
                       on_progress=None) -> Optional[bytes]:
        images = image_data_list or []
        if len(images) > self.MAX_REFERENCE_IMAGES:
            raise ValueError(f"当前 ComfyUI 节点最多接收 {self.MAX_REFERENCE_IMAGES} 张参考图")
        if workflow_upscale_only and len(images) != 1:
            raise ValueError("单独放大需要且只支持一张原图")
        filenames = []
        for index, data in enumerate(images, 1):
            try:
                processed = await asyncio.to_thread(
                    self._prepare_image, data, preserve_original_size=workflow_upscale_only
                )
            except (OSError, ValueError) as error:
                raise ValueError(f"第 {index} 张图片处理失败：{error}") from error
            filename = await self.api.upload_image(f"qwen21_{index}.png", processed)
            if not filename:
                raise ValueError(f"第 {index} 张图片上传失败，请重试")
            filenames.append(filename)
        workflow = self.build_workflow(
            filenames, prompt, negative, width, height, scale, lora_selection,
            workflow_upscale, workflow_upscale_only
        )
        result = await self.api.queue_and_wait_image(
            workflow, max_wait=max_wait, on_wait_callback=on_wait_callback,
            on_submitted_callback=on_submitted_callback,
            on_result_reference=on_result_reference,
            output_node=self.template.node_id("output_image"),
            on_progress=on_progress,
        )
        if not result:
            logger.error("[ComfyUI] Qwen 图片生成失败或等待结果超时")
        return result
