"""Mask input and automatic SAM 3 segmentation for local ComfyUI."""

import asyncio
import json
import math
import re
from pathlib import Path
import uuid
from dataclasses import dataclass
from io import BytesIO

from PIL import Image, ImageChops, ImageFilter, ImageOps

from .qwen_image import VisualLoRAControls
from .workflow import WorkflowTemplate, randomize_seeds


MAX_EDIT_PIXELS = 12_000_000
MAX_SOURCE_PIXELS = 64_000_000
MAX_EDIT_EDGE = 8192
MAX_PREVIEW_EDGE = 2048
MASK_ORIGINAL_FORMATS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "BMP": "bmp", "TIFF": "tiff"}
JPEG_SCAN_MARKER = re.compile(rb"\xff[^\x00\xff\xd0-\xd7]")
MPO_THUMBNAIL_TYPES = {"Large Thumbnail (VGA Equivalent)", "Large Thumbnail (Full HD Equivalent)"}
MAX_MASK_BYTES = 8 * 1024 * 1024
MAX_MASK_SOURCE_BYTES = 48 * 1024 * 1024
MAX_REFERENCE_BYTES = 8 * 1024 * 1024
MAX_EDIT_BYTES = 20 * 1024 * 1024
MASK_ROLES = ("image", "mask", "positive", "output")
ALTERNATE_ROLES = ("mask_remove_output", "mask_full_output", "mask_full_reference")
LORA_SECTIONS = {"repaint": "output", "remove": "mask_remove_output"}
LORA_STRENGTHS = ("strength_model", "strength_clip")


def normalize_mask_lora_defaults(value):
    if not isinstance(value, dict) or value.keys() - LORA_SECTIONS.keys():
        raise ValueError("局部重绘 LoRA 设置格式无效")
    for rows in value.values():
        if not isinstance(rows, dict) or len(rows) > 64:
            raise ValueError("每条支路最多配置 64 个 LoRA")
        for name, row in rows.items():
            if not isinstance(name, str) or not 1 <= len(name) <= 512 or not isinstance(row, dict):
                raise ValueError("局部重绘 LoRA 条目格式无效")
            if type(row.get("enabled")) is not bool or any(
                type(row.get(key)) not in (int, float) or not -10 <= row[key] <= 10
                for key in LORA_STRENGTHS
            ):
                raise ValueError("局部重绘 LoRA 强度必须在 -10 至 10 之间")
    return {
        section: {
            name: {key: row[key] for key in ("enabled", *LORA_STRENGTHS)}
            for name, row in rows.items()
        }
        for section, rows in value.items() if rows
    }


def _visual_lora_rows(workflow):
    for node in workflow.values():
        inputs = node["inputs"]
        if node["class_type"] != "VisualLoRALoader" or not isinstance(inputs.get("lora_data"), str):
            continue
        try:
            rows = json.loads(inputs["lora_data"])
        except ValueError:
            continue
        if isinstance(rows, list):
            yield inputs, rows


def _lora_path(row):
    name = row.get("name") if isinstance(row, dict) else None
    return name.replace("\\", "/").lstrip("/") if isinstance(name, str) else ""


def _png(image):
    output = BytesIO()
    image.save(output, format="PNG", compress_level=1)
    return output.getvalue()


def _bounded_dimensions(size, max_pixels, max_edge):
    width, height = size
    factor = min(1, math.sqrt(max_pixels / (width * height)), max_edge / max(size))
    return tuple(max(1, int(side * factor)) for side in size)


def _edit_dimensions(size):
    return _bounded_dimensions(size, MAX_EDIT_PIXELS, MAX_EDIT_EDGE)


@dataclass(frozen=True)
class MaskOriginal:
    data: bytes
    size: tuple
    extension: str
    preview: bytes
    preview_size: tuple
    preview_extension: str


def _primary_photo_bytes(data):
    """Discard MPF thumbnails without decoding or re-encoding the primary JPEG."""
    if not data.startswith(b"\xff\xd8"):
        return data
    with Image.open(BytesIO(data), formats=["JPEG"]) as source:
        if source.format != "MPO":
            return data
        pictures = source.mpinfo.get(0xB002, ())
        if (not pictures or len(pictures) != source.n_frames or
                pictures[0]["Attribute"]["MPType"] != "Baseline MP Primary Image" or
                any(picture["Attribute"]["MPType"] not in MPO_THUMBNAIL_TYPES for picture in pictures[1:])):
            raise ValueError("请选择单张照片。这份 MPO 包含多张独立图像。")

    view = memoryview(data)
    segments = [view[:2]]
    offset = 2
    while offset < len(data) and data[offset] == 0xff:
        start = offset
        while offset < len(data) and data[offset] == 0xff:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker == 0xd9:
            segments.append(view[start:offset])
            return b"".join(segments)
        if marker == 0x01:
            segments.append(view[start:offset])
            continue
        if marker in (0x00, 0xd8) or 0xd0 <= marker <= 0xd7 or offset + 2 > len(data):
            break
        length = int.from_bytes(view[offset:offset + 2], "big")
        end = offset + length
        if length < 2 or end > len(data):
            break
        # Drop only the multi-picture index. Preserve EXIF, ICC and all JPEG scan data.
        if marker != 0xe2 or length < 6 or data[offset + 2:offset + 6] != b"MPF\x00":
            segments.append(view[start:end])
        offset = end
        if marker == 0xda:
            next_marker = JPEG_SCAN_MARKER.search(data, offset)
            if next_marker is None:
                break
            segments.append(view[offset:next_marker.start()])
            offset = next_marker.start()
    raise ValueError("无法从 JPG 读取完整的主照片。")


def prepare_mask_original(data):
    with Image.open(BytesIO(_primary_photo_bytes(data)), formats=list(MASK_ORIGINAL_FORMATS)) as source:
        if getattr(source, "is_animated", False):
            raise ValueError("请上传静态图片")
        width, height = source.size
        if width * height > MAX_SOURCE_PIXELS:
            raise ValueError("原图超过 6400 万像素，请先缩小后再引用")
        orientation = source.getexif().get(274, 1)
        # Pillow decodes TIFF orientation itself, including its dimensions.
        if source.format == "TIFF":
            source.load()
            width, height = source.size
            orientation = source.getexif().get(274, 1)
        size = (height, width) if orientation in (5, 6, 7, 8) else (width, height)
        extension = MASK_ORIGINAL_FORMATS[source.format]
        target = _bounded_dimensions(source.size, 3_000_000, MAX_PREVIEW_EDGE)
        if target != source.size:
            source.draft(source.mode, target)
        preview = ImageOps.exif_transpose(source)
        preview.thumbnail(_bounded_dimensions(preview.size, 3_000_000, MAX_PREVIEW_EDGE), Image.Resampling.LANCZOS)
        if "A" in preview.getbands() or "transparency" in preview.info:
            image = _png(preview.convert("RGBA"))
            preview_extension = "png"
        else:
            output = BytesIO()
            preview.convert("RGB").save(output, format="JPEG", quality=90)
            image = output.getvalue()
            preview_extension = "jpg"
        return MaskOriginal(data, size, extension, image, preview.size, preview_extension)


def prepare_edit_image(data, *, formats=None, resize_large=False):
    with Image.open(BytesIO(_primary_photo_bytes(data)), formats=formats) as src:
        if formats and getattr(src, "is_animated", False):
            raise ValueError("请上传静态图片")
        w, h = src.size
        if resize_large and w * h > MAX_SOURCE_PIXELS:
            raise ValueError("原图超过 6400 万像素，请先缩小后再引用")
        if resize_large and _edit_dimensions(src.size) != src.size:
            src.draft(src.mode, _edit_dimensions(src.size))
        elif not resize_large and w * h > MAX_EDIT_PIXELS:
            raise ValueError("局部重绘原图最多 1200 万像素，请先缩小图片")
        oriented = ImageOps.exif_transpose(src)
        if resize_large:
            oriented.thumbnail(_edit_dimensions(oriented.size), Image.Resampling.LANCZOS)
        image = oriented.convert("RGBA" if "A" in oriented.getbands() or "transparency" in oriented.info else "RGB")
        result = _png(image)
        while resize_large and len(result) > MAX_EDIT_BYTES:
            image.thumbnail(tuple(max(1, int(side * 0.8)) for side in image.size), Image.Resampling.LANCZOS)
            result = _png(image)
        if len(result) > MAX_EDIT_BYTES:
            raise ValueError("原图转换后超过 20 MiB，请先缩小图片")
        return result, image.size


MASK_GROW_LIMIT = 100
MASK_FEATHER_LIMIT = 100


def _shifted(mask, dx, dy):
    # 平移时边缘补 0，避免 ImageChops.offset 的环绕伪影。
    return mask.transform(mask.size, Image.Transform.AFFINE, (1, 0, dx, 0, 1, dy))


def _dilate_once(mask, cross):
    if not cross:
        return mask.filter(ImageFilter.MaxFilter(3))
    grown = mask
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        grown = ImageChops.lighter(grown, _shifted(mask, dx, dy))
    return grown


def grow_mask(mask, amount):
    """按像素扩张（正）或收缩（负）选区；方形与十字交替，生长形状接近圆形。"""
    if not amount:
        return mask
    source = mask if amount > 0 else ImageOps.invert(mask)
    for step in range(abs(int(amount))):
        source = _dilate_once(source, cross=bool(step % 2))
    return source if amount > 0 else ImageOps.invert(source)


def prepare_mask(data, expected_size, *, allow_empty=False, output_size=None, expand_auto_edges=False,
                 grow=0, feather=0):
    if len(data) > MAX_MASK_BYTES:
        raise ValueError("遮罩超过 8 MiB")
    if not (isinstance(grow, int) and not isinstance(grow, bool) and -MASK_GROW_LIMIT <= grow <= MASK_GROW_LIMIT
            and isinstance(feather, int) and not isinstance(feather, bool)
            and 0 <= feather <= MASK_FEATHER_LIMIT):
        raise ValueError("遮罩扩张或羽化参数超出范围")
    with Image.open(BytesIO(data)) as src:
        if src.format != "PNG" or src.size != expected_size or getattr(src, "is_animated", False):
            raise ValueError("遮罩必须为与编辑原图尺寸一致的静态 PNG")
        mask = src.convert("L")
        if not mask.getbbox():
            if allow_empty:
                return None
            raise ValueError("没有检测到选区，请涂抹或更换自动选择目标")
        if expand_auto_edges:
            # 先对自动遮罩做形态学闭运算（先膨胀再等幅腐蚀），自动封死内部细小裂缝与碎洞
            close_radius = max(2, min(8, round(min(mask.size) * 0.006)))
            mask = grow_mask(grow_mask(mask, close_radius), -close_radius)
            radius = max(1, min(6, round(min(mask.size) * 0.004)))
            mask = mask.filter(ImageFilter.MaxFilter(2 * radius + 1))
        if grow:
            mask = grow_mask(mask, grow)
            if not mask.getbbox():
                if allow_empty:
                    return None
                raise ValueError("收缩后选区已消失，请减小收缩量")
        if feather:
            mask = mask.filter(ImageFilter.GaussianBlur(feather))
        if output_size is not None and mask.size != output_size:
            mask = mask.resize(output_size, Image.Resampling.BILINEAR)
        return _png(mask)


class MaskedWorkflow(VisualLoRAControls):
    def __init__(self, api, template, aliases):
        self.api = api
        self.template = template
        self.lora_aliases = self._load_lora_aliases(aliases)
        branches = set(ALTERNATE_ROLES).intersection(template.bindings)
        if branches and branches != set(ALTERNATE_ROLES):
            raise ValueError("移除和整图重绘缺少所需的节点绑定")
        self.has_alternate_modes = bool(branches)
        encoder = template.workflow[template.node_id("positive")]
        self.accepts_second_image = (
            encoder["class_type"] == "TextEncodeQwenImage21"
            and "images.image_1" in encoder["inputs"]
            and "images.image_2" not in encoder["inputs"]
        )
        self.MAX_REFERENCE_IMAGES = 2 if self.accepts_second_image else 1

    @staticmethod
    def _output_branch(workflow, output):
        branch = {}
        pending = [output]
        while pending:
            node_id = pending.pop()
            if node_id in branch:
                continue
            node = branch[node_id] = workflow[node_id]
            pending.extend(
                value[0] for value in node["inputs"].values()
                if isinstance(value, list) and len(value) == 2
                and isinstance(value[0], str) and value[0] in workflow
                and isinstance(value[1], int)
            )
        return branch

    def lora_catalog(self):
        result = {}
        for section, role in LORA_SECTIONS.items():
            if role not in self.template.bindings:
                continue
            branch = self._output_branch(self.template.workflow, self.template.node_id(role))
            records = {}
            for _, rows in _visual_lora_rows(branch):
                for row in rows:
                    name = _lora_path(row)
                    if name and name not in records:
                        strengths = {}
                        for key in LORA_STRENGTHS:
                            value = row.get(key, 1)
                            strengths[key] = value if type(value) in (int, float) and -10 <= value <= 10 else 1
                        records[name] = {"name": name, "enabled": bool(row.get("enabled")), **strengths}
            if records:
                result[section] = list(records.values())
        return result

    @staticmethod
    def _apply_branch_lora_defaults(workflow, rows):
        if not rows:
            return
        for inputs, entries in _visual_lora_rows(workflow):
            for entry in entries:
                override = rows.get(_lora_path(entry))
                if override is not None:
                    entry.update(override)
            inputs["lora_data"] = json.dumps(entries, ensure_ascii=False)

    async def generate(self, image, mask, prompt, *, on_submitted_callback=None, lora_selection=None,
                       lora_defaults=None, reference_image=None, image_extension="png",
                       on_result_reference=None, on_progress=None):
        if not (mask or prompt) or (not self.has_alternate_modes and not (mask and prompt)):
            raise ValueError("当前工作流缺少所需选区或修改要求")
        if reference_image is not None and (not self.accepts_second_image or not prompt.strip()):
            raise ValueError("使用参考图 2 需要支持双图的重绘流程及修改要求")
        if image_extension == "jpg":
            image = await asyncio.to_thread(_primary_photo_bytes, image)
        workflow = self.template.copy()
        output_role = "output"
        if not prompt:
            output_role = "mask_remove_output"
        elif mask is None:
            output_role = "mask_full_output"
            self.template.set(workflow, "mask_full_reference", [self.template.node_id("image"), 0])
        output = self.template.node_id(output_role)
        if self.has_alternate_modes:
            workflow = self._output_branch(workflow, output)
        if lora_defaults:
            section = "repaint" if prompt else "remove"
            self._apply_branch_lora_defaults(workflow, lora_defaults.get(section))
        if prompt:
            self.template.set(workflow, "positive", prompt)
            self._apply_lora_selection(workflow, lora_selection)

        for role, data in (("image", image), ("mask", mask)):
            if data is None:
                continue
            extension = image_extension if role == "image" else "png"
            filename = await self.api.upload_image(f"hub_inpaint.{extension}", data)
            if not filename:
                raise ValueError("局部重绘输入上传失败")
            self.template.set(workflow, role, filename)

        if reference_image is not None:
            filename = await self.api.upload_image("hub_inpaint_reference.png", reference_image)
            if not filename:
                raise ValueError("参考图 2 上传到 ComfyUI 失败")
            node_id = "hub_image_2"
            while node_id in workflow:
                node_id += "_"
            workflow[node_id] = {"class_type": "LoadImage", "inputs": {"image": filename}}
            encoder_inputs = workflow[self.template.node_id("positive")]["inputs"]
            encoder_inputs["images.image_2"] = [node_id, 0]

        randomize_seeds(workflow)
        return await self.api.queue_and_wait_image(
            workflow, max_wait=self.api.timeout, on_submitted_callback=on_submitted_callback,
            output_node=output, on_result_reference=on_result_reference,
            on_progress=on_progress,
        )


async def remove_background(api, image, *, image_extension="png", on_submitted_callback=None,
                            on_result_reference=None):
    """使用 ComfyUI 的 BiRefNet 节点对单张图片抠图（输出带透明通道的 PNG）。"""
    if image_extension == "jpg":
        image = await asyncio.to_thread(_primary_photo_bytes, image)
    filename = await api.upload_image(f"hub_rmbg.{image_extension}", image)
    if not filename:
        raise ValueError("抠图原图上传失败")
    graph = {
        "image": {"class_type": "LoadImage", "inputs": {"image": filename}},
        "model": {
            "class_type": "LoadRembgByBiRefNetModel",
            "inputs": {"model": "General.safetensors", "device": "AUTO", "use_weight": False, "dtype": "float32"},
        },
        "rembg": {
            "class_type": "RembgByBiRefNet",
            "inputs": {"model": ["model", 0], "images": ["image", 0]},
        },
        "output": {
            "class_type": "SaveImage",
            "inputs": {"images": ["rembg", 0], "filename_prefix": "hub_rmbg"},
        },
    }
    return await api.queue_and_wait_image(
        graph, max_wait=min(120, api.timeout),
        on_submitted_callback=on_submitted_callback, output_node="output",
        on_result_reference=on_result_reference,
    )


def plan_tiled_upscale(image_bytes: bytes, *, scale_factor: float = None, target_mp: float = None) -> tuple:
    """根据原图像素与用户可选参数规划分块放大方案。

    Returns:
        (target_mp, fast_turbo, source_mp, reject_reason)
        - 小图 (< 2 MP)：默认放大到 4 MP，关闭 8 步加速 LoRA（25 步精细）
        - 中图 (2 MP <= 原图 < 12 MP)：默认按原图面积 ×2.25（边长 ×1.5），向上取整到 0.5 MP，最高封顶 12 MP，关闭 8 步加速 LoRA
        - 大图 (>= 12 MP)：未指定倍数时拒绝并提示使用“放大 2x”；指定倍数强制放大时开启 8 步加速 LoRA
    """
    with Image.open(BytesIO(image_bytes)) as img:
        w, h = img.size
    if w <= 0 or h <= 0:
        raise ValueError("无法识别原图尺寸")
    source_mp = (w * h) / 1e6
    is_large = source_mp >= 12.0

    if scale_factor is not None:
        side_scale = float(scale_factor)
        if not (1.0 < side_scale <= 4.0):
            return None, False, source_mp, "放大倍数请在 1.1x 至 4x 之间，例如“放大 2x”。"
        computed_mp = math.ceil(source_mp * (side_scale ** 2) * 2.0) / 2.0
        final_mp = max(0.5, min(64.0, computed_mp))
        fast_turbo = is_large
        return final_mp, fast_turbo, source_mp, None

    if target_mp is not None:
        asked = float(target_mp)
        if is_large:
            return (
                None, False, source_mp,
                "原始图片≥ 12 MP，无需放大，如需强制放大请使用“放大 2x”指定放大倍数",
            )
        final_mp = max(0.5, min(16.0, asked))
        return final_mp, False, source_mp, None

    if is_large:
        return (
            None, False, source_mp,
            "原始图片≥ 12 MP，无需放大，如需强制放大请使用“放大 2x”指定放大倍数",
        )
    if source_mp < 2.0:
        final_mp = 4.0
    else:
        final_mp = min(12.0, math.ceil(source_mp * 2.25 * 2.0) / 2.0)
    return final_mp, False, source_mp, None


async def tiled_upscale_image(api, image, *, megapixels: float = 4.0, fast: bool = False,
                              image_extension="png", on_submitted_callback=None,
                              on_result_reference=None, on_progress=None, can_submit=None):
    """使用内置的 AusBoss Qwen Image 2.1 Tiled Upscale 工作流对单张图片分块重绘高清放大。"""
    if image_extension == "jpg":
        image = await asyncio.to_thread(_primary_photo_bytes, image)
    workflow_path = Path(__file__).parent / "workflows" / "example_qwen_upscale.json"
    template = WorkflowTemplate(str(workflow_path), ("image", "output"))
    workflow = template.copy()
    filename = await api.upload_image(f"hub_upscale.{image_extension}", image)
    if can_submit is not None and not can_submit():
        return None
    if not filename:
        raise ValueError("放大原图上传失败")
    template.set(workflow, "image", filename)
    if megapixels is not None and "megapixels" in template.bindings:
        mp = max(0.5, min(64.0, float(megapixels)))
        template.set(workflow, "megapixels", mp)
    if "steps" in template.bindings:
        template.set(workflow, "steps", 8 if fast else 25)
    if "loras" in template.bindings:
        raw_loras = template.get(workflow, "loras")
        try:
            rows = json.loads(raw_loras) if isinstance(raw_loras, str) else []
            for row in rows:
                if isinstance(row, dict) and "viggle-turbo" in str(row.get("name", "")).casefold():
                    row["enabled"] = bool(fast)
            template.set(workflow, "loras", json.dumps(rows, ensure_ascii=False))
        except ValueError:
            pass
    output_node = template.node_id("output")
    randomize_seeds(workflow)
    return await api.queue_and_wait_image(
        workflow, max_wait=api.timeout,
        on_submitted_callback=on_submitted_callback,
        output_node=output_node,
        on_result_reference=on_result_reference,
        on_progress=on_progress,
    )


async def auto_mask(api, image, target, *, image_extension="png"):
    filename = await api.upload_image(f"hub_sam3.{image_extension}", image)
    if not filename:
        raise ValueError("自动遮罩原图上传失败")
    graph = {
        "image": {"class_type": "LoadImage", "inputs": {"image": filename}},
        "model": {"class_type": "LoadSAM3Model", "inputs": {"precision": "auto", "compile": False}},
        "mask": {"class_type": "SAM3Grounding", "inputs": {
            "sam3_model_config": ["model", 0], "image": ["image", 0],
            "confidence_threshold": 0.2, "text_prompt": target, "max_detections": -1,
        }},
        "combined": {"class_type": "Masks Combine Batch", "inputs": {"masks": ["mask", 0]}},
        "preview": {"class_type": "MaskToImage", "inputs": {"mask": ["combined", 0]}},
        "output": {"class_type": "SaveImage", "inputs": {
            "images": ["preview", 0], "filename_prefix": "hub_auto_mask",
        }},
    }
    return await api.queue_and_wait_image(graph, max_wait=min(60, api.timeout), output_node="output")
