"""Limited, single-task mask editor served separately from the AstrBot dashboard."""

import asyncio
import json
import re
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from aiohttp import web
from PIL import Image
from astrbot.api import logger
from .comfyui_api import ComfyUIExecutionError
from .hub_common import GENERATED_MEDIA_FILENAME
from .mask_original_cache import MaskOriginalCache, SHA256_PATTERN

from .masked_workflow import (
    MAX_MASK_BYTES, MAX_MASK_SOURCE_BYTES, MAX_REFERENCE_BYTES,
    MaskOriginal, auto_mask, prepare_edit_image, prepare_mask, prepare_mask_original,
)


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{43}")
ENGLISH_TARGET = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ,.'/-]{0,119}")
EDITOR_TTL = 15 * 60
ORIGINAL_CHUNK_BYTES = 1024 * 1024
ORIGINAL_CHUNK_PARALLELISM = 2
UPLOAD_ID = re.compile(r"[0-9a-f]{32}")


@dataclass
class OriginalUpload:
    id: str
    size: int
    chunks: dict[int, bytes] = field(default_factory=dict, repr=False)
    inflight: set[int] = field(default_factory=set, repr=False)
    received: int = 0
    received_total: int = 0
    status: str = "receiving"
    error: str = ""

    def state(self):
        return {
            "id": self.id, "received": self.received, "received_total": self.received_total,
            "status": self.status, "error": self.error,
        }

    def acknowledge(self, offset):
        return web.json_response({**self.state(), "acknowledged_offset": offset})

    def record(self, offset, chunk):
        self.chunks[offset] = chunk
        self.received_total += len(chunk)
        while self.received in self.chunks:
            self.received += len(self.chunks[self.received])


@dataclass
class AutoMaskJob:
    id: str = field(default_factory=lambda: secrets.token_hex(8))
    status: str = "queued"
    ahead: int = 0
    ticket: object = field(default=None, repr=False)
    mask: bytes | None = field(default=None, repr=False)
    error: str = ""


@dataclass
class MaskDraft:
    owner: tuple
    original: MaskOriginal | None
    prompt: str
    submit: object
    alternate_modes: bool = False
    accepts_second_image: bool = False
    translate_target: object = None
    check_image: object = None
    upload: OriginalUpload | None = None
    previous_original: MaskOriginal | None = field(default=None, repr=False)
    original_revision: int = 0
    auto_job: AutoMaskJob | None = None
    expires: float = field(default_factory=lambda: time.monotonic() + EDITOR_TTL)
    busy: bool = False


class MaskEditor:
    def __init__(self, api, assets: Path, public_url: str, port: int, *, cache_directory=None, generation_queue=None):
        self.api = api
        self.assets = assets
        self.url = public_url
        self.port = port
        self.original_cache = MaskOriginalCache(cache_directory) if cache_directory is not None else None
        self.generation_queue = generation_queue
        self.runner = None
        self.available = False
        self.drafts = {}
        self.result_downloads: dict[str, tuple[Path, float]] = {}
        self.background = set()
        self.segmentation = asyncio.Lock()
        self.sweep = None

    async def start(self):
        if not self.url:
            return
        parsed = urlsplit(self.url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
            raise ValueError("涂抹页地址必须是 HTTPS 网站根地址")
        app = web.Application(client_max_size=MAX_MASK_BYTES + MAX_REFERENCE_BYTES + 16386,
                              middlewares=[self._headers])
        app.router.add_get("/", self._index)
        app.router.add_get("/editor.js", self._script)
        app.router.add_get("/api/draft", self._state)
        app.router.add_get("/api/image", self._image)
        app.router.add_get("/result/{token}", self._download_result)
        app.router.add_post("/api/original", self._upload_original)
        app.router.add_post("/api/original/replace", self._replace_original)
        app.router.add_post("/api/original/restore", self._restore_original)
        app.router.add_post("/api/original/cache", self._reuse_original)
        app.router.add_post("/api/original/chunk", self._upload_original_chunk)
        app.router.add_post("/api/auto", self._auto)
        app.router.add_get("/api/auto", self._auto_status)
        app.router.add_post("/api/submit", self._submit)
        runner = web.AppRunner(app, access_log=None, shutdown_timeout=3)
        await runner.setup()
        try:
            await web.TCPSite(runner, "0.0.0.0", self.port).start()
        except BaseException:
            await runner.cleanup()
            raise
        self.runner = runner
        self.available = True
        self.sweep = asyncio.create_task(self._cleanup_loop())

    async def close(self):
        self.available = False
        if self.runner is not None:
            await self.runner.cleanup()
        tasks = list(self.background)
        if self.sweep:
            tasks.append(self.sweep)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.drafts.clear()
        self.result_downloads.clear()

    @web.middleware
    async def _headers(self, request, handler):
        try:
            response = await handler(request)
        except web.HTTPException as error:
            response = error
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' blob:; style-src 'self' 'unsafe-inline'; "
            "frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
        )
        return response

    def share_result(self, path: Path, temp_dir: Path) -> str:
        if not self.available:
            raise ValueError("HTTPS 下载入口未启用")
        path = path.resolve(strict=True)
        if path.parent != temp_dir.resolve() or not GENERATED_MEDIA_FILENAME.fullmatch(path.name) or not path.is_file():
            raise ValueError("下载入口只支持已生成的图片或视频")
        self._expire()
        if len(self.result_downloads) >= 64:
            raise ValueError("下载链接数量达到上限，请稍后再试")
        token = secrets.token_urlsafe(32)
        self.result_downloads[token] = (path, time.monotonic() + EDITOR_TTL)
        return f"{self.url.rstrip('/')}/result/{token}"

    async def _download_result(self, request):
        token = request.match_info["token"]
        item = self.result_downloads.get(token)
        if item is None or item[1] <= time.monotonic() or not item[0].is_file():
            raise web.HTTPNotFound()
        path = item[0]
        return web.FileResponse(
            path, headers={"Content-Disposition": f'attachment; filename="ComfyUI{path.suffix}"'},
        )

    def _expire(self):
        now = time.monotonic()
        for token, (_, expires) in list(self.result_downloads.items()):
            if expires <= now:
                self.result_downloads.pop(token, None)
        for token, draft in list(self.drafts.items()):
            if draft.expires <= now and not draft.busy:
                self.drafts.pop(token, None)

    async def _cleanup_loop(self):
        while True:
            await asyncio.sleep(60)
            self._expire()
            if self.original_cache is not None:
                try:
                    await asyncio.to_thread(self.original_cache.prune)
                except OSError as error:
                    logger.warning("[ComfyUI] 无法清理原图缓存：%s", type(error).__name__)

    async def create(self, owner, data, prompt, submit, *, alternate_modes=False,
                     accepts_second_image=False, translate_target=None, check_image=None):
        if not self.available:
            raise ValueError("网页涂抹页尚未配置，请联系管理员")
        original = await self._prepare_original(data, check_image) if data is not None else None
        self._expire()
        if any(draft.owner == owner for draft in self.drafts.values()):
            raise ValueError("你已有一张待编辑的图片，请先提交或等链接过期")
        if len(self.drafts) + len(self.background) >= 8:
            raise ValueError("局部重绘任务已满，请稍后再试")
        token = secrets.token_urlsafe(32)
        self.drafts[token] = MaskDraft(
            owner, original, prompt, submit, alternate_modes=alternate_modes,
            accepts_second_image=accepts_second_image, translate_target=translate_target,
            check_image=check_image,
            original_revision=1 if original else 0,
        )
        return token, self.url.rstrip("/") + "/#" + token

    def revoke(self, token):
        self.drafts.pop(token, None)

    def _get(self, request):
        token = request.headers.get("Authorization", "").removeprefix("Bearer ")
        draft = self.drafts.get(token) if TOKEN_PATTERN.fullmatch(token) else None
        if not draft or draft.expires <= time.monotonic():
            raise web.HTTPNotFound(text="编辑链接无效、已提交或已过期")
        return token, draft

    async def _index(self, _):
        # FileResponse can introduce validators; serve the small static editor directly.
        return web.Response(body=(self.assets / "index.html").read_bytes(), content_type="text/html", charset="utf-8")

    async def _script(self, _):
        return web.Response(body=(self.assets / "editor.js").read_bytes(), content_type="text/javascript", charset="utf-8")

    async def _state(self, request):
        _, draft = self._get(request)
        return web.json_response(self._draft_state(draft))

    def _draft_state(self, draft):
        original = draft.original
        width, height = original.preview_size if original else (0, 0)
        return {
            "width": width, "height": height, "original_size": original.size if original else None,
            "prompt": draft.prompt, "needs_original": original is None,
            "original_revision": draft.original_revision,
            "can_replace_original": original is not None,
            "can_restore_original": draft.previous_original is not None,
            "auto_job": ({"id": draft.auto_job.id, "status": draft.auto_job.status}
                         if draft.auto_job else None),
            "max_original_bytes": MAX_MASK_SOURCE_BYTES,
            "original_chunk_bytes": ORIGINAL_CHUNK_BYTES,
            "original_chunk_parallelism": ORIGINAL_CHUNK_PARALLELISM,
            "original_cache_enabled": self.original_cache is not None,
            "original_upload": draft.upload.state() if draft.upload else None,
            "alternate_modes": draft.alternate_modes,
            "max_reference_bytes": MAX_REFERENCE_BYTES if draft.accepts_second_image else 0,
        }

    async def _image(self, request):
        _, draft = self._get(request)
        original = self._require_original(request, draft)
        kind = "image/png" if original.preview_extension == "png" else "image/jpeg"
        return web.Response(body=original.preview, content_type=kind)

    @staticmethod
    def _require_original(request, draft):
        if draft.original is None:
            raise web.HTTPConflict(text="请先上传原图。")
        MaskEditor._check_original_revision(request, draft)
        return draft.original

    @staticmethod
    def _check_original_revision(request, draft):
        revision = request.headers.get("X-Original-Revision")
        if revision != str(draft.original_revision) and (revision is not None or draft.original_revision > 1):
            raise web.HTTPConflict(text="原图已更换，请刷新页面后再继续编辑。")

    async def _replace_original(self, request):
        _, draft = self._get(request)
        if draft.busy:
            raise web.HTTPConflict(text="原图或遮罩正在处理，请稍后再试。")
        if draft.previous_original is None:
            draft.previous_original = self._require_original(request, draft)
            draft.original = None
            draft.upload = None
        return web.json_response(self._draft_state(draft))

    async def _restore_original(self, request):
        _, draft = self._get(request)
        if draft.busy:
            raise web.HTTPConflict(text="新图还在处理，请稍后再试。")
        self._check_original_revision(request, draft)
        if draft.previous_original is None:
            if draft.original is None:
                raise web.HTTPConflict(text="当前没有可恢复的原图。")
            return web.json_response(self._draft_state(draft))
        draft.original = draft.previous_original
        draft.previous_original = None
        draft.upload = None
        return web.json_response(self._draft_state(draft))

    @staticmethod
    def _commit_original(draft, original):
        draft.original = original
        draft.previous_original = None
        draft.auto_job = None
        draft.original_revision += 1

    @staticmethod
    async def _prepare_original(data, check_image):
        original = await asyncio.to_thread(prepare_mask_original, data)
        await MaskEditor._check_original(original, check_image)
        return original

    @staticmethod
    async def _check_original(original, check_image):
        if check_image:
            safe, message = await check_image(original.preview)
            if not safe:
                raise ValueError(message or "原图未通过检查")

    async def _cache_original(self, owner, original):
        if self.original_cache is not None:
            try:
                await asyncio.to_thread(self.original_cache.put, owner, original)
            except OSError as error:
                logger.warning("[ComfyUI] 无法保存原图缓存：%s", type(error).__name__)

    async def _reuse_original(self, request):
        token, draft = self._get(request)
        if self.original_cache is None:
            return web.json_response({"hit": False})
        if draft.original is not None or draft.busy:
            raise web.HTTPConflict(text="原图已上传或正在处理。")
        try:
            limited = request.clone(client_max_size=512)
            digest = await asyncio.wait_for(limited.text(), timeout=8)
            if not SHA256_PATTERN.fullmatch(digest):
                raise ValueError("SHA-256 格式无效")
        except (ValueError, asyncio.TimeoutError) as error:
            raise web.HTTPBadRequest(text="原图缓存查询参数无效。") from error
        if (draft.original is not None or draft.busy or self.drafts.get(token) is not draft
                or draft.expires <= time.monotonic()):
            raise web.HTTPConflict(text="原图已上传或正在处理。")
        draft.busy = True
        try:
            original = await asyncio.to_thread(self.original_cache.get, draft.owner, digest)
            if original is None:
                return web.json_response({"hit": False})
            await self._check_original(original, draft.check_image)
            if self.drafts.get(token) is not draft or draft.expires <= time.monotonic():
                raise web.HTTPNotFound(text="编辑链接无效、已提交或已过期")
            self._commit_original(draft, original)
            return web.json_response({"hit": True, **self._draft_state(draft)})
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        finally:
            draft.busy = False

    async def _upload_original(self, request):
        token, draft = self._get(request)
        if draft.original is not None or draft.busy:
            raise web.HTTPConflict(text="原图已上传或正在处理。")
        draft.busy = True
        try:
            limited = request.clone(client_max_size=MAX_MASK_SOURCE_BYTES + 1)
            data = await asyncio.wait_for(limited.read(), timeout=120)
            original = await self._prepare_original(data, draft.check_image)
            await self._cache_original(draft.owner, original)
            if self.drafts.get(token) is not draft or draft.expires <= time.monotonic():
                raise web.HTTPNotFound(text="编辑链接无效、已提交或已过期")
            self._commit_original(draft, original)
            return await self._state(request)
        except web.HTTPRequestEntityTooLarge as error:
            raise web.HTTPBadRequest(text="原图最大 48 MiB。") from error
        except asyncio.TimeoutError as error:
            raise web.HTTPBadRequest(text="原图上传超时，请重试。") from error
        except (OSError, Image.DecompressionBombError) as error:
            raise web.HTTPBadRequest(text="无法读取图片，请上传 JPG、PNG、WebP、BMP 或 TIFF。") from error
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        finally:
            draft.busy = False

    async def _upload_original_chunk(self, request):
        token, draft = self._get(request)
        upload_id = request.headers.get("X-Original-Upload", "")
        try:
            offset = int(request.headers.get("X-Original-Offset", ""))
            total = int(request.headers.get("X-Original-Size", ""))
        except ValueError as error:
            raise web.HTTPBadRequest(text="原图分段参数无效。") from error
        if (not UPLOAD_ID.fullmatch(upload_id) or not 0 <= offset < total <= MAX_MASK_SOURCE_BYTES
                or offset % ORIGINAL_CHUNK_BYTES):
            raise web.HTTPBadRequest(text="原图分段参数无效，文件最大 48 MiB。")

        upload = draft.upload
        if upload is not None and upload.id == upload_id:
            if upload.size != total:
                raise web.HTTPBadRequest(text="原图大小已变化，请重新选择文件。")
            if offset < upload.received or offset in upload.chunks:
                return upload.acknowledge(offset)
            if draft.original is not None or (draft.busy and not upload.inflight):
                raise web.HTTPConflict(text="原图已上传或正在处理。")
        else:
            if draft.original is not None or draft.busy:
                raise web.HTTPConflict(text="原图正在接收、已上传或正在处理。")
            # The second worker can reach the server before the first one.
            if offset and (upload is not None or offset != ORIGINAL_CHUNK_BYTES):
                raise web.HTTPConflict(text="原图上传已被替换，请重新选择文件。")
            draft.upload = upload = OriginalUpload(upload_id, total)
        if offset in upload.inflight or len(upload.inflight) >= ORIGINAL_CHUNK_PARALLELISM:
            raise web.HTTPConflict(text="原图分段正在接收，请稍后重试。")
        length = min(ORIGINAL_CHUNK_BYTES, total - offset)
        if request.content_length is not None and request.content_length != length:
            raise web.HTTPBadRequest(text="原图分段长度不正确。")
        upload.inflight.add(offset)
        draft.busy = True
        try:
            chunk = await asyncio.wait_for(request.clone(client_max_size=length + 1).read(), timeout=25)
            if len(chunk) != length:
                raise web.HTTPBadRequest(text="原图分段长度不正确。")
            if self.drafts.get(token) is not draft or draft.expires <= time.monotonic():
                raise web.HTTPNotFound(text="编辑链接无效、已提交或已过期")
            upload.record(offset, chunk)
        except asyncio.TimeoutError as error:
            logger.warning("[ComfyUI] 原图分段接收超时：已收 %s/%s 字节", upload.received_total, total)
            raise web.HTTPRequestTimeout(text="群晖接收原图分段超时。") from error
        finally:
            upload.inflight.discard(offset)
            draft.busy = bool(upload.inflight)

        if upload.received == total:
            upload.status = "processing"
            draft.busy = True
            task = asyncio.create_task(self._finish_original_upload(token, draft, upload))
            self.background.add(task)
            task.add_done_callback(self.background.discard)
        return upload.acknowledge(offset)

    async def _finish_original_upload(self, token, draft, upload):
        try:
            data = b"".join(upload.chunks[offset] for offset in range(0, upload.size, ORIGINAL_CHUNK_BYTES))
            upload.chunks.clear()
            original = await asyncio.wait_for(self._prepare_original(data, draft.check_image), timeout=75)
            await self._cache_original(draft.owner, original)
            if self.drafts.get(token) is draft and draft.expires > time.monotonic():
                self._commit_original(draft, original)
                upload.status = "ready"
        except asyncio.TimeoutError:
            upload.error = "原图已收到，但生成预览超时，请重试。"
        except ValueError as error:
            upload.error = str(error)
        except Exception as error:
            logger.warning("[ComfyUI] 原图预览生成失败：%s", type(error).__name__)
            upload.error = "无法读取原图，请确认文件是完整的 JPG、PNG、WebP、BMP 或 TIFF。"
        finally:
            if upload.error:
                upload.status = "failed"
            draft.busy = False

    def _auto_job_payload(self, job: AutoMaskJob) -> dict:
        ahead = job.ahead
        if job.status == "queued" and self.generation_queue is not None and job.ticket is not None:
            live = self.generation_queue.ahead(job.ticket)
            if live is not None:
                ahead = live
        return {"id": job.id, "status": job.status, "ahead": int(ahead or 0)}

    async def _auto(self, request):
        token, draft = self._get(request)
        original = self._require_original(request, draft)
        if draft.busy:
            raise web.HTTPConflict(text="自动遮罩正在处理，请稍后重试")
        try:
            target = (await asyncio.wait_for(request.text(), timeout=10)).strip()
        except (UnicodeError, asyncio.TimeoutError) as error:
            raise web.HTTPBadRequest(text="识别目标格式不正确") from error
        # Requests may overlap while the body is being read.
        if self.drafts.get(token) is not draft or draft.expires <= time.monotonic() or draft.busy:
            raise web.HTTPConflict(text="当前编辑请求已失效或正在处理中")
        if draft.original is not original:
            raise web.HTTPConflict(text="原图已更换，请刷新页面后再继续编辑。")
        if not 1 <= len(target) <= 120:
            raise web.HTTPBadRequest(text="识别目标需要 1 至 120 个字符")
        draft.busy = True
        draft.auto_job = job = AutoMaskJob()
        if self.generation_queue is not None:
            ticket, initial_ahead = await self.generation_queue.enqueue(
                draft.owner, "自动遮罩", model_group="sam3"
            )
            job.ticket = ticket
            job.ahead = initial_ahead
        task = asyncio.create_task(self._generate_auto_mask(token, draft, original, target, job))
        self.background.add(task)
        task.add_done_callback(self.background.discard)
        return web.json_response(self._auto_job_payload(job), status=202)

    async def _auto_status(self, request):
        _, draft = self._get(request)
        self._require_original(request, draft)
        job = draft.auto_job
        if job is None or request.query.get("job") != job.id:
            raise web.HTTPNotFound(text="自动遮罩任务不存在，请重新识别。")
        if job.status == "ready":
            return web.Response(body=job.mask, content_type="image/png")
        if job.status == "failed":
            raise web.HTTPUnprocessableEntity(text=job.error)
        return web.json_response(self._auto_job_payload(job), status=202)

    async def _generate_auto_mask(self, token, draft, original, target, job):
        ticket = job.ticket
        try:
            if self.generation_queue is not None and ticket is not None:
                if not await self.generation_queue.wait_turn(ticket):
                    raise ValueError("自动遮罩任务已取消")
            async with self.segmentation:
                if self.drafts.get(token) is not draft or draft.expires <= time.monotonic():
                    raise web.HTTPNotFound(text="编辑链接无效、已提交或已过期")
                job.status = "processing"
                job.ahead = 0
                target = await self._translate_auto_target(draft, target)
                mask = await auto_mask(
                    self.api, original.preview, target, image_extension=original.preview_extension,
                )
            if not mask:
                raise ValueError("没有找到目标")
            job.mask = await asyncio.to_thread(
                prepare_mask, mask, original.preview_size, expand_auto_edges=True,
            )
            if ticket is not None:
                ticket.generated = time.monotonic()
                ticket.finish("sent")
            job.status = "ready"
        except web.HTTPException as error:
            if ticket is not None and ticket.status in ("queued", "generating"):
                ticket.finish("failed")
            job.error = error.text
            job.status = "failed"
        except ComfyUIExecutionError as error:
            if ticket is not None and ticket.status in ("queued", "generating"):
                ticket.finish("failed")
            job.error = error.user_message
            job.status = "failed"
        except Exception as error:
            if ticket is not None and ticket.status in ("queued", "generating"):
                ticket.finish("failed")
            logger.warning("[ComfyUI] 自动遮罩失败：%s", type(error).__name__)
            job.error = "自动遮罩未取得有效结果。试试改用更具体的描述，或手动涂抹。"
            job.status = "failed"
        finally:
            if self.generation_queue is not None and ticket is not None:
                if ticket.status in ("queued", "generating"):
                    ticket.finish("failed")
                await self.generation_queue.release(ticket)
            draft.busy = False

    @staticmethod
    async def _translate_auto_target(draft, target):
        if any(not char.isascii() and char.isalpha() for char in target):
            try:
                translated = await asyncio.wait_for(draft.translate_target(target), timeout=15)
                phrase = translated.strip().strip("\"'`").strip()
                if not ENGLISH_TARGET.fullmatch(phrase):
                    raise ValueError("翻译结果无效")
                target = phrase
            except Exception as error:
                logger.warning("[ComfyUI] 自动遮罩目标翻译失败：%s", type(error).__name__)
                raise web.HTTPUnprocessableEntity(
                    text="目标翻译失败，请直接填写英文，或手动涂抹。"
                ) from error
        return target

    async def _submit(self, request):
        token, draft = self._get(request)
        original = self._require_original(request, draft)
        if draft.busy:
            raise web.HTTPConflict(text="当前任务正在处理中")
        reference = None
        grow = feather = 0
        try:
            body = await asyncio.wait_for(request.read(), timeout=20)
            encoded, separator, raw = body.partition(b"\n")
            if not separator or len(encoded) > 16384:
                raise ValueError("修改要求格式不正确")
            value = json.loads(encoded.decode("utf-8"))
            if isinstance(value, dict):
                grow = value.get("grow", 0)
                feather = value.get("feather", 0)
                mask_bytes = value.get("mask_bytes")
                if mask_bytes is not None:
                    if (not draft.accepts_second_image or type(mask_bytes) is not int
                            or not 0 < mask_bytes < len(raw) or mask_bytes > MAX_MASK_BYTES
                            or len(raw) - mask_bytes > MAX_REFERENCE_BYTES):
                        raise web.HTTPBadRequest(text="参考图 2 格式、大小或工作流不符合要求。")
                    raw, reference = raw[:mask_bytes], raw[mask_bytes:]
                value = value.get("prompt")
            if not isinstance(value, str):
                raise ValueError("修改要求格式不正确")
            prompt = value.strip()
            if not draft.alternate_modes:
                prompt = prompt or draft.prompt
            if len(prompt) > 4000:
                raise ValueError("修改要求最多 4000 个字符")
        except (ValueError, asyncio.TimeoutError) as error:
            raise web.HTTPBadRequest(text="修改要求缺失、太长或上传超时") from error
        if reference is not None and not prompt:
            raise web.HTTPBadRequest(text="上传参考图 2 后，请填写修改要求。")
        if self.drafts.get(token) is not draft or draft.busy or draft.expires <= time.monotonic():
            raise web.HTTPConflict(text="编辑任务已经提交、过期或正在处理中")
        if draft.original is not original:
            raise web.HTTPConflict(text="原图已更换，请刷新页面后再继续编辑。")
        draft.busy = True
        try:
            mask = await asyncio.to_thread(
                prepare_mask, raw, original.preview_size, allow_empty=draft.alternate_modes,
                output_size=original.size, grow=grow, feather=feather,
            )
            if reference is not None:
                try:
                    reference, _ = await asyncio.to_thread(
                        prepare_edit_image, reference, formats=("JPEG", "PNG", "WEBP"),
                    )
                except (OSError, ValueError, Image.DecompressionBombError) as error:
                    raise web.HTTPBadRequest(
                        text="参考图 2 须为 8 MiB 以内的静态 JPG、PNG 或 WebP，最多 1200 万像素。"
                    ) from error
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            raise web.HTTPBadRequest(text="请涂出有效选区。遮罩必须与预览图同尺寸且不超过 8 MiB。") from error
        finally:
            draft.busy = False
        if not (mask or prompt):
            raise web.HTTPBadRequest(text="请涂抹要移除的区域，或输入修改要求。")
        if not draft.alternate_modes and not prompt:
            raise web.HTTPBadRequest(text="请输入修改要求。")
        if self.drafts.get(token) is not draft or draft.expires <= time.monotonic():
            raise web.HTTPConflict(text="编辑任务已经过期")
        del self.drafts[token]
        task = asyncio.create_task(self._run(draft, mask, prompt, reference))
        self.background.add(task)
        task.add_done_callback(self.background.discard)
        return web.json_response({"accepted": True})

    async def _run(self, draft, mask, prompt, reference=None):
        try:
            args = (draft.original, mask, prompt)
            if reference is not None:
                await draft.submit(*args, reference)
            else:
                await draft.submit(*args)
        except Exception as error:
            logger.warning("[ComfyUI] 网页提交任务处理失败：%s", type(error).__name__)
