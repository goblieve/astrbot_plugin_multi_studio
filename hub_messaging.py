import aiohttp
import asyncio
import json
import re
import stat
import time
import uuid
from PIL import Image as PILImage
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.message_components import File as FileComponent
from astrbot.api.message_components import Image as ImageComponent
from astrbot.api.message_components import Reply
from io import BytesIO
from pathlib import Path
from typing import Optional
from typing import Tuple
from .draw_tasks import ImageTask
from .image_inputs import load_executor_images
from .llm_tools import IMAGE_COMPLETION_FEEDBACK_HINT
from .masked_workflow import MAX_MASK_SOURCE_BYTES
from astrbot.api.message_components import Video as VideoComponent
from .hub_common import COMFY_PROMPT_ID, GENERATED_MEDIA_FILENAME, GENERATED_MEDIA_TTL_SECONDS, IMAGE_RESEND_ID, INPUT_IMAGE_TIMEOUT, MASK_IMAGE_FILE_EXTENSIONS, MAX_INPUT_IMAGE_BYTES, PLATFORM_FILE_SIZE_LIMIT, QQ_IMAGE_FILE_THRESHOLD, RECENT_FILE_TTL, SIZE_LIMITED_PLATFORMS


class MessagingMixin:
    @staticmethod
    def _is_edit_image(component, allow_image_files=False):
        return isinstance(component, ImageComponent) or (
            allow_image_files and isinstance(component, FileComponent)
            and Path(component.name or "").suffix.lower() in MASK_IMAGE_FILE_EXTENSIONS
        )

    async def _recover_quoted_image(self, event):
        if event.get_platform_name() != "aiocqhttp":
            return None
        reply = next((part for part in event.get_messages() if isinstance(part, Reply)), None)
        raw = getattr(getattr(event, "message_obj", None), "raw_message", {})
        segments = raw.get("message", ()) if hasattr(raw, "get") else ()
        quote_id = reply.id if reply else next((
            segment.get("data", {}).get("id") for segment in (segments or ())
            if isinstance(segment, dict) and segment.get("type") == "reply"
            and isinstance(segment.get("data"), dict)
        ), None)
        if not quote_id or not str(quote_id).lstrip("-").isdecimal():
            logger.info("[ComfyUI] 引用恢复：未找到可用的引用消息 ID（Reply 组件=%s，原始段类型=%s）",
                        reply is not None,
                        [s.get("type") for s in (segments or ()) if isinstance(s, dict)])
            return None
        logger.info("[ComfyUI] 引用恢复：开始 get_msg，quote_id=%s", quote_id)
        try:
            routing = {"self_id": int(event.get_self_id())}
            result = await self._call_qq_action(event, "get_msg", message_id=int(quote_id), **routing)
            message = self._onebot_result_data(result)
            group = event.get_group_id()
            if message.get("group_id") and str(message["group_id"]) != str(group):
                return None
            segment_types = [s.get("type") for s in (message.get("message") or ()) if isinstance(s, dict)]
            logger.info("[ComfyUI] 引用消息恢复：get_msg 返回段类型=%s", segment_types)
            for segment in message.get("message") or ():
                if not isinstance(segment, dict) or not isinstance(segment.get("data"), dict):
                    continue
                data = segment["data"]
                url = data.get("url")
                if segment.get("type") == "file":
                    image = await self._download_qq_file(event, data)
                    if image:
                        return image
                    continue
                if segment.get("type") != "image":
                    continue
                if isinstance(url, str) and url.startswith(("http://", "https://")):
                    image = await self._get_image_data(FileComponent(name="quoted image", url=url),
                                                       max_bytes=MAX_MASK_SOURCE_BYTES)
                    if image:
                        return image
                    logger.warning("[ComfyUI] 引用图片下载失败（见上方错误），url 主机=%s",
                                   url.split("/")[2] if url.count("/") >= 2 else "?")
                else:
                    logger.warning("[ComfyUI] 未能获得引用图片的下载地址，段类型=%s", segment.get("type"))
        except Exception as error:
            logger.warning("[ComfyUI] 通过 QQ 接口恢复引用图片失败：%s: %s", type(error).__name__, error)
        return None

    async def _download_qq_file(self, event, data):
        """根据 OneBot file 段数据取得图片文件字节：自带 url → 文件地址接口 → NapCat get_file。"""
        name = data.get("file_name") or data.get("name") or data.get("file")
        if name and Path(str(name)).suffix.lower() not in MASK_IMAGE_FILE_EXTENSIONS:
            return None
        url = data.get("url") or data.get("_component_url")
        file_id = data.get("file_id")
        group = event.get_group_id()
        logger.info("[ComfyUI] QQ 文件：name=%s，file_id=%s，size=%s，自带url=%s",
                    name, file_id, data.get("file_size"), bool(url))
        try:
            routing = {"self_id": int(event.get_self_id())}
        except (TypeError, ValueError):
            routing = {}
        if not url and file_id:
            args = {"group_id": int(group)} if group else {}
            action = "get_group_file_url" if group else "get_private_file_url"
            try:
                file_result = await self._call_qq_action(
                    event, action, timeout=45, file_id=file_id, **args, **routing)
                url = self._onebot_result_data(file_result).get("url")
            except Exception as error:
                logger.warning("[ComfyUI] %s 失败：%s: %s", action, type(error).__name__, error)
            if not url:
                try:
                    file_result = await self._call_qq_action(
                        event, "get_file", timeout=120, file_id=file_id, **routing)
                    info = self._onebot_result_data(file_result)
                    url = info.get("url")
                    local = info.get("file")
                    readable = isinstance(local, str) and Path(local).is_file()
                    logger.info("[ComfyUI] get_file 返回：url=%s，file=%s，本地可读=%s", bool(url), local, readable)
                    if not url and readable and Path(local).stat().st_size <= MAX_MASK_SOURCE_BYTES:
                        return await asyncio.to_thread(Path(local).read_bytes)
                except Exception as error:
                    logger.warning("[ComfyUI] get_file 失败：%s: %s", type(error).__name__, error)
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            image = await self._get_image_data(FileComponent(name=str(name or "file"), url=url),
                                               max_bytes=MAX_MASK_SOURCE_BYTES)
            if not image:
                logger.warning("[ComfyUI] QQ 文件下载失败（见上方错误），url 主机=%s",
                               url.split("/")[2] if url.count("/") >= 2 else "?")
            return image
        logger.warning("[ComfyUI] 未能获得 QQ 文件的下载地址")
        return None

    def _remember_recent_file(self, event):
        """记录用户最近发送的图片文件（仅元数据），供不带引用的“局部重绘”使用。"""
        raw = getattr(getattr(event, "message_obj", None), "raw_message", None)
        segments = raw.get("message", ()) if hasattr(raw, "get") else ()
        if not isinstance(segments, (list, tuple)):
            return
        for segment in segments:
            if not isinstance(segment, dict) or segment.get("type") != "file" \
                    or not isinstance(segment.get("data"), dict):
                continue
            data = dict(segment["data"])
            name = data.get("file_name") or data.get("name") or data.get("file")
            if name and Path(str(name)).suffix.lower() not in MASK_IMAGE_FILE_EXTENSIONS:
                continue
            component = next((c for c in (event.get_messages() or []) if isinstance(c, FileComponent)), None)
            if component is not None and getattr(component, "url", None):
                data["_component_url"] = component.url
            store = self.__dict__.setdefault("_recent_files", {})
            now = time.monotonic()
            for key in [k for k, (stamp, _) in store.items() if now - stamp > RECENT_FILE_TTL]:
                store.pop(key, None)
            store[(str(event.get_group_id() or ""), str(event.get_sender_id()))] = (now, data)
            logger.info("[ComfyUI] 已记录最近发送的图片文件：%s", name)
            return

    def _remember_chat_image(self, event, messages):
        image = next((item for item in messages if isinstance(item, ImageComponent)), None)
        if image is None:
            return False
        store = self.__dict__.setdefault("_recent_chat_images", {})
        now = time.monotonic()
        for owner, (stamp, _) in list(store.items()):
            if now - stamp > RECENT_FILE_TTL:
                del store[owner]
        owner = self._image_task_owner(event)
        store.pop(owner, None)
        if len(store) >= 256:
            del store[next(iter(store))]
        store[owner] = (now, image)
        return True

    def _recent_chat_image_component(self, event):
        adapter = getattr(event, "adapter", None)
        get_path = getattr(adapter, "recent_image_path", None)
        if get_path is not None:
            path = get_path(event.session, event.get_sender_id())
            return ImageComponent.fromFileSystem(path) if path else None
        item = self.__dict__.get("_recent_chat_images", {}).get(self._image_task_owner(event))
        if item is not None and time.monotonic() - item[0] <= RECENT_FILE_TTL:
            return item[1]
        return None

    async def _recent_file_image(self, event, *, max_bytes=MAX_MASK_SOURCE_BYTES):
        if event.get_platform_name() == "synochat_adapter":
            component = self._recent_chat_image_component(event)
            if component is not None:
                return await self._get_image_data(component, max_bytes=max_bytes)
            return None
        store = self.__dict__.get("_recent_files") or {}
        item = store.get((str(event.get_group_id() or ""), str(event.get_sender_id())))
        if item is None or time.monotonic() - item[0] > RECENT_FILE_TTL:
            return None
        logger.info("[ComfyUI] 未发现引用，改用最近发送的图片文件")
        return await self._download_qq_file(event, item[1])

    def _recent_generated_image_bytes(self, event, *, max_bytes=MAX_MASK_SOURCE_BYTES) -> Optional[bytes]:
        """读取当前用户在当前会话最近一次成功生成的本地无损图片字节（用于“继续改”或免引用改图）。"""
        owner = self._image_task_owner(event)
        for task in self._image_generation_queue.recent(owner, limit=10):
            if task.path is not None and task.path.is_file():
                try:
                    if task.path.stat().st_size <= max_bytes:
                        logger.info("[ComfyUI] 复用上一张生成的本地图片：%s (task=%s)", task.path.name, task.id)
                        return task.path.read_bytes()
                except OSError:
                    continue
        return None

    @staticmethod
    def _onebot_result_data(result):
        if not isinstance(result, dict):
            return {}
        return result["data"] if isinstance(result.get("data"), dict) else result

    async def _send_text_message(self, event: AstrMessageEvent, message: str) -> Optional[str]:
        """发送文字；QQ 额外返回可用于引用和撤回的消息 ID。"""
        if event.get_platform_name() == "synochat_adapter":
            chain = event.plain_result(message)
            if event.pending_reply is not None and not event.pending_reply.done():
                await event.send(chain)
            else:
                await event.adapter.send_deferred_reply(event.session, chain)
            return None
        result = await self._call_send_api(event, message)
        return self._extract_and_record_message(result, event)

    async def _send_image_message(self, event: AstrMessageEvent, image_file: Path, chain: bool = False) -> Optional[str]:
        """发送图片消息并记录消息ID"""
        group_id = event.get_group_id()

        if group_id and chain:
            forward_msg = [{
                "type": "node",
                "data": {
                    "user_id": int(event.get_sender_id()),
                    "nickname": "ComfyUI",
                    "content": [{"type": "image", "data": {"file": f"file://{image_file}"}}],
                },
            }]
            result = await self._call_qq_action(
                event, "send_group_forward_msg", group_id=int(group_id), messages=forward_msg,
            )
            if self._qq_send_rejected(result):
                result = await self._call_send_api(event, f"[CQ:image,file=file://{image_file}]")
        else:
            result = await self._call_send_api(event, f"[CQ:image,file=file://{image_file}]")

        message_id = self._extract_and_record_message(result, event)
        if message_id is None:
            raise RuntimeError("QQ 未确认收到图片")
        return message_id

    async def _send_qq_image_file(self, event, path, task_id):
        group_id = event.get_group_id()
        action = "upload_group_file" if group_id else "upload_private_file"
        target = {"group_id": int(group_id)} if group_id else {"user_id": int(event.get_sender_id())}
        result = await self._call_qq_action(
            event, action, timeout=120, file=str(path.resolve()),
            name=f"ComfyUI_{task_id}{path.suffix}", **target,
        )
        # aiocqhttp raises on a nonzero retcode and may return null data on success.
        if result is False or self._qq_send_rejected(result):
            raise RuntimeError("QQ 未确认收到图片文件")

    async def _call_send_api(self, event: AstrMessageEvent, message):
        """调用发送消息 API"""
        if event.get_platform_name() != "aiocqhttp":
            return None

        group_id = event.get_group_id()
        user_id = event.get_sender_id()

        if group_id:
            return await self._call_qq_action(
                event,
                "send_group_msg",
                group_id=int(group_id),
                message=message,
            )
        if user_id:
            return await self._call_qq_action(
                event,
                "send_private_msg",
                user_id=int(user_id),
                message=message,
            )
        return None

    @staticmethod
    async def _call_qq_action(event, action, *, timeout=15, **kwargs):
        return await asyncio.wait_for(event.bot.api.call_action(action, **kwargs), timeout=timeout)

    @staticmethod
    def _qq_send_rejected(result):
        return isinstance(result, dict) and (
            result.get("status") == "failed" or result.get("retcode") not in (None, 0, "0")
        )

    def _extract_and_record_message(self, result, event: AstrMessageEvent) -> Optional[str]:
        """从API返回结果中提取并记录消息ID"""
        message_id = None

        if self._qq_send_rejected(result):
            return None
        if result:
            if isinstance(result, dict):
                data = result.get('data')
                if isinstance(data, dict):
                    message_id = data.get('message_id')
                elif data:
                    message_id = str(data)
                elif 'message_id' in result:
                    message_id = result['message_id']
            elif isinstance(result, (int, str)):
                message_id = str(result)

        if not re.fullmatch(r"-?[0-9]+", str(message_id)) or int(message_id) == 0:
            return None
        if message_id:
            key = str(event.get_group_id()) if event.get_group_id() else str(event.get_sender_id())
            self.sent_messages.setdefault(key, []).append({
                'message_id': str(message_id),
                'timestamp': time.time(),
                'user_id': str(event.get_sender_id()),
            })
            self._save_sent_messages()

        return message_id

    def _get_download_session(self) -> aiohttp.ClientSession:
        session = getattr(self, "_download_session", None)
        if session is None or session.closed:
            connector = aiohttp.TCPConnector(limit=16, ttl_dns_cache=300)
            session = aiohttp.ClientSession(connector=connector)
            self._download_session = session
        return session

    async def _close_download_session(self):
        session = getattr(self, "_download_session", None)
        if session is not None and not session.closed:
            await session.close()
        self._download_session = None

    @staticmethod
    def _read_local_file_capped(file_path: str, max_bytes: int) -> Optional[bytes]:
        path = Path(file_path)
        if not path.exists():
            return None
        with open(path, "rb") as f:
            data = f.read(max_bytes + 1)
        if len(data) > max_bytes:
            return None
        return data

    async def _get_image_data(self, image_component, *, max_bytes=MAX_INPUT_IMAGE_BYTES) -> Optional[bytes]:
        """从 Image 组件获取图片数据，复用下载连接池，带超时和大小限制"""
        try:
            if hasattr(image_component, 'url') and image_component.url:
                # 大文件（原图最大 48 MiB）需要更长的总时限；只要持续有数据到达就不判超时。
                total = INPUT_IMAGE_TIMEOUT if max_bytes <= MAX_INPUT_IMAGE_BYTES else 300
                timeout = aiohttp.ClientTimeout(total=total, sock_connect=15, sock_read=30)
                session = self._get_download_session()
                async with session.get(image_component.url, timeout=timeout) as resp:
                    if resp.status != 200:
                        return None
                    # 通过 Content-Length 提前拒绝
                    content_length = resp.headers.get("Content-Length")
                    if content_length and int(content_length) > max_bytes:
                        logger.error(
                            f"图片过大 ({content_length} 字节)，拒绝下载"
                        )
                        return None
                    # 按 chunk 累加，超过限制中断
                    buffer = bytearray()
                    async for chunk in resp.content.iter_chunked(64 * 1024):
                        buffer.extend(chunk)
                        if len(buffer) > max_bytes:
                            logger.error(
                                f"图片下载超过 {max_bytes} 字节限制，已中断"
                            )
                            return None
                    return bytes(buffer)
        except Exception as e:
            logger.error(f"通过URL获取图片失败: {e}")

        try:
            if hasattr(image_component, 'convert_to_file_path') and callable(image_component.convert_to_file_path):
                file_path = await image_component.convert_to_file_path()
                if file_path:
                    data = await asyncio.to_thread(self._read_local_file_capped, str(file_path), max_bytes)
                    if data is None and Path(file_path).exists():
                        logger.error("本地图片文件过大，拒绝读取")
                        return None
                    return data
        except Exception as e:
            logger.error(f"通过文件路径获取图片失败: {e}")

        return None

    def _maybe_compress_for_platform(self, image_data: bytes, platform_name: str) -> Tuple[Path, Optional[str]]:
        """对超大图片做平台限制下的压缩，返回 (临时文件路径, 警告信息或 None)"""
        temp_file = self.temp_dir / f"{uuid.uuid4().hex}.png"
        with open(temp_file, "wb") as f:
            f.write(image_data)

        if platform_name not in SIZE_LIMITED_PLATFORMS:
            return temp_file, None

        file_size = len(image_data)
        if file_size <= PLATFORM_FILE_SIZE_LIMIT:
            return temp_file, None

        size_mb = file_size / (1024 * 1024)
        logger.info(f"图片大小 {size_mb:.1f}MB 超过限制，尝试压缩...")

        try:
            img = PILImage.open(BytesIO(image_data))
        except Exception as e:
            logger.error(f"图片压缩失败: {e}")
            return temp_file, f"⚠️ 警告：生成的图片为 {size_mb:.1f}MB，超过平台默认 10MB 限制，压缩失败"

        # 依次尝试 WebP90 → AVIF85 → WebP 80/70/60/50
        attempts = [
            ('WEBP', 90, '.webp'),
            ('AVIF', 85, '.avif'),
            ('WEBP', 80, '.webp'),
            ('WEBP', 70, '.webp'),
            ('WEBP', 60, '.webp'),
            ('WEBP', 50, '.webp'),
        ]
        for fmt, quality, ext in attempts:
            try:
                buffer = BytesIO()
                img.save(buffer, format=fmt, quality=quality)
                if buffer.tell() <= PLATFORM_FILE_SIZE_LIMIT:
                    new_temp = temp_file.with_suffix(ext)
                    with open(new_temp, "wb") as f:
                        f.write(buffer.getvalue())
                    final_mb = buffer.tell() / (1024 * 1024)
                    logger.info(f"成功压缩为 {fmt}（quality={quality}），大小 {final_mb:.1f}MB")
                    return new_temp, None
            except Exception as e:
                logger.error(f"{fmt}(quality={quality}) 压缩失败: {e}")
                continue

        return temp_file, f"⚠️ 警告：原图 {size_mb:.1f}MB，压缩后仍超过 10MB 限制，可能无法发送"

    async def _collect_images_from_event(self, event: AstrMessageEvent, take_first_only: bool = False,
                                         image_indices=None, image_paths=None, *, allow_image_files=False,
                                         use_recent_chat_image=False) -> list:
        """读取消息图片或代码执行器产物，序号从 1 开始。"""
        if image_paths and not image_indices:
            return await load_executor_images(self.context, event, image_paths, self.temp_dir,
                                              MAX_INPUT_IMAGE_BYTES, INPUT_IMAGE_TIMEOUT)
        components = []
        for message in event.get_messages():
            chain = message.chain if isinstance(message, Reply) and message.chain else [message]
            components.extend(item for item in chain if self._is_edit_image(item, allow_image_files))
        if (not components and use_recent_chat_image and not image_paths
                and event.get_platform_name() == "synochat_adapter"):
            recent = self._recent_chat_image_component(event)
            if recent is not None:
                components.append(recent)
        if image_indices:
            selected = []
            for index in image_indices:
                if isinstance(index, bool) or not isinstance(index, (int, float)) or not 1 <= index <= len(components) or int(index) != index:
                    raise ValueError(f"图片序号无效：当前消息及引用消息共有 {len(components)} 张图，序号从 1 开始")
                selected.append(components[int(index) - 1])
            components = selected
        if take_first_only:
            components = components[:1]
        images = []
        max_bytes = MAX_MASK_SOURCE_BYTES if allow_image_files else MAX_INPUT_IMAGE_BYTES
        for component in components:
            logger.info("[ComfyUI] 读取输入原图：组件=%s，名称=%s，有url=%s，file=%s",
                        type(component).__name__, getattr(component, "name", None),
                        bool(getattr(component, "url", None)), str(getattr(component, "file", ""))[:80])
            data = await self._get_image_data(component, max_bytes=max_bytes)
            if not data:
                if allow_image_files:
                    raise ValueError("无法读取原图。请确认 QQ 文件未过期、大小不超过 48 MiB，再引用重试。")
                raise ValueError(f"第 {len(images) + 1} 张图片读取失败，请重试")
            images.append(data)
        if image_paths:
            images.extend(await load_executor_images(self.context, event, image_paths, self.temp_dir,
                                                     MAX_INPUT_IMAGE_BYTES, INPUT_IMAGE_TIMEOUT))
        return images

    @staticmethod
    def _image_delivery_notices(result):
        return [text for text in (result.get("warning"), result.get("error")) if text]

    def _load_image_resend(self, owner, task_id):
        task_id = task_id.casefold()
        if not IMAGE_RESEND_ID.fullmatch(task_id):
            return None
        record = self.temp_dir / f"{task_id}.delivery.json"
        try:
            info = record.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_size > 4096:
                return None
            if info.st_mtime < time.time() - GENERATED_MEDIA_TTL_SECONDS:
                return None
            item = json.loads(record.read_text(encoding="utf-8"))
            name = item.get("path")
            if item.get("owner") != list(owner):
                return None
            prompt_id = item.get("comfy_prompt_id")
            if not isinstance(prompt_id, str) or not COMFY_PROMPT_ID.fullmatch(prompt_id):
                prompt_id = None
            output_node = item.get("comfy_output_node")
            if output_node is not None and (not isinstance(output_node, str) or len(output_node) > 160):
                return None
            path = None
            if isinstance(name, str) and GENERATED_MEDIA_FILENAME.fullmatch(name):
                candidate = self.temp_dir / name
                try:
                    if candidate.suffix != ".mp4" and stat.S_ISREG(candidate.lstat().st_mode):
                        path = candidate
                except FileNotFoundError:
                    pass
            if path is None and prompt_id is None:
                return None
            return ImageTask(owner, str(item.get("backend", "Qwen")), id=task_id,
                             status="delivery_failed" if path else "result_unavailable", path=path,
                             comfy_prompt_id=prompt_id, comfy_output_node=output_node)
        except (OSError, ValueError, AttributeError):
            return None

    def _save_image_resend(self, task):
        if task.path is None and not task.comfy_prompt_id:
            return
        record = self.temp_dir / f"{task.id}.delivery.json"
        pending = record.with_suffix(".tmp")
        try:
            pending.write_text(json.dumps({
                "owner": task.owner, "backend": task.backend, "path": task.path.name if task.path else None,
                "comfy_prompt_id": task.comfy_prompt_id, "comfy_output_node": task.comfy_output_node,
            }, ensure_ascii=False), encoding="utf-8")
            pending.replace(record)
        except OSError as error:
            logger.warning("[ComfyUI] 保存图片重发记录失败：%s", type(error).__name__)

    def _clear_image_resend(self, task_id):
        try:
            (self.temp_dir / f"{task_id}.delivery.json").unlink(missing_ok=True)
        except OSError as error:
            logger.warning("[ComfyUI] 无法清理图片重发记录：%s", type(error).__name__)

    async def _send_chat_large_attachment(self, event, path, label):
        if event.get_platform_name() != "synochat_adapter":
            return False
        limit = getattr(event.adapter, "max_attachment_bytes", 32 * 1024 * 1024)
        if path.stat().st_size <= limit:
            return False
        if event.get_group_id() or not self._mask_editor.available:
            raise ValueError("超出 Chat 附件上限，限时下载需要私聊和可用的 HTTPS 编辑入口")
        link = self._mask_editor.share_result(path, self.temp_dir)
        await self._send_text_message(event, f"{label}超过附件上限，请在 15 分钟内下载：<{link}|下载{label}>")
        return True

    async def _send_generated_video(self, event, path):
        if event.get_platform_name() == "aiocqhttp":
            await self._call_send_api(event, f"[CQ:video,file=file://{path}]")
        elif not await self._send_chat_large_attachment(event, path, "视频"):
            await event.send(event.chain_result([VideoComponent.fromFileSystem(str(path))]))

    async def _deliver_generated_image(self, event, task, data=None, chain=False, *, force_qq_file=False, finish_task=True):
        result = {"status": "failed", "media_type": "image", "media_sent": False, "file_path": None}
        task.status = "sending"
        if task.send_started is None:
            task.send_started = time.monotonic()
        try:
            if data is not None:
                task.path, warning = await asyncio.to_thread(
                    self._maybe_compress_for_platform, data, event.get_platform_name()
                )
                task.chain = chain
                if warning:
                    result["warning"] = warning
            result["file_path"] = str(task.path.resolve())
            if event.get_platform_name() == "aiocqhttp":
                if force_qq_file or task.path.stat().st_size >= QQ_IMAGE_FILE_THRESHOLD:
                    await self._send_qq_image_file(event, task.path, task.id)
                else:
                    await self._send_image_message(event, task.path, task.chain)
            else:
                if not await self._send_chat_large_attachment(event, task.path, "图片"):
                    await event.send(event.image_result(str(task.path)))
        except asyncio.CancelledError:
            task.finish("delivery_failed" if task.path else "failed")
            self._save_image_resend(task)
            raise
        except Exception as error:
            logger.warning(f"[ComfyUI] 图片发送未确认：{type(error).__name__}")
            task.finish("delivery_failed" if task.path else "failed")
            self._save_image_resend(task)
            result["status"] = task.status
            result["error"] = (
                f"图片已生成，发送未获确认。可用 /重发图片 {task.id} 重发。"
                if task.path else "图片已生成，但保存文件失败。"
            )
        else:
            if finish_task:
                task.finish("sent")
                self._clear_image_resend(task.id)
            result.update(status="success", media_sent=True)
        return result

    async def _send_tool_image(self, event, data, error, task=None):
        if data is None:
            status = "cancelled" if task and task.status == "cancelled" else "failed"
            result = {
                "status": status, "media_type": "image", "media_sent": False, "file_path": None,
                "message": error or ("用户已取消排队任务。" if status == "cancelled" else "图片未生成。"),
            }
        else:
            result = await self._deliver_generated_image(event, task, data, self.default_chain)
            result["message"] = result.pop("error", "图片已生成并发送到当前会话。")
        result["feedback_hint"] = IMAGE_COMPLETION_FEEDBACK_HINT
        return json.dumps(result, ensure_ascii=False)
