import asyncio
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import aiohttp
from astrbot.api import logger


class ComfyUIUnavailable(aiohttp.ClientConnectionError):
    """Connection failed before a workflow could have been accepted."""


class ComfyUISubmissionUncertain(Exception):
    """The request may have been accepted. Do not submit it again."""


class ComfyUIResultUnavailable(Exception):
    """ComfyUI accepted the workflow but did not return an image."""


class ComfyUIExecutionError(Exception):
    """ComfyUI reported a node execution error with diagnostic details."""
    def __init__(
        self,
        prompt_id: str,
        user_message: str,
        *,
        node_type: str = "",
        exception_message: str = "",
        is_oom: bool = False,
    ):
        super().__init__(user_message)
        self.prompt_id = prompt_id
        self.user_message = user_message
        self.node_type = node_type
        self.exception_message = exception_message
        self.is_oom = is_oom


class ComfyUIAPI:
    def __init__(self, server_url: str = "http://127.0.0.1:8188", timeout: int = 300):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        # 用 UUID 避免多 AstrBot 实例共享同一 ComfyUI 时 client_id 撞号导致的归属判错
        self.client_id = uuid.uuid4().hex
        self._http_session = None
        # WebSocket 事件通道：仅用于更快发现完成与获取进度，结果仍以 /history 为准，断线时自动退回轮询。
        self._ws_task = None
        self._ws_connected = False
        self._done = {}      # prompt_id -> (state, monotonic time)，state 为 "ok" / "error" / "interrupted"
        self._errors = {}    # prompt_id -> (node_type, exc_type, exc_msg, monotonic time)
        self._waiters = {}   # prompt_id -> asyncio.Event
        self._progress = {}  # prompt_id -> callback(value, maximum, node)
        self._uploaded_images = OrderedDict()  # content-derived name -> saved input path (LRU, cap 512)

    @asynccontextmanager
    async def session(self):
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15, connect=4)
            )
        yield self._http_session

    async def close(self):
        if self._ws_task is not None:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except (asyncio.CancelledError, Exception):
                pass
            self._ws_task = None
        if self._http_session is not None:
            await self._http_session.close()

    # ---- WebSocket 事件通道 ----

    def _ws_url(self) -> str:
        base = self.server_url
        if base.startswith("https://"):
            base = "wss://" + base[len("https://"):]
        elif base.startswith("http://"):
            base = "ws://" + base[len("http://"):]
        return f"{base}/ws?clientId={self.client_id}"

    def _ensure_ws(self):
        if self._ws_task is None or self._ws_task.done():
            self._ws_task = asyncio.get_running_loop().create_task(self._ws_loop())

    async def _ws_loop(self):
        delay = 1.0
        while True:
            try:
                timeout = aiohttp.ClientTimeout(total=None, connect=5)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.ws_connect(self._ws_url(), heartbeat=20, max_msg_size=0) as ws:
                        self._ws_connected = True
                        delay = 1.0
                        logger.info("[ComfyUI] WebSocket 事件通道已连接")
                        async for message in ws:
                            if message.type == aiohttp.WSMsgType.TEXT:
                                self._handle_ws_message(message.data)
                            elif message.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if self._ws_connected:
                    logger.info("[ComfyUI] WebSocket 事件通道断开，将自动重连：%s", type(error).__name__)
                else:
                    logger.debug("[ComfyUI] WebSocket 连接失败：%s", type(error).__name__)
            self._ws_connected = False
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)

    def _handle_ws_message(self, raw):
        try:
            message = json.loads(raw)
            kind = message.get("type")
            data = message.get("data") or {}
            prompt_id = data.get("prompt_id")
        except (ValueError, AttributeError):
            return
        if not prompt_id:
            return
        prompt_id = str(prompt_id)
        if kind == "progress":
            callback = self._progress.get(prompt_id)
            if callback is not None:
                try:
                    callback(data.get("value"), data.get("max"), data.get("node"))
                except Exception as error:
                    logger.debug("[ComfyUI] 进度回调异常：%s", type(error).__name__)
        elif kind == "executing" and data.get("node") is None:
            self._mark_done(prompt_id, "ok")
        elif kind == "execution_success":
            self._mark_done(prompt_id, "ok")
        elif kind == "execution_interrupted":
            self._mark_done(prompt_id, "interrupted")
        elif kind == "execution_error":
            node_type = str(data.get("node_type") or "").strip()
            exc_type = str(data.get("exception_type") or "").strip()
            exc_msg = str(data.get("exception_message") or "").strip()
            self._record_error(prompt_id, node_type, exc_type, exc_msg)
            self._mark_done(prompt_id, "error")

    def _record_error(self, prompt_id: str, node_type: str, exc_type: str, exc_msg: str):
        now = time.monotonic()
        for key in [k for k, entry in self._errors.items() if now - entry[3] > 600]:
            self._errors.pop(key, None)
        self._errors[prompt_id] = (node_type, exc_type, exc_msg, now)
        if node_type or exc_type or exc_msg:
            logger.error(
                "[ComfyUI] 任务 %s 节点执行失败：node_type=%s, %s: %s",
                prompt_id, node_type or "?", exc_type or "Error", exc_msg[:300],
            )

    @staticmethod
    def _extract_history_error(record: dict) -> tuple[str, str, str]:
        messages = (record.get("status") or {}).get("messages") or []
        for item in messages:
            if isinstance(item, (list, tuple)) and len(item) >= 2 and item[0] == "execution_error":
                detail = item[1] if isinstance(item[1], dict) else {}
                return (
                    str(detail.get("node_type") or "").strip(),
                    str(detail.get("exception_type") or "").strip(),
                    str(detail.get("exception_message") or "").strip(),
                )
        return "", "", ""

    @staticmethod
    def _format_execution_error(node_type: str, exc_type: str, exc_msg: str) -> tuple[str, bool]:
        combined = f"{exc_type} {exc_msg}".casefold()
        is_oom = any(
            marker in combined
            for marker in ("out of memory", "cuda error", "cublas", "allocation on device", "oom")
        )
        if is_oom:
            return "ComfyUI 显存不足（OOM），已自动清理显存缓存，请稍后重试或适当降低分辨率。", True
        is_missing = any(
            marker in combined
            for marker in ("filenotfounderror", "no such file", "not found", "value not in list")
        )
        if is_missing:
            where = node_type or exc_type or "未知节点"
            return f"ComfyUI 缺少模型或节点依赖（{where}），请联系管理员检查。", False
        short_msg = " ".join(exc_msg.splitlines()).strip()
        if len(short_msg) > 120:
            short_msg = short_msg[:117] + "..."
        if node_type and short_msg:
            return f"ComfyUI 执行失败（{node_type}：{short_msg}）", False
        if short_msg:
            return f"ComfyUI 执行失败（{short_msg}）", False
        if node_type:
            return f"ComfyUI 节点 {node_type} 执行失败，请检查 ComfyUI 日志。", False
        return "生成失败，请检查 ComfyUI 日志", False

    def _mark_done(self, prompt_id: str, state: str):
        now = time.monotonic()
        for key in [k for k, (_, stamp) in self._done.items() if now - stamp > 600]:
            self._done.pop(key, None)
        self._done[prompt_id] = (state, now)
        event = self._waiters.get(prompt_id)
        if event is not None:
            event.set()

    async def _pace(self, prompt_id: str) -> bool:
        """代替固定 sleep(1)：WebSocket 报告完成后立即返回；WS 连通时放宽兜底轮询间隔减少无效 /history 请求。"""
        self._ensure_ws()
        done = self._done.get(prompt_id)
        if done is not None:
            if done[0] == "interrupted":
                return False
            if done[0] == "error" and prompt_id in self._errors:
                return False
            if time.monotonic() - done[1] > 20:
                return False
            await asyncio.sleep(0.15)
            return True
        event = self._waiters.setdefault(prompt_id, asyncio.Event())
        poll_interval = 12.0 if self._ws_connected else 2.0
        try:
            await asyncio.wait_for(event.wait(), timeout=poll_interval)
        except asyncio.TimeoutError:
            pass
        return True

    def _forget(self, prompt_id: str):
        self._waiters.pop(prompt_id, None)
        self._progress.pop(prompt_id, None)

    async def get_queue_info(self) -> dict:
        """获取 ComfyUI 队列状态（运行中和等待中的任务）

        每个队列项为 [number, prompt_id, workflow, extra_data, outputs_to_execute]。
        """
        try:
            async with self.session() as session:
                async with session.get(f"{self.server_url}/queue") as resp:
                    if resp.status == 200:
                        return await resp.json()
        except Exception:
            pass
        return {"queue_running": [], "queue_pending": []}

    def _get_prompt_id_from_item(self, queue_item) -> Optional[str]:
        if isinstance(queue_item, (list, tuple)) and len(queue_item) >= 2:
            return str(queue_item[1])
        return None

    async def _submit_prompt(self, workflow: dict) -> Optional[str]:
        """提交任务到 ComfyUI，返回 prompt_id"""
        try:
            async with self.session() as session:
                async with session.post(
                    f"{self.server_url}/prompt",
                    json={"prompt": workflow, "client_id": self.client_id},
                    allow_redirects=False,
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        if isinstance(result, dict) and result.get("prompt_id"):
                            return str(result["prompt_id"])
                        raise ComfyUISubmissionUncertain("响应中没有任务 ID")
                    logger.error(f"[ComfyUI] 提交任务失败，状态码: {resp.status}")
                    if resp.status >= 500:
                        raise ComfyUISubmissionUncertain(f"HTTP {resp.status}")
                    return None
        except aiohttp.ClientConnectorError as error:
            raise ComfyUIUnavailable("连接 ComfyUI 提交接口失败") from error
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as error:
            raise ComfyUISubmissionUncertain(type(error).__name__) from error

    async def queue_prompt(self, workflow: dict) -> Optional[str]:
        """提交任务，返回 prompt_id（仅提交，不等待结果，不经过队列缓冲）"""
        return await self._submit_prompt(workflow)

    async def _queue_and_wait(self, workflow: dict, fetch_fn, label: str,
                              max_wait: Optional[float] = None,
                              on_wait_callback=None,
                              on_submitted_callback=None, on_reference=None,
                              on_progress=None):
        """直接提交工作流并通过 fetch_fn 获取结果（排队由外层 LocalImageGenerationQueue 或 ComfyUI 自身队列负责）。"""
        self._ensure_ws()  # 提前建立 WebSocket，避免错过早期事件
        logger.info(f"[ComfyUI] 开始提交{label}任务")
        try:
            prompt_id = await self._submit_prompt(workflow)
        except (ComfyUIUnavailable, ComfyUISubmissionUncertain) as error:
            if label == "图片":
                raise
            logger.warning(f"[ComfyUI] {label}任务提交异常：{error}")
            return None
        if not prompt_id:
            logger.error("[ComfyUI] 提交任务失败")
            return None
        if on_progress is not None:
            self._progress[prompt_id] = on_progress
        self._ensure_ws()
        if on_reference is not None:
            try:
                await on_reference(prompt_id)
            except Exception as error:
                logger.warning("[ComfyUI] 保存任务引用失败：%s", type(error).__name__)

        queue_position, tasks_ahead, extra_timeout = await self._calc_queue_timeout(prompt_id)
        if on_submitted_callback:
            try:
                await on_submitted_callback(prompt_id, queue_position, tasks_ahead)
            except Exception as e:
                logger.error(f"[ComfyUI] 提交回调异常: {e}")

        # Individual workflows can request more time than the plugin default.
        base_timeout = self.timeout if max_wait is None else max(self.timeout, max_wait)
        total_timeout = base_timeout + extra_timeout
        logger.info(f"[ComfyUI] 任务 {prompt_id} 已提交，等待{label}结果...（总超时: {total_timeout}s）")
        try:
            result = await asyncio.wait_for(
                fetch_fn(prompt_id, total_timeout), timeout=total_timeout
            )
        except asyncio.TimeoutError:
            result = None
        finally:
            self._forget(prompt_id)

        if result:
            logger.info(f"[ComfyUI] 任务 {prompt_id} 完成")
        else:
            done = self._done.get(prompt_id)
            if done and done[0] == "interrupted":
                logger.info(f"[ComfyUI] 任务 {prompt_id} 已被中断")
                return None
            err = self._errors.get(prompt_id)
            if (done and done[0] == "error") or err is not None:
                node_type, exc_type, exc_msg = (err[0], err[1], err[2]) if err else ("", "", "")
                user_message, is_oom = self._format_execution_error(node_type, exc_type, exc_msg)
                if is_oom:
                    logger.warning("[ComfyUI] 任务 %s 触发显存不足（OOM），自动调用 /free 清理显存缓存", prompt_id)
                    await self.free_memory(unload_models=True, free_memory=True)
                raise ComfyUIExecutionError(
                    prompt_id, user_message,
                    node_type=node_type, exception_message=exc_msg, is_oom=is_oom,
                )
            logger.error(f"[ComfyUI] 任务 {prompt_id} 等待结果超时或失败")
            if label == "图片":
                raise ComfyUIResultUnavailable(prompt_id)

        return result

    async def queue_and_wait_image(self, workflow: dict, max_wait: Optional[float] = None,
                                   on_wait_callback=None, on_submitted_callback=None,
                                   output_node: Optional[str] = None, on_result_reference=None,
                                   on_progress=None) -> Optional[bytes]:
        async def fetch(prompt_id: str, total_timeout: float):
            return await self.wait_result(prompt_id, output_node=output_node, total_timeout=total_timeout)

        async def record(prompt_id):
            if on_result_reference:
                await on_result_reference(prompt_id, output_node)

        return await self._queue_and_wait(
            workflow, fetch, "图片",
            max_wait=max_wait,
            on_wait_callback=on_wait_callback,
            on_submitted_callback=on_submitted_callback,
            on_reference=record,
            on_progress=on_progress,
        )

    async def queue_and_wait_images(self, workflow: dict, output_nodes: list[str],
                                    max_wait: Optional[float] = None,
                                    on_wait_callback=None, on_submitted_callback=None,
                                    on_result_reference=None, on_progress=None) -> Optional[list[bytes]]:
        """提交工作流并按指定输出节点顺序等待多张图片结果。"""
        nodes = [str(n) for n in output_nodes]

        async def fetch(prompt_id: str, total_timeout: float):
            return await self.wait_results(prompt_id, nodes, total_timeout=total_timeout)

        async def record(prompt_id):
            if on_result_reference:
                await on_result_reference(prompt_id, nodes[-1] if nodes else None)

        return await self._queue_and_wait(
            workflow, fetch, "图片",
            max_wait=max_wait,
            on_wait_callback=on_wait_callback,
            on_submitted_callback=on_submitted_callback,
            on_reference=record,
            on_progress=on_progress,
        )

    async def queue_and_wait_video(self, workflow: dict, max_wait: Optional[float] = None,
                                   on_wait_callback=None, on_submitted_callback=None,
                                   on_progress=None, on_result_reference=None) -> Optional[bytes]:
        """提交工作流并等待视频结果"""
        async def fetch(prompt_id: str, total_timeout: float):
            return await self.wait_video_result(prompt_id, total_timeout=total_timeout)

        async def record(prompt_id):
            if on_result_reference:
                await on_result_reference(prompt_id)

        return await self._queue_and_wait(
            workflow, fetch, "视频",
            max_wait=max_wait,
            on_wait_callback=on_wait_callback,
            on_submitted_callback=on_submitted_callback,
            on_reference=record,
            on_progress=on_progress,
        )

    async def queue_and_wait_text(self, workflow: dict, output_node: str = "",
                                  max_wait: Optional[float] = None,
                                  on_wait_callback=None, on_submitted_callback=None) -> Optional[str]:
        """提交工作流并等待文本结果（带队列缓冲）"""
        async def fetch(prompt_id: str, total_timeout: float):
            return await self.wait_text_result(prompt_id, output_node, total_timeout=total_timeout)

        return await self._queue_and_wait(
            workflow, fetch, "文本",
            max_wait=max_wait,
            on_wait_callback=on_wait_callback,
            on_submitted_callback=on_submitted_callback,
        )

    async def wait_video_result(self, prompt_id: str, extra_timeout: int = 0, *,
                                total_timeout: Optional[float] = None) -> Optional[bytes]:
        """等待并下载视频结果

        SaveVideo 节点的输出格式类似图片，存储在 outputs[node]["videos"] 中，
        每项包含 filename / subfolder / type 字段。
        """
        if total_timeout is None:
            total_timeout = self.timeout + extra_timeout
        loop = asyncio.get_event_loop()
        deadline = loop.time() + total_timeout
        async with self.session() as session:
            while loop.time() < deadline:
                if not await self._pace(prompt_id):
                    return None
                try:
                    async with session.get(f"{self.server_url}/history/{prompt_id}") as resp:
                        if resp.status == 200:
                            history = await resp.json()
                            if prompt_id in history:
                                record = history[prompt_id]
                                if record.get("status", {}).get("status_str") == "error":
                                    node_type, exc_type, exc_msg = self._extract_history_error(record)
                                    if prompt_id not in self._errors:
                                        self._record_error(prompt_id, node_type, exc_type, exc_msg)
                                    self._mark_done(prompt_id, "error")
                                    return None
                                outputs = record.get("outputs", {})
                                for node_output in outputs.values():
                                    # 兼容多种字段：videos / gifs / images（部分视频节点输出沿用 images 字段）
                                    for key in ("videos", "gifs", "images"):
                                        items = node_output.get(key)
                                        if not items:
                                            continue
                                        for item in items:
                                            filename = item.get("filename", "")
                                            # 跳过纯图片输出
                                            if key == "images" and not self._is_video_filename(filename):
                                                continue
                                            params = {"filename": filename, "subfolder": item.get("subfolder", ""), "type": item.get("type", "output")}
                                            async with session.get(f"{self.server_url}/view", params=params) as v_resp:
                                                if v_resp.status == 200:
                                                    return await v_resp.read()
                except (aiohttp.ClientError, asyncio.TimeoutError, KeyError):
                    continue
        return None

    @staticmethod
    def _is_video_filename(filename: str) -> bool:
        if not filename:
            return False
        lower = filename.lower()
        return any(lower.endswith(ext) for ext in (".mp4", ".webm", ".mov", ".mkv", ".avi", ".gif"))

    async def _calc_queue_timeout(self, prompt_id: str) -> tuple:
        """根据任务在队列中的位置计算额外超时

        检测提交的任务是否已在运行中：
        - 如果已在运行，无需额外超时
        - 如果在排队中，根据前方排队数量计算额外超时

        Returns:
            (queue_position, tasks_ahead, extra_timeout)
            queue_position: 0=已在运行, >0=排队位置(1-based), -1=不在队列
            tasks_ahead: 前方任务数（运行中+排队前方）
            extra_timeout: 额外超时秒数
        """
        info = await self.get_queue_info()
        running = info.get("queue_running", [])
        pending = info.get("queue_pending", [])

        # 检查任务是否已在运行中
        for item in running:
            if self._get_prompt_id_from_item(item) == str(prompt_id):
                logger.info(f"[ComfyUI] 任务 {prompt_id} 已在运行中，无需额外超时")
                return (0, 0, 0)

        # 计算任务在排队中的位置（前方有多少个任务）
        own_position = 0
        found = False
        for item in pending:
            if self._get_prompt_id_from_item(item) == str(prompt_id):
                found = True
                break
            own_position += 1

        if found:
            # 前方 own_position 个排队任务 + 运行中的任务
            tasks_ahead = own_position + len(running)
            extra = max(120, tasks_ahead * 60)
            logger.info(f"[ComfyUI] 任务 {prompt_id} 在队列第 {own_position + 1} 位（前方 {tasks_ahead} 个任务），额外超时 {extra}s")
            return (own_position + 1, tasks_ahead, extra)
        else:
            # 任务不在队列中（可能已执行完毕或提交失败），不增加超时
            return (-1, 0, 0)

    async def wait_result(self, prompt_id: str, extra_timeout: int = 0,
                          output_node: Optional[str] = None, *,
                          total_timeout: Optional[float] = None) -> Optional[bytes]:
        """等待并下载图片结果

        Args:
            prompt_id: 任务ID
            extra_timeout: 额外超时秒数，用于强制提交时补偿队列排队时间
        """
        if total_timeout is None:
            total_timeout = self.timeout + extra_timeout
        loop = asyncio.get_event_loop()
        deadline = loop.time() + total_timeout
        async with self.session() as session:
            while loop.time() < deadline:
                if not await self._pace(prompt_id):
                    logger.error("[ComfyUI] 任务 %s 已结束但迟迟没有结果记录", prompt_id)
                    return None
                status, data = await self._fetch_image_once(session, prompt_id, output_node)
                if status == "ready":
                    return data
                if status == "failed":
                    logger.error("[ComfyUI] 任务 %s 失败或声明的图片输出为空: %s", prompt_id, output_node)
                    return None
        return None

    async def wait_results(self, prompt_id: str, output_nodes: list[str], extra_timeout: int = 0, *,
                           total_timeout: Optional[float] = None) -> Optional[list[bytes]]:
        """等待并按顺序下载多个输出节点的图片结果。"""
        if total_timeout is None:
            total_timeout = self.timeout + extra_timeout
        loop = asyncio.get_event_loop()
        deadline = loop.time() + total_timeout
        async with self.session() as session:
            while loop.time() < deadline:
                if not await self._pace(prompt_id):
                    logger.error("[ComfyUI] 任务 %s 已结束但迟迟没有结果记录", prompt_id)
                    return None
                status, images_data = await self._fetch_images_once(session, prompt_id, output_nodes)
                if status == "ready":
                    return images_data
                if status == "failed":
                    logger.error("[ComfyUI] 任务 %s 失败或声明的多图输出为空: %s", prompt_id, output_nodes)
                    return None
        return None

    async def _fetch_images_once(self, session, prompt_id, output_nodes: list[str]):
        try:
            async with session.get(f"{self.server_url}/history/{prompt_id}") as response:
                if response.status != 200:
                    return "unavailable", None
                record = (await response.json()).get(prompt_id)
            if not record:
                return "pending", None
            status = record.get("status", {})
            if status.get("status_str") == "error":
                node_type, exc_type, exc_msg = self._extract_history_error(record)
                if prompt_id not in self._errors:
                    self._record_error(prompt_id, node_type, exc_type, exc_msg)
                self._mark_done(prompt_id, "error")
                return "failed", None
            outputs = record.get("outputs", {})
            picked = []
            for node_id in output_nodes:
                node_images = [
                    image for image in outputs.get(node_id, {}).get("images", [])
                    if isinstance(image, dict) and image.get("filename")
                ]
                if not node_images:
                    return ("failed" if status.get("completed") else "pending"), None
                picked.append(node_images[0])
            downloaded = []
            for image in picked:
                params = {
                    "filename": image["filename"],
                    "subfolder": image.get("subfolder", ""),
                    "type": image.get("type", "output"),
                }
                async with session.get(
                    f"{self.server_url}/view", params=params,
                    timeout=aiohttp.ClientTimeout(total=90, connect=4),
                ) as response:
                    if response.status != 200:
                        return "unavailable", None
                    data = await response.read()
                    if not data:
                        return "unavailable", None
                    downloaded.append(data)
            return "ready", downloaded
        except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, AttributeError, TypeError, ValueError):
            pass
        return "unavailable", None

    async def get_image_result(self, prompt_id, output_node=None):
        async with self.session() as session:
            return await self._fetch_image_once(session, prompt_id, output_node)

    async def _fetch_image_once(self, session, prompt_id, output_node):
        try:
            async with session.get(f"{self.server_url}/history/{prompt_id}") as response:
                if response.status != 200:
                    return "unavailable", None
                record = (await response.json()).get(prompt_id)
            if not record:
                return "pending", None
            status = record.get("status", {})
            if status.get("status_str") == "error":
                node_type, exc_type, exc_msg = self._extract_history_error(record)
                if prompt_id not in self._errors:
                    self._record_error(prompt_id, node_type, exc_type, exc_msg)
                self._mark_done(prompt_id, "error")
                return "failed", None
            outputs = record.get("outputs", {})
            nodes = [outputs.get(output_node, {})] if output_node is not None else outputs.values()
            images = [image for node in nodes for image in node.get("images", [])
                      if isinstance(image, dict) and image.get("filename")]
            if output_node is None:
                images.sort(key=lambda image: image.get("type", "output") != "output")
            if not images:
                return ("failed" if status.get("completed") else "pending"), None
            for image in images:
                params = {"filename": image["filename"], "subfolder": image.get("subfolder", ""),
                          "type": image.get("type", "output")}
                async with session.get(f"{self.server_url}/view", params=params,
                                       timeout=aiohttp.ClientTimeout(total=90, connect=4)) as response:
                    if response.status == 200:
                        data = await response.read()
                        if data:
                            return "ready", data
        except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, AttributeError, TypeError, ValueError):
            pass
        return "unavailable", None

    @staticmethod
    def _content_hash_filename(filename: str, image_data: bytes) -> str:
        ext = Path(filename or "").suffix.lower().lstrip(".") or "png"
        if ext not in {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "gif"}:
            ext = "png"
        digest = hashlib.sha256(image_data).hexdigest()[:24]
        return f"hub_{digest}.{ext}"

    async def upload_image(self, filename: str, image_data: bytes) -> Optional[str]:
        """按内容 SHA-256 去重上传图片到 ComfyUI input/ 目录，成功返回实际保存的文件名，失败返回 None。"""
        target_name = self._content_hash_filename(filename, image_data)
        try:
            async with self.session() as session:
                cached = self._uploaded_images.get(target_name)
                if cached is not None:
                    subfolder, _, basename = cached.rpartition("/")
                    params = {"filename": basename, "subfolder": subfolder, "type": "input"}
                    try:
                        async with session.head(
                            f"{self.server_url}/view", params=params,
                            timeout=aiohttp.ClientTimeout(total=3, connect=2),
                        ) as response:
                            exists = response.status == 200
                    except (aiohttp.ClientError, asyncio.TimeoutError):
                        exists = False
                    if exists:
                        if self._uploaded_images.get(target_name) == cached:
                            self._uploaded_images.move_to_end(target_name)
                        logger.debug("[ComfyUI] 复用已上传输入图片：%s", cached)
                        return cached
                    if self._uploaded_images.get(target_name) == cached:
                        self._uploaded_images.pop(target_name)
                data = aiohttp.FormData()
                data.add_field("image", image_data, filename=target_name)
                async with session.post(
                    f"{self.server_url}/upload/image", data=data,
                    timeout=aiohttp.ClientTimeout(total=30, connect=4),
                ) as resp:
                    if resp.status == 200:
                        try:
                            payload = await resp.json()
                            actual = str(payload.get("name") or target_name)
                            subfolder = str(payload.get("subfolder") or "").replace("\\", "/").strip("/")
                            if subfolder:
                                actual = f"{subfolder}/{actual}"
                        except (aiohttp.ClientError, ValueError, AttributeError):
                            return target_name
                        self._uploaded_images[target_name] = actual
                        self._uploaded_images.move_to_end(target_name)
                        while len(self._uploaded_images) > 512:
                            self._uploaded_images.popitem(last=False)
                        return actual
                    if resp.status in (502, 503, 504):
                        raise ComfyUIUnavailable(f"ComfyUI 上传接口返回 HTTP {resp.status}")
                    logger.error(f"[ComfyUI] 上传图片失败，状态码: {resp.status}")
                    return None
        except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as error:
            raise ComfyUIUnavailable("连接 ComfyUI 上传接口失败") from error

    async def cancel_prompt(self, prompt_id: Optional[str] = None) -> bool:
        """中断当前正在运行或位于 ComfyUI 队列中的任务。"""
        if prompt_id:
            self._mark_done(str(prompt_id), "interrupted")
        ok = False
        try:
            async with self.session() as session:
                if prompt_id:
                    info = await self.get_queue_info()
                    running_ids = {
                        self._get_prompt_id_from_item(item)
                        for item in info.get("queue_running", [])
                    }
                    pending_ids = {
                        self._get_prompt_id_from_item(item)
                        for item in info.get("queue_pending", [])
                    }
                    if str(prompt_id) in pending_ids:
                        async with session.post(
                            f"{self.server_url}/queue",
                            json={"delete": [str(prompt_id)]},
                        ) as resp:
                            ok = resp.status == 200 or ok
                    if str(prompt_id) in running_ids or (
                        str(prompt_id) not in pending_ids
                    ):
                        async with session.post(
                            f"{self.server_url}/interrupt",
                            json={"prompt_id": str(prompt_id)},
                        ) as resp:
                            ok = resp.status == 200 or ok
                else:
                    async with session.post(f"{self.server_url}/interrupt") as resp:
                        ok = resp.status == 200
        except Exception as error:
            logger.warning("[ComfyUI] 请求中断任务失败：%s", type(error).__name__)
        return ok

    async def free_memory(self, *, unload_models: bool = True, free_memory: bool = True) -> bool:
        """调用 ComfyUI /free 接口卸载模型并释放显存/内存缓存。"""
        try:
            async with self.session() as session:
                async with session.post(
                    f"{self.server_url}/free",
                    json={"unload_models": bool(unload_models), "free_memory": bool(free_memory)},
                    timeout=aiohttp.ClientTimeout(total=15, connect=4),
                ) as resp:
                    return resp.status == 200
        except Exception as error:
            logger.warning("[ComfyUI] 释放显存请求失败：%s", type(error).__name__)
            return False

    async def get_system_stats(self, *, timeout: float = 2.5) -> Optional[dict]:
        """获取 ComfyUI /system_stats 中的 GPU 显存占用信息。"""
        try:
            async with self.session() as session:
                async with session.get(
                    f"{self.server_url}/system_stats",
                    timeout=aiohttp.ClientTimeout(total=timeout, connect=min(timeout, 2.0)),
                ) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()
        except Exception:
            return None
        devices = data.get("devices") if isinstance(data, dict) else None
        if not isinstance(devices, list) or not devices:
            return None
        dev = next((d for d in devices if isinstance(d, dict) and d.get("type") == "cuda"), devices[0])
        if not isinstance(dev, dict):
            return None
        total = float(dev.get("vram_total") or 0)
        free = float(dev.get("vram_free") or 0)
        used = max(0.0, total - free)
        return {
            "name": str(dev.get("name") or "GPU").split(":")[0].strip(),
            "vram_total_gb": round(total / (1024 ** 3), 1) if total > 0 else 0.0,
            "vram_used_gb": round(used / (1024 ** 3), 1) if total > 0 else 0.0,
            "vram_percent": int(round(used * 100 / total)) if total > 0 else 0,
        }

    async def wait_text_result(self, prompt_id: str, output_node: str = "", extra_timeout: int = 0, *,
                               total_timeout: Optional[float] = None) -> Optional[str]:
        """等待并获取文本结果

        Args:
            prompt_id: 任务ID
            output_node: 输出节点ID
            extra_timeout: 额外超时秒数，用于强制提交时补偿队列排队时间
        """
        if total_timeout is None:
            total_timeout = self.timeout + extra_timeout
        loop = asyncio.get_event_loop()
        deadline = loop.time() + total_timeout
        async with self.session() as session:
            while loop.time() < deadline:
                if not await self._pace(prompt_id):
                    return None
                try:
                    async with session.get(f"{self.server_url}/history/{prompt_id}") as resp:
                        if resp.status == 200:
                            history = await resp.json()
                            if prompt_id in history:
                                record = history[prompt_id]
                                if record.get("status", {}).get("status_str") == "error":
                                    node_type, exc_type, exc_msg = self._extract_history_error(record)
                                    if prompt_id not in self._errors:
                                        self._record_error(prompt_id, node_type, exc_type, exc_msg)
                                    self._mark_done(prompt_id, "error")
                                    return None
                                outputs = record.get("outputs", {})

                                # 如果指定了输出节点，只查找该节点的输出
                                if output_node and output_node in outputs:
                                    node_output = outputs[output_node]

                                    # 检查 string 字段（常见输出）
                                    if "string" in node_output and node_output["string"]:
                                        # 如果是数组，返回第一个元素；如果是字符串，直接返回
                                        return node_output["string"][0] if isinstance(node_output["string"], list) and node_output["string"] else node_output["string"]

                                    # 检查 tags 字段（WD14Tagger 输出）
                                    if "tags" in node_output and node_output["tags"]:
                                        # 如果是数组，返回第一个元素；如果是字符串，直接返回
                                        return node_output["tags"][0] if isinstance(node_output["tags"], list) and node_output["tags"] else node_output["tags"]

                                # 否则查找所有节点的文本输出
                                for node_output in outputs.values():
                                    # 检查 string 字段（常见输出）
                                    if "string" in node_output and node_output["string"]:
                                        # 如果是数组，返回第一个元素；如果是字符串，直接返回
                                        return node_output["string"][0] if isinstance(node_output["string"], list) and node_output["string"] else node_output["string"]

                                    # 检查 tags 字段（WD14Tagger 输出）
                                    if "tags" in node_output and node_output["tags"]:
                                        # 如果是数组，返回第一个元素；如果是字符串，直接返回
                                        return node_output["tags"][0] if isinstance(node_output["tags"], list) and node_output["tags"] else node_output["tags"]
                except (aiohttp.ClientError, asyncio.TimeoutError, KeyError):
                    continue
        return None
