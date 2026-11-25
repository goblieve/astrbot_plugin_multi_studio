import aiohttp
import asyncio
import hashlib
import json
import time
import uuid
from PIL import Image as PILImage
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from io import BytesIO
from pathlib import Path
from .comfyui_api import ComfyUIExecutionError, ComfyUIResultUnavailable, ComfyUISubmissionUncertain, ComfyUIUnavailable
from .llm_tools import COMPLETION_FEEDBACK_HINT
from .masked_workflow import MaskOriginal, MaskedWorkflow, plan_tiled_upscale, tiled_upscale_image


class GenerationMixin:
    @staticmethod
    def _image_progress_message(backend: str, images, fallback: bool = False) -> str:
        verb = "编辑图片" if images else "生成图片"
        ending = "..." if images else "…"
        if fallback:
            return f"GPT image正在{verb}{ending}"
        if backend == "qwen":
            return f"Qwen image正在{verb}{ending}"
        return f"正在{verb}{ending}"

    def _make_queue_callbacks(self, event: AstrMessageEvent, generating_msg: str, notify: bool = True):
        if not notify:
            return None, None

        async def on_submitted(prompt_id: str, queue_position: int, tasks_ahead: int):
            if queue_position > 0:
                await self._send_text_message(
                    event,
                    f"{generating_msg} 已提交，ComfyUI 队列第 {queue_position} 位（前方 {tasks_ahead} 个任务）",
                )
            else:
                await self._send_text_message(event, generating_msg)

        return None, on_submitted

    REROLL_TTL_SECONDS = 30 * 60
    REROLL_MAX_ENTRIES = 32

    def _save_reroll_blob_sync(self, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()[:24]
        path = self.temp_dir / f"reroll_ref_{digest}.bin"
        if path.is_file():
            try:
                path.touch()
            except OSError:
                pass
            return str(path)
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_bytes(data)
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
        return str(path)

    @staticmethod
    def _read_reroll_blob_sync(path_str: str):
        path = Path(path_str)
        if not path.is_file():
            return None
        return path.read_bytes()

    async def _remember_reroll(self, event: AstrMessageEvent, spec: dict):
        """将最近一次成功的生成参数写入轻量内存索引，原图/遮罩大字节落盘复用 SHA-256 缓存文件。"""
        store = getattr(self, "_last_generation_params", None)
        if store is None:
            store = {}
            self._last_generation_params = store
        now = time.monotonic()
        for key in [k for k, v in store.items() if now - v.get("created_at", 0) > self.REROLL_TTL_SECONDS]:
            store.pop(key, None)

        compact = dict(spec)
        try:
            if isinstance(compact.get("image_bytes"), (bytes, bytearray)):
                compact["image_bytes_path"] = await asyncio.to_thread(
                    self._save_reroll_blob_sync, bytes(compact.pop("image_bytes"))
                )
            if isinstance(compact.get("reference_images"), list):
                paths = []
                for item in compact.pop("reference_images"):
                    if isinstance(item, (bytes, bytearray)):
                        paths.append(await asyncio.to_thread(self._save_reroll_blob_sync, bytes(item)))
                compact["reference_image_paths"] = paths or None
            if isinstance(compact.get("edit_mask"), (bytes, bytearray)):
                compact["edit_mask_path"] = await asyncio.to_thread(
                    self._save_reroll_blob_sync, bytes(compact.pop("edit_mask"))
                )
            native_orig = compact.get("native_mask_original")
            if isinstance(native_orig, MaskOriginal):
                compact.pop("native_mask_original", None)
                compact["native_mask_original_ref"] = {
                    "data_path": await asyncio.to_thread(self._save_reroll_blob_sync, native_orig.data),
                    "preview_path": await asyncio.to_thread(self._save_reroll_blob_sync, native_orig.preview),
                    "size": native_orig.size,
                    "extension": native_orig.extension,
                    "preview_size": native_orig.preview_size,
                    "preview_extension": native_orig.preview_extension,
                }
        except OSError as error:
            logger.warning("[ComfyUI] 保存重画原图缓存失败：%s", type(error).__name__)
            return

        owner = self._image_task_owner(event)
        store.pop(owner, None)
        store[owner] = {**compact, "created_at": now}
        while len(store) > self.REROLL_MAX_ENTRIES:
            oldest = next(iter(store))
            store.pop(oldest, None)

    def _get_reroll_spec(self, event: AstrMessageEvent):
        store = getattr(self, "_last_generation_params", None)
        if not store:
            return None
        owner = self._image_task_owner(event)
        spec = store.get(owner)
        if spec is None:
            return None
        if time.monotonic() - spec.get("created_at", 0) > self.REROLL_TTL_SECONDS:
            store.pop(owner, None)
            return None
        return spec

    async def _materialize_reroll_spec(self, event: AstrMessageEvent):
        """从磁盘缓存恢复重画所需的原始图片与遮罩字节。"""
        raw = self._get_reroll_spec(event)
        if raw is None:
            return None
        spec = dict(raw)
        try:
            if "image_bytes_path" in spec:
                data = await asyncio.to_thread(self._read_reroll_blob_sync, spec.pop("image_bytes_path"))
                if data is None:
                    return None
                spec["image_bytes"] = data
            if "reference_image_paths" in spec:
                paths = spec.pop("reference_image_paths")
                if paths:
                    loaded = []
                    for p in paths:
                        data = await asyncio.to_thread(self._read_reroll_blob_sync, p)
                        if data is None:
                            return None
                        loaded.append(data)
                    spec["reference_images"] = loaded
                else:
                    spec["reference_images"] = None
            if "edit_mask_path" in spec:
                mask = await asyncio.to_thread(self._read_reroll_blob_sync, spec.pop("edit_mask_path"))
                if mask is None:
                    return None
                spec["edit_mask"] = mask
            if "native_mask_original_ref" in spec:
                ref = spec.pop("native_mask_original_ref")
                orig_data = await asyncio.to_thread(self._read_reroll_blob_sync, ref["data_path"])
                orig_preview = await asyncio.to_thread(self._read_reroll_blob_sync, ref["preview_path"])
                if orig_data is None or orig_preview is None:
                    return None
                spec["native_mask_original"] = MaskOriginal(
                    orig_data, ref["size"], ref["extension"],
                    orig_preview, ref["preview_size"], ref["preview_extension"],
                )
        except OSError as error:
            logger.warning("[ComfyUI] 读取重画原图缓存失败：%s", type(error).__name__)
            return None
        return spec

    def _check_user_queue_limit(self, event: AstrMessageEvent, limit: int = 2):
        """限制单用户同时排队/运行中的任务数（管理员不受限），防群聊刷屏堵死队列。"""
        try:
            if event.is_admin():
                return None
        except Exception:
            pass
        owner = self._image_task_owner(event)
        active = self._image_generation_queue.count_active(owner)
        if active >= limit:
            return (
                f"你当前已有 {active} 个任务正在排队或执行中，请等其中一个完成后再提交"
                "（可用 /绘图状态 查看进度，或 /取消绘图 取消排队）。"
            )
        return None

    def _make_progress_callback(
        self,
        event: AstrMessageEvent,
        label: str,
        *,
        notify: bool = True,
        task=None,
        min_interval: float = 45.0,
        expected_passes: int = 1,
        stage_resolver=None,
    ):
        """生成 ComfyUI 进度回调：实时更新 task.progress 与 task.stage，并按里程碑+冷却向用户发送进度提示。"""
        if not notify and task is None:
            return None
        started = time.monotonic()
        last_sent = started
        sent_milestones = set()
        current_node = None
        prev_step = 0
        pass_index = -1

        async def _notify(text: str):
            try:
                await self._send_text_message(event, text)
            except Exception as error:
                logger.debug("[ComfyUI] 进度提示发送失败：%s", type(error).__name__)

        def on_progress(value, maximum, node):
            nonlocal last_sent, current_node, prev_step, pass_index
            try:
                step = int(value)
                total = int(maximum)
            except (TypeError, ValueError):
                return
            if total <= 0 or step <= 0:
                return
            if expected_passes > 1:
                if total >= 4:
                    if current_node is None or node != current_node or step < prev_step:
                        pass_index = min(expected_passes - 1, pass_index + 1)
                    current_node = node
                    prev_step = step
                    overall_step = pass_index * total + step
                    overall_total = expected_passes * total
                    stage_text = (
                        stage_resolver(pass_index, expected_passes, node)
                        if callable(stage_resolver)
                        else f"第 {pass_index + 1}/{expected_passes} 张"
                    )
                    if task is not None:
                        task.progress = (overall_step, overall_total)
                        task.stage = stage_text
                    if not notify or overall_step >= overall_total:
                        return
                    percent = int(overall_step * 100 // overall_total)
                    milestone = (percent // 25) * 25
                    if milestone not in (25, 50, 75) or milestone in sent_milestones:
                        return
                    now = time.monotonic()
                    if now - last_sent < min_interval:
                        return
                    sent_milestones.add(milestone)
                    last_sent = now
                    asyncio.get_running_loop().create_task(
                        _notify(f"{label}进度 {percent}%（{stage_text} · {overall_step}/{overall_total}步）")
                    )
                return

            if task is not None:
                task.progress = (step, total)
                if callable(stage_resolver):
                    task.stage = stage_resolver(0, 1, node)
                elif not task.stage:
                    task.stage = f"{label}采样中"
            if not notify or total < 4 or step >= total:
                return
            if node != current_node:
                current_node = node
                sent_milestones.clear()
            percent = int(step * 100 // total)
            milestone = (percent // 25) * 25
            if milestone not in (25, 50, 75) or milestone in sent_milestones:
                return
            now = time.monotonic()
            if now - last_sent < min_interval:
                return
            sent_milestones.add(milestone)
            last_sent = now
            asyncio.get_running_loop().create_task(
                _notify(f"{label}进度 {percent}%（{step}/{total}）")
            )

        return on_progress

    async def _is_comfyui_available(self):
        """Return whether the configured ComfyUI API responds to its health endpoint."""
        url = f"{self.api.server_url}/system_stats"
        timeout = aiohttp.ClientTimeout(total=self.codex_fallback_health_timeout)
        try:
            async with self.api.session() as session:
                async with session.get(url, timeout=timeout) as response:
                    if response.status == 200:
                        return True
                    logger.warning(f"[ComfyUI] health check returned HTTP {response.status}")
                    return False
        except (aiohttp.ClientError, asyncio.TimeoutError, TimeoutError) as error:
            logger.warning(f"[ComfyUI] health check failed ({type(error).__name__})")
            return False
        except Exception as error:
            logger.warning(f"[ComfyUI] health check failed ({type(error).__name__})")
            return False

    def _select_codex_image_provider(self):
        provider_id = self.codex_fallback_provider_id
        try:
            candidates = self.context.get_all_providers() or []
        except Exception as error:
            logger.warning(f"[ComfyUI] could not enumerate providers ({type(error).__name__})")
            candidates = []
        matching = []
        for candidate in candidates:
            config = getattr(candidate, "provider_config", {})
            if not hasattr(config, "get"):
                continue
            candidate_id = str(config.get("id", "")).strip()
            if candidate_id == provider_id or candidate_id.startswith(provider_id.rstrip("/") + "/"):
                matching.append((candidate_id != provider_id, candidate))
        for _, candidate in sorted(matching, key=lambda item: item[0]):
            capabilities = getattr(candidate, "capabilities", {})
            if isinstance(capabilities, dict) and capabilities.get("image_generate") and callable(
                getattr(candidate, "generate_image", None)
            ):
                return candidate, None
        if matching:
            return None, "当前 Codex OAuth 提供商没有启用图片生成接口。"
        return None, f"未找到 Codex OAuth 图片提供商（{provider_id}）；请确认插件已启用且提供商已注册。"

    @staticmethod
    def _save_codex_ref_png(image_data: bytes, reference_path: Path):
        with PILImage.open(BytesIO(image_data)) as reference_image:
            reference_image.load()
            reference_image.save(reference_path, format="PNG", compress_level=1)

    async def _generate_with_codex(self, prompt, negative=None, width=None, height=None, images=None):
        provider, error = self._select_codex_image_provider()
        if provider is None:
            return None, error
        generate_image = provider.generate_image

        reference_paths = []
        try:
            for image_data in images or []:
                reference_path = self.temp_dir / f"codex_fallback_ref_{uuid.uuid4().hex}.png"
                await asyncio.to_thread(self._save_codex_ref_png, image_data, reference_path)
                reference_paths.append(str(reference_path))

            image_prompt = str(prompt or "").strip()
            if negative and str(negative).strip():
                image_prompt += f"\n\n请避免在图像中出现以下内容：{str(negative).strip()}"
            size = None
            if width and height:
                size = f"{int(width)}x{int(height)}"
            results = await asyncio.wait_for(
                generate_image(
                    prompt=image_prompt,
                    size=size,
                    n=1,
                    reference_images=reference_paths or None,
                    action="edit" if reference_paths else "generate",
                ),
                timeout=self.api.timeout,
            )
            if not results:
                return None, "Codex OAuth 图片接口没有返回图片。"
            output_path = Path(results[0].path)
            if not output_path.is_file():
                return None, "Codex OAuth 图片已返回，但找不到生成文件。"
            return output_path.read_bytes(), None
        except Exception as error:
            logger.error(f"[ComfyUI] Codex OAuth fallback failed ({type(error).__name__})")
            return None, "ComfyUI 当前不可达，GPT Image 回退失败；请检查授权状态和插件日志。"
        finally:
            for path in reference_paths:
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:
                    pass

    async def _record_image_result(self, task, prompt_id, output_node):
        task.comfy_prompt_id = prompt_id
        task.comfy_output_node = output_node
        if task.status == "cancelled":
            await self.api.cancel_prompt(prompt_id)
        else:
            self._save_image_resend(task)

    async def _run_post_upscale(self, event, task, data, label, upscale_arg):
        """Run the second stage while the caller still holds the local queue slot."""
        if task.status == "cancelled":
            return None
        if task.backend == "GPT Image":
            await self._send_text_message(
                event, "当前使用GPT image，无法使用放大流程，其它选项已提交"
            )
            return data

        task.comfy_prompt_id = None
        task.comfy_output_node = None
        task.progress = None
        self._clear_image_resend(task.id)

        async def record_upscale(prompt_id, output_node):
            await self._record_image_result(task, prompt_id, output_node)

        try:
            scale_factor, target_mp = self._parse_upscale_arg(upscale_arg)
            mp, fast, source_mp, rejection = await asyncio.to_thread(
                plan_tiled_upscale, data, scale_factor=scale_factor, target_mp=target_mp
            )
            if rejection:
                await self._send_text_message(event, f"{label}已完成（{rejection}）")
                return data

            mode_label = "8步加速" if fast else "25步精细"
            _, notify_submitted = self._make_queue_callbacks(
                event,
                f"{label}已完成，正在使用 Qwen 2.1 分块高清放大（{source_mp:.2f}MP → {mp:g}MP · {mode_label}）…",
            )

            async def on_submitted(prompt_id, position, ahead):
                if task.status != "cancelled":
                    await notify_submitted(prompt_id, position, ahead)

            upscaled = await tiled_upscale_image(
                self.api, data, megapixels=mp, fast=fast,
                on_submitted_callback=on_submitted, on_result_reference=record_upscale,
                on_progress=self._make_progress_callback(event, "高清放大", task=task),
                can_submit=lambda: task.status != "cancelled",
            )
            if task.status == "cancelled":
                return None
            if upscaled:
                return upscaled
        except Exception as error:
            logger.warning("[ComfyUI] %s后接分块放大失败：%s", label, type(error).__name__)
        if task.status == "cancelled":
            return None
        await self._send_text_message(event, f"⚠️ 分块高清放大失败，将发送{label}未放大的结果。")
        return data

    async def _generate_image(self, event, backend, prompt, negative=None, width=None, height=None,
                              scale=None, mode="generate", image_indices=None, image_paths=None, notify_progress=True,
                              lora_selection=None, reference_images=None, qwen_workflow_upscale=None,
                              qwen_workflow_upscale_only=False, qwen_edit_has_watermark=False,
                              edit_engine=None, edit_mask=None, mask_lora_defaults=None, native_mask_original=None,
                              post_upscale=None, edit_profile_command=None):
        """Queue image requests locally before dispatching to ComfyUI or GPT Image."""
        engine = self.txt2img if backend == "sd" else (edit_engine or self._qwen_engine)
        if engine is None:
            return None, None, "绘图功能未开启或工作流加载失败"
        passed, reject_msg = self._check_whitelist(event)
        if not passed:
            return None, None, reject_msg
        mask_removal = isinstance(engine, MaskedWorkflow) and engine.has_alternate_modes and edit_mask is not None
        if not prompt.strip() and not ((backend == "qwen" and qwen_workflow_upscale_only) or mask_removal):
            return None, None, "请输入提示词"

        limit_msg = self._check_user_queue_limit(event)
        if limit_msg:
            return None, None, limit_msg

        owner = self._image_task_owner(event)
        ticket, initial_ahead = await self._image_generation_queue.enqueue(
            owner, "动漫" if backend == "sd" else "Qwen"
        )
        try:
            if initial_ahead > 0 and notify_progress:
                try:
                    await self._send_text_message(
                        event, f"仍在排队等待中...（前面还有 {initial_ahead} 个任务）"
                    )
                except Exception as error:
                    logger.warning(f"[ComfyUI] 本地队列通知失败：{type(error).__name__}")
            if not await self._image_generation_queue.wait_turn(ticket):
                return ticket, None, None
            data, error = await self._generate_image_impl(
                event, backend, prompt, negative=negative, width=width, height=height,
                scale=scale, mode=mode, image_indices=image_indices, image_paths=image_paths,
                notify_progress=notify_progress, lora_selection=lora_selection,
                reference_images=reference_images,
                qwen_workflow_upscale=qwen_workflow_upscale,
                qwen_workflow_upscale_only=qwen_workflow_upscale_only,
                qwen_edit_has_watermark=qwen_edit_has_watermark, task=ticket, engine=engine, edit_mask=edit_mask,
                mask_lora_defaults=mask_lora_defaults,
                native_mask_original=native_mask_original,
                edit_profile_command=edit_profile_command,
            )
            if ticket.status == "cancelled":
                self._clear_image_resend(ticket.id)
                return ticket, None, None
            if data is not None and post_upscale is not None:
                data = await self._run_post_upscale(event, ticket, data, "改图", post_upscale)
            if ticket.status == "cancelled":
                self._clear_image_resend(ticket.id)
                return ticket, None, None
            if data is not None:
                ticket.generated = time.monotonic()
                ticket.status = "ready"
            elif ticket.status != "result_unavailable":
                ticket.finish("failed")
                self._clear_image_resend(ticket.id)
            return ticket, data, error
        finally:
            if ticket.status in ("queued", "generating"):
                ticket.finish("failed")
            await self._image_generation_queue.release(ticket)

    async def _generate_image_impl(self, event, backend, prompt, negative=None, width=None, height=None,
                              scale=None, mode="generate", image_indices=None, image_paths=None, notify_progress=True,
                              lora_selection=None, reference_images=None, qwen_workflow_upscale=None,
                              qwen_workflow_upscale_only=False, qwen_edit_has_watermark=False,
                              task=None, engine=None, edit_mask=None, mask_lora_defaults=None, native_mask_original=None,
                              edit_profile_command=None):
        used_codex_fallback = False
        if engine is None:
            return None, "绘图功能未开启或工作流加载失败"

        async def record_comfy_result(prompt_id, output_node):
            if task is not None:
                await self._record_image_result(task, prompt_id, output_node)

        images = []
        try:
            if backend == "qwen" and mode != "generate":
                if reference_images is not None:
                    images = list(reference_images)
                else:
                    images = await self._collect_images_from_event(
                        event, image_indices=image_indices, image_paths=image_paths,
                        use_recent_chat_image=mode == "edit",
                    )
                if mode == "edit" and not images:
                    return None, "编辑需要图片，请在当前消息附图或回复包含图片的消息"
                if len(images) > engine.MAX_REFERENCE_IMAGES:
                    return None, f"当前 ComfyUI 节点最多接收 {engine.MAX_REFERENCE_IMAGES} 张参考图"
            for image in images:
                safe, message = await self._check_img2img_input_censorship(event, image)
                if not safe:
                    return None, message
            mask_removal = isinstance(engine, MaskedWorkflow) and not prompt
            if not qwen_workflow_upscale_only and not mask_removal:
                prompt, message = await self._run_input_text_censorship(event, prompt)
                if prompt is None:
                    return None, message or "已被审查系统拒绝"
            if negative is None:
                negative = self.default_negative if backend == "sd" else ""
            if isinstance(engine, MaskedWorkflow):
                if not images or (edit_mask is None and not engine.has_alternate_modes):
                    return None, "当前编辑需要原图和有效遮罩"
                if not await self._is_comfyui_available():
                    return None, "ComfyUI 不可达，暂时无法执行此编辑流程。"
                if edit_mask is None:
                    progress_message = "正在整图重绘…"
                elif mask_removal:
                    progress_message = "正在移除选区内容…"
                else:
                    progress_message = "正在局部重绘…"
                _, progress = self._make_queue_callbacks(event, progress_message, notify=notify_progress)
                step_progress = self._make_progress_callback(
                    event, "局部重绘", notify=notify_progress, task=task
                )
                try:
                    data = await engine.generate(
                        native_mask_original.data if native_mask_original else images[0], edit_mask, prompt,
                        on_submitted_callback=progress, lora_selection=lora_selection,
                        lora_defaults=mask_lora_defaults,
                        reference_image=images[1] if len(images) == 2 else None,
                        image_extension=native_mask_original.extension if native_mask_original else "png",
                        on_result_reference=record_comfy_result,
                        on_progress=step_progress,
                    )
                except ComfyUIUnavailable:
                    return None, "ComfyUI 连接失败，局部重绘未提交。"
                if not data:
                    return None, "局部重绘没有返回图片，请检查 ComfyUI 日志。"
                safe, message = await self._check_output_censorship(event, data, "Qwen")
                return (data, None) if safe else (None, message)
            is_menu_edit = edit_profile_command is not None

            def allows_fallback():
                return self.codex_fallback_enabled and (
                    not is_menu_edit or self._edit_profile_allows_fallback(edit_profile_command)
                )

            unavailable_message = (
                "ComfyUI不在线噢，快让喵蛋去开启"
                if is_menu_edit else "无法连接 ComfyUI，任务未提交，请稍后重试"
            )
            check_health = is_menu_edit or self.codex_fallback_enabled
            comfy_available = await self._is_comfyui_available() if check_health else True
            if comfy_available:
                try:
                    progress_message = (
                        "正在使用 Qwen 2.1 分块高清放大原图…"
                        if backend == "qwen" and qwen_workflow_upscale_only
                        else self._image_progress_message(backend, images)
                    )
                    on_wait, on_submitted = self._make_queue_callbacks(
                        event, progress_message, notify=notify_progress
                    )
                    step_progress = self._make_progress_callback(
                        event, "高清放大" if qwen_workflow_upscale_only else "绘图",
                        notify=notify_progress, task=task
                    )
                    if backend == "qwen" and qwen_workflow_upscale_only:
                        mp, fast_turbo, source_mp, reject_reason = await asyncio.to_thread(plan_tiled_upscale, images[0])
                        if reject_reason:
                            return None, reject_reason
                        mp_label = int(mp) if mp == int(mp) else round(mp, 2)
                        src_label = round(source_mp, 2)
                        mode_label = "8步加速" if fast_turbo else "25步精细"
                        on_wait, on_submitted = self._make_queue_callbacks(
                            event,
                            f"正在使用 Qwen 2.1 分块高清放大（原图 {src_label}MP → 目标 {mp_label}MP · {mode_label}）…",
                            notify=notify_progress,
                        )
                        data = await tiled_upscale_image(
                            self.api, images[0], megapixels=mp, fast=fast_turbo,
                            on_submitted_callback=on_submitted,
                            on_result_reference=record_comfy_result,
                            on_progress=step_progress,
                        )
                    else:
                        kwargs = dict(width=width, height=height, scale=scale, max_wait=self.api.timeout,
                                      on_wait_callback=on_wait, on_submitted_callback=on_submitted,
                                      on_result_reference=record_comfy_result,
                                      on_progress=step_progress)
                        if backend == "qwen":
                            kwargs["image_data_list"] = images
                            kwargs["lora_selection"] = lora_selection
                            if qwen_workflow_upscale is not None:
                                kwargs["workflow_upscale"] = qwen_workflow_upscale
                            if qwen_workflow_upscale_only:
                                kwargs["workflow_upscale_only"] = True
                        data = await engine.generate(prompt, negative, **kwargs)
                except ComfyUIUnavailable:
                    comfy_available = False
            if not comfy_available:
                if not allows_fallback():
                    return None, unavailable_message
                if backend == "qwen" and (qwen_workflow_upscale or qwen_workflow_upscale_only):
                    if qwen_workflow_upscale_only or not prompt.strip():
                        return None, (
                            "当前使用 GPT Image，无法使用放大流程；本次只选择了“放大”，"
                            "没有其他编辑选项可执行。请恢复 ComfyUI 或选择其他编辑选项。"
                        )
                    try:
                        await self._send_text_message(
                            event, "当前使用GPT image，无法使用放大流程，其它选项已提交"
                        )
                    except Exception as warning_error:
                        logger.warning(
                            f"[ComfyUI] could not send upscale fallback notice ({type(warning_error).__name__})"
                        )
                    logger.info("[ComfyUI] skipped selected workflow upscale and will continue with GPT Image")
                if backend == "qwen" and qwen_edit_has_watermark:
                    try:
                        await self._send_text_message(
                            event, "GPT image版权受限，可能无法正常去除水印，请留意"
                        )
                    except Exception as warning_error:
                        logger.warning(
                            f"[ComfyUI] could not send watermark fallback notice ({type(warning_error).__name__})"
                        )
                if lora_selection:
                    try:
                        await self._send_text_message(
                            event, "GPT image无法使用LoRA，部分调整效果可能欠佳"
                        )
                    except Exception as warning_error:
                        logger.warning(
                            f"[ComfyUI] could not send LoRA fallback notice ({type(warning_error).__name__})"
                        )
                    logger.info("[ComfyUI] ignored selected LoRAs for GPT Image fallback")
                if task is not None:
                    task.backend = "GPT Image"
                logger.warning("[ComfyUI] unavailable; falling back to Codex OAuth image generation")
                if notify_progress:
                    try:
                        await self._send_text_message(
                            event, self._image_progress_message(backend, images, fallback=True)
                        )
                    except Exception as progress_error:
                        logger.warning(
                            f"[ComfyUI] could not send GPT Image progress ({type(progress_error).__name__})"
                        )
                if not allows_fallback():
                    return None, unavailable_message
                data, fallback_error = await self._generate_with_codex(
                    prompt, negative=negative, width=width, height=height, images=images
                )
                if data is None:
                    return None, fallback_error
                used_codex_fallback = True
        except ComfyUIExecutionError as error:
            if task is not None and task.status == "cancelled":
                return None, None
            return None, error.user_message
        except ComfyUIResultUnavailable:
            if task is not None and task.status == "cancelled":
                return None, None
            if task is not None and task.comfy_prompt_id:
                task.finish("result_unavailable")
                return None, f"暂时未能取回 ComfyUI 图片。可用 /重发图片 {task.id} 从历史取回，无需重新生成。"
            return None, "ComfyUI 已受理任务，但未取得图片。请查看 ComfyUI 的队列或历史记录。"
        except ComfyUISubmissionUncertain as error:
            logger.warning(f"[ComfyUI] 无法确认任务是否已受理：{error}")
            return None, "暂时无法确认 ComfyUI 是否已受理任务。请先查看 ComfyUI 队列，避免重复提交。"
        except ValueError as error:
            return None, str(error)
        except (aiohttp.ClientError, TimeoutError) as error:
            logger.error(f"[ComfyUI] 绘图连接失败：{error}")
            return None, "ComfyUI 连接失败，请稍后重试"
        if not data:
            if task is not None and task.status == "cancelled":
                return None, None
            return None, "生成失败，请检查 ComfyUI 日志"
        source_name = (
            "Codex OAuth GPT Image" if used_codex_fallback
            else ("ComfyUI 原图放大" if qwen_workflow_upscale_only
                  else ("文生图" if backend == "sd" else "Qwen"))
        )
        safe, message = await self._check_output_censorship(event, data, source_name)
        if safe:
            await self._remember_reroll(event, {
                "kind": "image",
                "backend": backend,
                "prompt": prompt,
                "negative": negative,
                "width": width,
                "height": height,
                "scale": scale,
                "mode": "edit" if images else mode,
                "lora_selection": dict(lora_selection) if isinstance(lora_selection, dict) else lora_selection,
                "reference_images": list(images) if images else None,
                "qwen_workflow_upscale": qwen_workflow_upscale,
                "qwen_workflow_upscale_only": qwen_workflow_upscale_only,
                "qwen_edit_has_watermark": qwen_edit_has_watermark,
                "edit_engine": engine if engine not in (self.txt2img, self._qwen_engine) else None,
                "edit_mask": edit_mask,
                "mask_lora_defaults": mask_lora_defaults,
                "native_mask_original": native_mask_original,
                "edit_profile_command": edit_profile_command,
            })
        return (data, None) if safe else (None, message)

    async def _generate_video(self, event, prompt="", negative="", image_index=None, fps=None, length=None, image_path=None, notify_progress=True, *, image_bytes=None):
        if self._img2video_engine is None:
            return None, "图生视频功能未开启或工作流加载失败"
        passed, reject_msg = self._check_whitelist(event)
        if not passed:
            return None, reject_msg
        try:
            images = [image_bytes] if image_bytes is not None else await self._collect_images_from_event(event, take_first_only=True,
                image_indices=[image_index] if image_index is not None and not image_path else None,
                image_paths=[image_path] if image_path else None, use_recent_chat_image=True)
            if not images:
                return None, ("请先在当前会话上传图片，再发送图生视频指令"
                              if event.get_platform_name() == "synochat_adapter"
                              else "图生视频需要图片，请在当前消息附图或回复包含图片的消息")
            safe, message = await self._check_img2img_input_censorship(event, images[0])
            if not safe:
                return None, message
            prompt, message = await self._run_input_text_censorship(event, prompt)
            if prompt is None:
                return None, message or "已被审查系统拒绝"
            limit_msg = self._check_user_queue_limit(event)
            if limit_msg:
                return None, limit_msg
            owner = self._image_task_owner(event)
            ticket, initial_ahead = await self._image_generation_queue.enqueue(owner, "视频")
            try:
                if initial_ahead > 0 and notify_progress:
                    await self._send_text_message(
                        event, f"仍在排队等待中...（前面还有 {initial_ahead} 个任务）"
                    )
                if not await self._image_generation_queue.wait_turn(ticket):
                    return None, "已取消排队任务"
                async def record_video(prompt_id):
                    ticket.comfy_prompt_id = prompt_id

                _, on_submitted = self._make_queue_callbacks(event, "正在生成视频...", notify=notify_progress)
                step_progress = self._make_progress_callback(
                    event, "视频生成", notify=notify_progress, task=ticket, min_interval=45.0
                )
                data = await self._img2video_engine.generate(images[0], prompt, negative, fps=fps, length=length,
                    max_wait=self.api.timeout, on_submitted_callback=on_submitted,
                    on_progress=step_progress, on_result_reference=record_video)
                if ticket.status == "cancelled":
                    return None, None
                if data:
                    ticket.generated = time.monotonic()
                    ticket.status = "ready"
                    await self._remember_reroll(event, {
                        "kind": "video",
                        "image_bytes": images[0],
                        "prompt": prompt,
                        "negative": negative,
                        "fps": fps,
                        "length": length,
                    })
                else:
                    ticket.finish("failed")
            finally:
                if ticket.status in ("queued", "generating"):
                    ticket.finish("failed")
                await self._image_generation_queue.release(ticket)
        except ComfyUIExecutionError as error:
            return None, error.user_message
        except ValueError as error:
            return None, str(error)
        except (aiohttp.ClientError, TimeoutError) as error:
            logger.error(f"[ComfyUI] 视频连接失败：{error}")
            return None, "ComfyUI 连接失败，请稍后重试"
        return ((ticket, data), None) if data else (None, "生成失败，请检查 ComfyUI 日志")

    async def _get_image_tags(self, event, image_index=None, image_path=None, notify_progress=True):
        if self.img2txt is None:
            return None, "Tagger 功能未开启或工作流加载失败"
        passed, reject_msg = self._check_whitelist(event)
        if not passed:
            return None, reject_msg
        try:
            images = await self._collect_images_from_event(event, take_first_only=True,
                image_indices=[image_index] if image_index is not None and not image_path else None,
                image_paths=[image_path] if image_path else None, use_recent_chat_image=True)
            if not images:
                return None, ("请先在当前会话上传图片，再发送标签指令"
                              if event.get_platform_name() == "synochat_adapter"
                              else "提取标签需要图片，请在当前消息附图或回复包含图片的消息")
            owner = self._image_task_owner(event)
            ticket, initial_ahead = await self._image_generation_queue.enqueue(
                owner, "标签反推", model_group="tagger"
            )
            try:
                if initial_ahead > 0 and notify_progress:
                    await self._send_text_message(
                        event, f"标签识别排队等待中...（前面还有 {initial_ahead} 个任务）"
                    )
                if not await self._image_generation_queue.wait_turn(ticket):
                    return None, "已取消排队任务"
                on_wait, on_submitted = self._make_queue_callbacks(event, "正在识别图片标签...", notify=notify_progress)
                text = await self.img2txt.generate(images[0], max_wait=self.api.timeout,
                    on_wait_callback=on_wait, on_submitted_callback=on_submitted)
                if text:
                    ticket.generated = time.monotonic()
                    ticket.finish("sent")
                else:
                    ticket.finish("failed")
            finally:
                if ticket.status in ("queued", "generating"):
                    ticket.finish("failed")
                await self._image_generation_queue.release(ticket)
        except ComfyUIExecutionError as error:
            return None, error.user_message
        except ValueError as error:
            return None, str(error)
        except (aiohttp.ClientError, TimeoutError) as error:
            logger.error(f"[ComfyUI] Tagger 连接失败：{error}")
            return None, "ComfyUI 连接失败，请稍后重试"
        return (text.replace("_", " "), None) if text else (None, "标签识别失败，请检查 ComfyUI 日志")

    async def tool_generate_image(self, event: AstrMessageEvent, prompt: str, negative=None, width=None, height=None, scale=None, lora=None):
        lora_selection, lora_error = self._parse_tool_lora_selection(lora)
        if lora_error:
            return await self._send_tool_image(event, None, lora_error)
        task, data, error = await self._generate_image(
            event, "qwen", prompt, negative, width, height, scale,
            notify_progress=True, lora_selection=lora_selection,
        )
        return await self._send_tool_image(event, data, error, task)

    async def tool_edit_image(self, event: AstrMessageEvent, prompt: str, negative=None, image_indices=None, width=None, height=None, scale=None, image_paths=None, lora=None):
        lora_selection, lora_error = self._parse_tool_lora_selection(lora)
        if lora_error:
            return await self._send_tool_image(event, None, lora_error)
        task, data, error = await self._generate_image(
            event, "qwen", prompt, negative, width, height, scale,
            mode="edit", image_indices=image_indices, image_paths=image_paths,
            notify_progress=True, lora_selection=lora_selection,
        )
        return await self._send_tool_image(event, data, error, task)

    async def tool_image_to_video(self, event: AstrMessageEvent, prompt: str = "", negative: str = "", image_index: int = 1, fps=None, length=None, image_path=None):
        generated, error = await self._generate_video(
            event, prompt, negative, image_index, fps, length, image_path=image_path,
            notify_progress=event.get_platform_name() == "synochat_adapter",
        )
        result = {"status": "failed", "media_type": "video", "media_sent": False, "file_path": None, "feedback_hint": COMPLETION_FEEDBACK_HINT}
        if generated is None:
            result["message"] = error or "请求未执行：当前会话无使用权限"
            return json.dumps(result, ensure_ascii=False)
        ticket, data = generated
        path = self.temp_dir / f"{uuid.uuid4().hex}.mp4"
        try:
            await asyncio.to_thread(path.write_bytes, data)
            result["file_path"] = str(path.resolve())
            await self._send_generated_video(event, path)
        except Exception as error:
            ticket.finish("delivery_failed" if result["file_path"] else "failed")
            logger.error(f"[ComfyUI] 工具视频发送失败：{error}")
            result["status"] = "delivery_failed" if result["file_path"] else "failed"
            result["message"] = "视频已生成，但保存或发送失败。请检查平台日志，不要重复提交生成任务。"
            return json.dumps(result, ensure_ascii=False)
        ticket.finish("sent")
        result.update(status="success", media_sent=True, message="视频已生成，结果已发送到当前会话。")
        return json.dumps(result, ensure_ascii=False)

    async def tool_image_tagger(self, event: AstrMessageEvent, image_index: int = 1, image_path=None):
        text, error = await self._get_image_tags(event, image_index, image_path=image_path, notify_progress=False)
        return text if text is not None else error or "请求未执行：当前会话无使用权限"
