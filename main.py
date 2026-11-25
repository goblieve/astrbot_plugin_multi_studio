import asyncio
import json
import re
import stat
import sys
import time
import uuid
from functools import wraps
from astrbot.api import AstrBotConfig
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.event import filter
from astrbot.api.message_components import Plain
from astrbot.api.message_components import Reply
from astrbot.api.star import Context
from astrbot.api.star import Star
from pathlib import Path
from typing import Optional
from .comfyui_api import ComfyUIAPI
from .draw_tasks import LocalImageGenerationQueue, TASK_LABELS
from .image_to_text import ImageToText
from .image_to_video import ImageToVideo
from .llm_tools import create_tools
from .mask_editor import MaskEditor
from .qwen_image import QwenImage
from .text_to_image import TextToImage
from .workflow_library import WorkflowLibrary
from .hub_common import (
    CONTINUE_EDIT_ALIASES, CUTOUT_COMMAND_ALIASES, DEFAULT_DRAW_EDIT_OPTION_ROWS, DEFAULT_DRAW_LORA_ALIASES,
    DRAW_ALIASES, EDIT_OPTION_TTL_SECONDS, GENERATED_MEDIA_CLEANUP_INTERVAL, GENERATED_MEDIA_FILENAME,
    GENERATED_MEDIA_TTL_SECONDS, IMAGE_RESEND_FILENAME, IMG2VIDEO_ALIASES, MULTIVIEW_COMMAND_ALIASES,
    SD_DRAW_ALIASES, UPSCALE_COMMAND_ALIASES, _strip_command_prefix,
)
from .hub_edit_options import EditOptionsMixin
from .hub_webapi import WebApiMixin
from .hub_moderation import ModerationMixin
from .hub_messaging import MessagingMixin
from .hub_prompting import PromptingMixin
from .hub_generation import GenerationMixin


def _stop_matched_command(handler):
    """Prevent handled commands from falling through to the chat LLM."""
    @wraps(handler)
    async def wrapped(self, event: AstrMessageEvent, *args, **kwargs):
        try:
            async for result in handler(self, event, *args, **kwargs):
                yield result
        finally:
            event.stop_event()

    return wrapped


class ComfyUIHub(EditOptionsMixin, WebApiMixin, ModerationMixin, MessagingMixin, PromptingMixin, GenerationMixin, Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._image_generation_queue = LocalImageGenerationQueue()

        self.default_negative = config.get("default_negative_prompt", "")
        self.default_chain = config.get("default_chain", False)
        self.draw_lora_aliases = config.get("draw_lora_aliases", DEFAULT_DRAW_LORA_ALIASES)
        self.draw_prompt_presets = self._load_draw_prompt_presets(config.get("draw_prompt_presets", []))
        self.draw_edit_command_profiles = self._parse_draw_edit_command_profiles(
            config.get("draw_edit_command_profiles", []),
            config.get("draw_edit_options", DEFAULT_DRAW_EDIT_OPTION_ROWS),
        )
        self._pending_draw_edit_options = {}
        self._pending_draw_edit_options_lock = asyncio.Lock()
        self._last_generation_params = {}

        plugin_dir = Path(__file__).parent
        data_root = plugin_dir.parent.parent / "plugin_data"
        data_dir = data_root / "astrbot_plugin_multi_studio"
        data_dir.mkdir(parents=True, exist_ok=True)
        self._previous_edit_profiles = data_dir / "draw_edit_profiles_previous.json"
        self._multiview_prompts_file = data_dir / "multiview_prompts.json"
        self.multiview_prompts = self._load_multiview_prompts()
        self._edit_profiles_lock = asyncio.Lock()

        workflow_dir = data_dir / "workflows"
        workflow_dir.mkdir(exist_ok=True)

        self.temp_dir = data_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)
        self._resending_images = set()
        self._cleanup_orphaned_draw_edit_option_files()
        self._generated_media_cleanup_task: Optional[asyncio.Task] = None

        self.block_tags_file = data_dir / "block_tags.json"
        self.output_block_tags_file = data_dir / "output_block_tags.json"
        self.blocked_users_file = data_dir / "blocked_users.json"
        self.censorship_config_file = data_dir / "censorship_config.json"
        self.sent_messages_file = data_dir / "sent_messages.json"
        self._load_block_data()

        server_url = config.get("server_url", "http://127.0.0.1:8188")
        timeout = config.get("timeout", 300)
        self.api = ComfyUIAPI(server_url, timeout)
        self._workflow_library = WorkflowLibrary(data_dir / "workflow_library", self.api, self.draw_lora_aliases)
        mask_url = config.get("mask_editor_public_url", "")
        self._mask_editor = MaskEditor(
            self.api, plugin_dir / "mask_view", str(mask_url or "").strip(), 6192,
            cache_directory=data_dir / "original_cache",
            generation_queue=self._image_generation_queue,
        )
        self.codex_fallback_enabled = bool(config.get("codex_fallback_enabled", True))
        self.codex_fallback_provider_id = str(
            config.get("codex_fallback_provider_id", "oauth_plug_openai_codex")
        ).strip() or "oauth_plug_openai_codex"
        try:
            health_timeout = float(config.get("codex_fallback_health_timeout", 3))
            self.codex_fallback_health_timeout = health_timeout if 0.5 <= health_timeout <= 10 else 3.0
        except (TypeError, ValueError, OverflowError):
            self.codex_fallback_health_timeout = 3.0

        self.txt2img = self._init_engine(TextToImage, "enable_txt2img", "sd_workflow", "example_sd_image.json", workflow_dir)
        self.img2txt = self._init_engine(ImageToText, "enable_tagger", "tagger_workflow", "example_tagger.json", workflow_dir)
        self._qwen_engine = self._init_engine(
            QwenImage, "enable_qwen", "qwen_workflow", "example_qwen_image.json", workflow_dir,
            lora_aliases=self.draw_lora_aliases,
        )
        self._img2video_engine = self._init_engine(
            ImageToVideo, "enable_img2video", "video_workflow", "example_wan_video.json", workflow_dir,
            max_frames=int(config.get("img2video_max_frames", 240)),
        )

        # 输入审查
        self.enable_input_censorship = config.get("enable_input_censorship", True)
        self.input_censorship_use_llm = config.get("input_censorship_use_llm", True)
        self.censorship_prompt = config.get("censorship_prompt", "")
        self.llm_provider_id = config.get("llm_provider_id", "")
        self.admin_bypass_censorship = config.get("admin_bypass_censorship", True)
        self.censorship_failure_mode = config.get("censorship_failure_mode", "fail_open")

        # 输出图片审查
        self.enable_output_censorship = config.get("enable_output_censorship", False)
        self.output_censorship_use_llm = config.get("output_censorship_use_llm", True)
        self.output_censorship_use_tagger = config.get("output_censorship_use_tagger", True)
        self.output_censorship_prompt = config.get("output_censorship_prompt", "")

        # 图生图输入图片审查
        self.enable_img2img_input_censorship = config.get("enable_img2img_input_censorship", False)
        self.img2img_input_censorship_use_llm = config.get("img2img_input_censorship_use_llm", True)

        # 图生图输出图片审查
        self.enable_img2img_output_censorship = config.get("enable_img2img_output_censorship", False)
        self.img2img_output_censorship_use_llm = config.get("img2img_output_censorship_use_llm", True)
        self.img2img_output_censorship_use_tagger = config.get("img2img_output_censorship_use_tagger", True)

        # 白名单
        self.enable_group_whitelist = config.get("enable_group_whitelist", False)
        self.enable_user_whitelist = config.get("enable_user_whitelist", False)
        self.whitelist_groups = {str(g).strip() for g in config.get("whitelist_groups", []) if str(g).strip()}
        self.whitelist_users = {str(u).strip() for u in config.get("whitelist_users", []) if str(u).strip()}
        self.whitelist_admin_bypass = config.get("whitelist_admin_bypass", True)
        self.whitelist_reject_message = config.get(
            "whitelist_reject_message",
            "⚠️ 当前会话未在白名单内，绘图服务暂不可用。",
        )

        self.context.add_llm_tools(*create_tools(self))
        self._register_draw_edit_options_web_api()

    async def initialize(self):
        try:
            await self._mask_editor.start()
        except (OSError, ValueError) as error:
            logger.warning("[ComfyUI] 网页涂抹服务启动失败：%s", type(error).__name__)
        self._generated_media_cleanup_task = asyncio.create_task(
            self._run_generated_media_cleanup()
        )

    async def terminate(self):
        await self._mask_editor.close()
        if self._generated_media_cleanup_task is not None:
            self._generated_media_cleanup_task.cancel()
            await asyncio.gather(self._generated_media_cleanup_task, return_exceptions=True)
            self._generated_media_cleanup_task = None
        flush_task = getattr(self, "_sent_messages_flush_task", None)
        if flush_task is not None and not flush_task.done():
            flush_task.cancel()
            await asyncio.gather(flush_task, return_exceptions=True)
        await self._flush_sent_messages()
        await self._close_download_session()
        await self.api.close()

    async def _run_generated_media_cleanup(self):
        while True:
            try:
                removed = await asyncio.to_thread(self._delete_expired_generated_media)
                if removed:
                    logger.info(f"[ComfyUI] 已清理 {removed} 个超过 7 天的临时图片或视频")
            except OSError as error:
                logger.warning(f"[ComfyUI] 无法扫描生成文件目录：{type(error).__name__}")
            await asyncio.sleep(GENERATED_MEDIA_CLEANUP_INTERVAL)

    def _delete_expired_generated_media(self) -> int:
        cutoff = time.time() - GENERATED_MEDIA_TTL_SECONDS
        removed = 0
        for path in self.temp_dir.iterdir():
            if not (GENERATED_MEDIA_FILENAME.fullmatch(path.name) or IMAGE_RESEND_FILENAME.fullmatch(path.name)):
                continue
            try:
                info = path.lstat()
                if stat.S_ISREG(info.st_mode) and info.st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except FileNotFoundError:
                continue
            except OSError as error:
                logger.warning(f"[ComfyUI] 无法清理过期生成文件：{type(error).__name__}")
        return removed

    @filter.event_message_type(filter.EventMessageType.ALL, priority=sys.maxsize - 2)
    async def handle_draw_edit_option_interaction(self, event: AstrMessageEvent):
        """Handle quoted QQ edit interactions and unquoted Synology Chat replies."""
        platform = event.get_platform_name()
        chat = platform == "synochat_adapter"
        if platform != "aiocqhttp" and not chat:
            return

        if not chat:
            try:
                self._remember_recent_file(event)
            except Exception as error:
                logger.warning("[ComfyUI] 记录最近文件失败：%s", type(error).__name__)
        await self._cleanup_pending_draw_edit_options()
        messages = event.get_messages() or []
        reply = next((item for item in messages if isinstance(item, Reply)), None)
        plain_parts = [item.text for item in messages if isinstance(item, Plain)]
        text = "".join(plain_parts).strip()
        if not text:
            text = str(getattr(event, "message_str", "") or "").strip()

        if text.strip().casefold() in {"再来一张", "重roll", "重画", "reroll"}:
            event.stop_event()
            await self._execute_reroll(event)
            return

        profile = self._match_draw_edit_command(text, self.draw_edit_command_profiles)
        if chat:
            has_image = self._remember_chat_image(event, messages)
            if profile is not None:
                event.stop_event()
                await self._start_draw_edit_option_menu(event, profile)
            elif text and not has_image and not text.startswith("/"):
                key = self._draw_edit_pending_key(event, None)
                if text == "取消":
                    async with self._pending_draw_edit_options_lock:
                        record = self._pending_draw_edit_options.pop(key, None)
                    if record is not None:
                        event.stop_event()
                        self._delete_draw_edit_record_files(record)
                        await self._send_text_message(event, "已取消本次改图。")
                else:
                    await self._handle_draw_edit_selection(event, key, text)
            return

        allow_image_files = bool(
            profile and (
                profile.get("direct_editor")
                or profile.get("direct_cutout")
                or profile.get("direct_upscale")
                or profile.get("multiview_menu")
            )
        )
        if reply is None:
            if profile is not None and (
                allow_image_files
                or any(self._is_edit_image(item) for item in messages)
                or self._recent_generated_image_bytes(event) is not None
            ):
                event.stop_event()
                await self._start_draw_edit_option_menu(event, profile)
            return

        pending_key = self._draw_edit_pending_key(event, reply.id)
        async with self._pending_draw_edit_options_lock:
            prompt_record = self._pending_draw_edit_options.get(pending_key)
            awaiting_prompt = bool(
                prompt_record
                and (prompt_record.get("awaiting_placeholders") or prompt_record.get("awaiting_placeholder"))
            )
        if awaiting_prompt:
            event.stop_event()
            await self._continue_draw_edit_prompt_collection(event, pending_key, prompt_record, text)
            return

        quoted_chain = reply.chain if reply.chain else []
        if "改" in text and len(text) <= 80:
            quoted_types = [type(item).__name__ for item in quoted_chain]
            matched_command = profile.get("command") if profile else None
            has_quoted_image = any(self._is_edit_image(item, allow_image_files) for item in quoted_chain)
            logger.info(
                f"[ComfyUI] 引用改图分流诊断：群聊={bool(event.get_group_id())}，"
                f"文本={text!r}，匹配指令={matched_command!r}，"
                f"引用组件={quoted_types}，含图片={has_quoted_image}"
            )
        if profile is not None:
            if allow_image_files or any(self._is_edit_image(item) for item in quoted_chain):
                event.stop_event()
                await self._start_draw_edit_option_menu(event, profile)
                return
            event.stop_event()
            await self._send_text_message(
                event,
                "引用消息里没有识别到图片。请直接引用原始图片消息后再发送改图指令。",
            )
            return

        await self._handle_draw_edit_selection(event, pending_key, text)

    async def _handle_draw_edit_selection(self, event, pending_key, text):
        async with self._pending_draw_edit_options_lock:
            record = self._pending_draw_edit_options.get(pending_key)
            if record is None:
                return
            awaiting_placeholder = bool(
                record.get("awaiting_placeholders") or record.get("awaiting_placeholder")
            )
            if not awaiting_placeholder:
                if str(text or "").strip() == "取消":
                    self._pending_draw_edit_options.pop(pending_key, None)
                    cancel_menu = True
                    options, prompt_values, additional_prompt, selection_error = None, {}, "", None
                else:
                    cancel_menu = False
                    options, prompt_values, additional_prompt, selection_error = self._parse_draw_edit_option_selection(
                        text,
                        record.get("options", []),
                    )
                    if selection_error:
                        pass
                    elif not options:
                        return
                    else:
                        # 保留菜单记录 10 分钟内可多次点选，仅刷新有效期
                        record["expires_at"] = time.monotonic() + EDIT_OPTION_TTL_SECONDS
            else:
                cancel_menu = False

        event.stop_event()
        if cancel_menu:
            self._delete_draw_edit_record_files(record)
            await self._send_text_message(event, "已关闭本次改图菜单。")
            return
        if awaiting_placeholder:
            await self._continue_draw_edit_prompt_collection(event, pending_key, record, text)
            return
        if selection_error:
            await self._send_text_message(event, selection_error)
            return

        selected_options = [dict(option) for option in options]
        if record.get("is_multiview"):
            inline_mode5 = prompt_values.get("prompt", "")
            for opt in selected_options:
                if opt.get("multiview_mode") == 5:
                    opt["prompt"] = inline_mode5
            await self._run_draw_edit_options(event, record, selected_options, additional_prompt)
            return
        placeholders = self._draw_edit_prompt_placeholders(selected_options)
        if placeholders:
            missing_placeholders = [
                item for item in placeholders
                if item["key"] not in prompt_values
            ]
            if missing_placeholders:
                sub_record = {
                    **record,
                    "selected_options": selected_options,
                    "prompt_placeholders": placeholders,
                    "prompt_values": dict(prompt_values),
                    "additional_prompt": additional_prompt,
                }
                await self._ask_draw_edit_prompt_placeholders(event, sub_record, missing_placeholders)
                return
            self._substitute_draw_edit_prompt_values(selected_options, prompt_values)
            await self._run_draw_edit_options(event, record, selected_options, additional_prompt)
            return

        await self._run_draw_edit_options(event, record, selected_options, additional_prompt)

    def _init_engine(self, engine, enabled_key, workflow_key, builtin, workflow_dir, **options):
        if not self.config.get(enabled_key, True):
            return None
        filename = self.config.get(workflow_key, "").strip()
        try:
            if filename:
                path = (workflow_dir / filename).resolve()
                if path.parent != workflow_dir.resolve():
                    raise ValueError("自定义工作流必须位于 workflows 目录内")
            else:
                path = Path(__file__).parent / "workflows" / builtin
            return engine(self.api, str(path), **options)
        except (OSError, ValueError) as error:
            logger.error(f"[ComfyUI] {workflow_key} 工作流加载失败：{error}")
            return None

    @filter.command("draw_sd")
    @_stop_matched_command
    async def draw_sd(self, event: AstrMessageEvent):
        """SD 动漫文生图。"""
        passed, reject_msg = self._check_whitelist(event)
        if not passed:
            if reject_msg:
                yield event.plain_result(reject_msg)
            return
        text = _strip_command_prefix(event.message_str.strip(), SD_DRAW_ALIASES)
        handled, message = await self._handle_admin_subcommand(event, text)
        if handled:
            if message:
                yield event.plain_result(message)
            return
        positive, negative, chain, width, height, scale = self._parse_params(text)
        task, data, error = await self._generate_image(event, "sd", positive, negative, width, height, scale)
        if data is None:
            if error:
                yield event.plain_result(error)
            return
        result = await self._deliver_generated_image(event, task, data, chain)
        for notice in self._image_delivery_notices(result):
            yield event.plain_result(notice)
        event.stop_event()

    @filter.command("delete", alias={'撤回', 'recall'})
    @_stop_matched_command
    async def delete_msg(self, event: AstrMessageEvent):
        """引用撤回绘图功能输出的消息"""
        chain = event.get_messages()
        if not chain:
            return

        first_seg = chain[0] if len(chain) > 0 else None
        if not first_seg:
            return

        if event.get_platform_name() != "aiocqhttp":
            yield event.plain_result("当前平台无法通过绘图指令撤回已发送的消息。可用 /取消绘图 取消尚未完成的任务。")
            return

        if not isinstance(first_seg, Reply):
            yield event.plain_result("❌ 请引用要撤回的绘图消息")
            return

        target_msg_id = str(first_seg.id)
        group_id = event.get_group_id()
        sender_id = event.get_sender_id()
        current_time = time.time()
        is_admin = event.is_admin()

        cache_key = str(group_id) if group_id else str(sender_id) if sender_id else None
        is_valid_message = is_admin
        matched_record = None

        if not is_admin and cache_key:
            sent_msgs = self.sent_messages.get(cache_key, [])
            valid_msgs = []
            for msg_data in sent_msgs:
                if not isinstance(msg_data, dict):
                    continue
                if current_time - msg_data.get('timestamp', 0) > self.message_cache_ttl:
                    continue
                valid_msgs.append(msg_data)
                if msg_data.get('message_id') == target_msg_id:
                    is_valid_message = True
                    matched_record = msg_data
            self.sent_messages[cache_key] = valid_msgs

        if not is_valid_message:
            return

        try:
            await event.bot.delete_msg(message_id=int(target_msg_id))
            if matched_record is not None and cache_key:
                try:
                    self.sent_messages[cache_key].remove(matched_record)
                except ValueError:
                    pass
                self._save_sent_messages()
            event.stop_event()
        except Exception as e:
            logger.error(f"撤回失败: {e}")

    @filter.command("tagger", alias={'tag', '标签'})
    @_stop_matched_command
    async def tagger(self, event: AstrMessageEvent):
        text, error = await self._get_image_tags(event)
        if text is None:
            if error:
                yield event.plain_result(error)
            return
        yield event.plain_result(f"【标签】\n{text}")
        event.stop_event()

    @filter.command("lora清单", alias={"可用lora"})
    @_stop_matched_command
    async def cmd_lora_list(self, event: AstrMessageEvent):
        """列出当前 Qwen /draw 工作流中可按别名选择的 LoRA。"""
        engine = self._qwen_engine
        if engine is None:
            yield event.plain_result("Qwen /draw 工作流未启用或加载失败，无法读取可用 LoRA。")
            event.stop_event()
            return

        loader_nodes = [
            node for node in engine.template.workflow.values()
            if isinstance(node, dict)
            and node.get("class_type") == "VisualLoRALoader"
            and isinstance(node.get("inputs"), dict)
        ]
        if len(loader_nodes) != 1:
            yield event.plain_result(
                "当前 Qwen 工作流必须包含且仅包含一个 VisualLoRALoader 节点，才能列出可用 LoRA。"
            )
            event.stop_event()
            return

        try:
            entries = json.loads(loader_nodes[0]["inputs"].get("lora_data", ""))
        except (TypeError, json.JSONDecodeError):
            entries = None
        if not isinstance(entries, list):
            yield event.plain_result("当前 Qwen 工作流的 VisualLoRALoader 数据无效，无法读取可用 LoRA。")
            event.stop_event()
            return

        workflow_files = {}
        for entry in entries:
            if isinstance(entry, dict) and entry.get("name"):
                filename = engine._lora_basename(entry["name"])
                workflow_files[filename] = workflow_files.get(filename, 0) + 1

        display_names = {}
        configured_rows = self.draw_lora_aliases
        if isinstance(configured_rows, str):
            configured_rows = [configured_rows]
        for row in configured_rows or []:
            if not isinstance(row, str) or "=" not in row:
                continue
            alias, filename = row.split("=", 1)
            alias, filename = alias.strip(), filename.strip()
            key = alias.casefold()
            if key in engine.lora_aliases and engine._lora_basename(filename) == engine._lora_basename(engine.lora_aliases[key]):
                display_names.setdefault(key, alias)

        available = []
        for key, filename in sorted(engine.lora_aliases.items()):
            basename = engine._lora_basename(filename)
            label = display_names.get(key, key)
            if workflow_files.get(basename, 0) == 1:
                available.append((label, filename))

        if not engine.lora_aliases:
            yield event.plain_result("当前没有配置 LoRA 别名。请在插件配置的 draw_lora_aliases 中添加。")
            event.stop_event()
            return

        aliases = [alias for alias, _filename in available]
        if aliases:
            lines = [f"Qwen /draw 可用 LoRA 别名（{len(aliases)} 项）：", "、".join(aliases)]
        else:
            lines = ["当前工作流中没有可用的 LoRA 别名。"]
        lines.append("")
        lines.append("用法：/draw --lora 别名 提示词；强度写成 别名=0.8；--lora off 关闭全部。")
        yield event.plain_result("\n".join(lines))
        event.stop_event()

    @filter.command("提示词预设")
    @_stop_matched_command
    async def cmd_draw_prompt_preset_list(self, event: AstrMessageEvent):
        """只列出 /draw 固定提示词预设名称，不显示模板内容。"""
        presets = sorted(
            (name for name, _template in self.draw_prompt_presets.values()),
            key=str.casefold,
        )
        if presets:
            lines = [f"当前可用提示词预设（{len(presets)} 项）："]
            lines.extend(f"• {name}" for name in presets)
            lines.extend(["", "用法：/draw 预设=名称 提示词"])
            message = "\n".join(lines)
        else:
            message = "当前没有配置提示词预设。"

        yield event.plain_result(message)
        event.stop_event()

    @filter.command("draw")
    @_stop_matched_command
    async def draw(self, event: AstrMessageEvent):
        """生成或编辑图片，群晖 Chat 支持使用最近上传的参考图。"""
        passed, reject_msg = self._check_whitelist(event)
        if not passed:
            if reject_msg:
                yield event.plain_result(reject_msg)
            return
        text = _strip_command_prefix(event.message_str.strip(), DRAW_ALIASES)
        handled, message = await self._handle_admin_subcommand(event, text)
        if handled:
            if message:
                yield event.plain_result(message)
            return

        text, aspect_ratio, ratio_error = self._extract_draw_aspect_ratio(text)
        if ratio_error:
            yield event.plain_result(ratio_error)
            event.stop_event()
            return

        text, preset_name, preset_error = self._extract_draw_preset(text)
        if preset_error:
            yield event.plain_result(preset_error)
            event.stop_event()
            return

        if preset_name is not None:
            preset = self.draw_prompt_presets.get(preset_name.casefold())
            if preset is None:
                names = ", ".join(name for name, _template in sorted(self.draw_prompt_presets.values(), key=lambda item: item[0].casefold()))
                message = f"未找到提示词预设“{preset_name}”。"
                message += f"可用预设：{names}" if names else "当前没有配置预设。"
                yield event.plain_result(message)
                event.stop_event()
                return

            preset_display_name, preset_template = preset
            preset_text, preset_lora_selection, preset_lora_error = self._parse_draw_lora_args(preset_template)
            if preset_lora_error:
                yield event.plain_result(f"预设“{preset_display_name}”中的 LoRA 设置有误：{preset_lora_error}")
                event.stop_event()
                return

            user_text, command_lora_selection, command_lora_error = self._parse_draw_lora_args(text)
            if command_lora_error:
                yield event.plain_result(command_lora_error)
                event.stop_event()
                return

            if "{prompt}" in preset_text:
                text = preset_text.replace("{prompt}", user_text).strip()
            else:
                text = " ".join(part for part in (preset_text, user_text) if part).strip()
            lora_selection = (
                command_lora_selection
                if command_lora_selection is not None
                else preset_lora_selection
            )
        else:
            text, lora_selection, lora_error = self._parse_draw_lora_args(text)
            if lora_error:
                yield event.plain_result(lora_error)
                event.stop_event()
                return

        text, preset_aspect_ratio, ratio_error = self._extract_draw_aspect_ratio(text)
        if ratio_error:
            yield event.plain_result(ratio_error)
            event.stop_event()
            return
        if aspect_ratio is None:
            aspect_ratio = preset_aspect_ratio

        positive, negative, chain, width, height, scale = self._parse_params(text, default_negative="")
        if aspect_ratio is not None:
            if self._qwen_engine is None:
                yield event.plain_result("Qwen 绘图功能未开启或工作流加载失败")
                event.stop_event()
                return
            try:
                width, height = self._qwen_engine.dimensions_for_aspect_ratio(
                    aspect_ratio, width=width, height=height
                )
            except ValueError as error:
                yield event.plain_result(str(error))
                event.stop_event()
                return
            logger.info(
                f"[Qwen] 长宽比 {aspect_ratio.numerator}:{aspect_ratio.denominator} "
                f"解析为画布 {width}x{height}"
            )

        reference_images = None
        if event.get_platform_name() == "synochat_adapter" and not any(
            self._is_edit_image(item) for item in event.get_messages()
        ):
            component = self._recent_chat_image_component(event)
            if component is not None:
                image = await self._get_image_data(component)
                if image is None:
                    yield event.plain_result("最近发送的图片无法读取或超出大小限制，请缩小后重新上传。")
                    event.stop_event()
                    return
                reference_images = [image]

        task, data, error = await self._generate_image(
            event, "qwen", positive, negative, width, height, scale,
            mode="auto", lora_selection=lora_selection, reference_images=reference_images,
        )
        if data is None:
            if error:
                yield event.plain_result(error)
            return
        result = await self._deliver_generated_image(event, task, data, chain)
        for notice in self._image_delivery_notices(result):
            yield event.plain_result(notice)
        event.stop_event()

    @filter.command("继续改", alias=set(CONTINUE_EDIT_ALIASES[1:]))
    @_stop_matched_command
    async def cmd_continue_edit(self, event: AstrMessageEvent):
        """基于本人在当前会话上一张生成的无损本地图片继续用 Qwen 修改，无需手动引用。"""
        passed, reject_msg = self._check_whitelist(event)
        if not passed:
            if reject_msg:
                yield event.plain_result(reject_msg)
            return
        last_image = self._recent_generated_image_bytes(event)
        if last_image is None:
            yield event.plain_result("当前会话还没有你最近生成的图片（或临时文件已过期）。请先生成一张图，或直接附图/引用图片改图。")
            event.stop_event()
            return
        text = _strip_command_prefix(event.message_str.strip(), CONTINUE_EDIT_ALIASES)
        if not text:
            default_profile = next(
                (p for p in self.draw_edit_command_profiles if not p.get("direct_editor") and not p.get("direct_cutout") and p.get("options")),
                None,
            )
            if default_profile is not None:
                await self._start_draw_edit_option_menu(event, default_profile)
                event.stop_event()
                return
            yield event.plain_result("请在命令后写明修改要求，例如：/继续改 把背景换成海边日落")
            event.stop_event()
            return
        text, lora_selection, lora_error = self._parse_draw_lora_args(text)
        if lora_error:
            yield event.plain_result(lora_error)
            event.stop_event()
            return
        positive, negative, chain, width, height, scale = self._parse_params(text, default_negative="")
        if not positive.strip():
            yield event.plain_result("请输入修改要求，例如：/继续改 把头发改成银白色")
            event.stop_event()
            return
        task, data, error = await self._generate_image(
            event, "qwen", positive, negative, width, height, scale,
            mode="edit", reference_images=[last_image], lora_selection=lora_selection,
        )
        if data is None:
            if error:
                yield event.plain_result(error)
            return
        result = await self._deliver_generated_image(event, task, data, chain)
        for notice in self._image_delivery_notices(result):
            yield event.plain_result(notice)
        event.stop_event()

    @filter.command("抠图", alias=set(CUTOUT_COMMAND_ALIASES[1:]))
    @_stop_matched_command
    async def cmd_cutout(self, event: AstrMessageEvent):
        """一键抠图去背景（输出透明通道 PNG），无需弹出选项菜单。"""
        await self._run_direct_cutout(event)
        event.stop_event()
        if False:
            yield

    @filter.command("放大", alias=set(UPSCALE_COMMAND_ALIASES[1:]))
    @_stop_matched_command
    async def cmd_upscale(self, event: AstrMessageEvent):
        """使用 Qwen Image 2.1 Tiled Upscale 分块重绘高清放大（支持如 /放大 或 /放大 2x）。"""
        scale_factor, target_mp = self._parse_upscale_arg(event.message_str)
        await self._run_direct_upscale(event, scale_factor=scale_factor, target_mp=target_mp)
        event.stop_event()
        if False:
            yield

    @filter.command("生成多视图", alias=set(MULTIVIEW_COMMAND_ALIASES[1:]))
    @_stop_matched_command
    async def cmd_multiview(self, event: AstrMessageEvent):
        """弹出 6 种多视图生成菜单，或通过 /生成多视图 1~6 直接执行。"""
        text = _strip_command_prefix(event.message_str.strip(), MULTIVIEW_COMMAND_ALIASES).strip()
        direct_mode = None
        custom_prompt = None
        if text:
            m = re.match(r"^([1-6])(?:[\s=:：]+(.*))?$", text, re.DOTALL)
            if m:
                direct_mode = int(m.group(1))
                custom_prompt = (m.group(2) or "").strip() or None
        await self._start_multiview_menu(event, direct_mode=direct_mode, custom_prompt=custom_prompt)
        event.stop_event()
        if False:
            yield

    @filter.command("img2video", alias=set(IMG2VIDEO_ALIASES[1:]))
    @_stop_matched_command
    async def cmd_img2video(self, event: AstrMessageEvent):
        text = _strip_command_prefix(event.message_str.strip(), IMG2VIDEO_ALIASES)
        positive, negative, _chain, _width, _height, _scale = self._parse_params(text)
        # 从 positive 中再额外提取 fps / length
        fps_value: Optional[float] = None
        length_value: Optional[float] = None
        if positive:
            fps_pattern = r'(?:\s+|^)(?:fps|帧率)\s*[:=]?\s*(\d+(?:\.\d+)?)'
            m = re.search(fps_pattern, positive, re.IGNORECASE)
            if m:
                fps_value = float(m.group(1))
                positive = re.sub(fps_pattern, '', positive, flags=re.IGNORECASE).strip()

            length_pattern = r'(?:\s+|^)(?:length|len|时长|长度|秒)\s*[:=]?\s*(\d+(?:\.\d+)?)'
            m = re.search(length_pattern, positive, re.IGNORECASE)
            if m:
                length_value = float(m.group(1))
                positive = re.sub(length_pattern, '', positive, flags=re.IGNORECASE).strip()

        generated, error = await self._generate_video(event, positive, negative, fps=fps_value, length=length_value)
        if generated is None:
            if error:
                yield event.plain_result(error)
            return
        ticket, video_data = generated
        temp_file = self.temp_dir / f"{uuid.uuid4().hex}.mp4"
        saved = False
        try:
            temp_file.write_bytes(video_data)
            saved = True
            await self._send_generated_video(event, temp_file)
        except Exception as error:
            ticket.finish("delivery_failed" if saved else "failed")
            logger.warning("[ComfyUI] 视频保存或发送失败：%s", type(error).__name__)
            detail = f"（{error}）" if isinstance(error, ValueError) else ""
            yield event.plain_result(f"视频已生成，保存或发送失败{detail}。")
            return
        ticket.finish("sent")

        event.stop_event()

    async def _execute_reroll(self, event: AstrMessageEvent):
        passed, reject_msg = self._check_whitelist(event)
        if not passed:
            if reject_msg:
                await self._send_text_message(event, reject_msg)
            return
        spec = self._get_reroll_spec(event)
        if spec is None:
            await self._send_text_message(
                event, "当前会话没有可重画的最近任务（记录保留 30 分钟），请先生成一次图片或视频。"
            )
            return
        kind = spec.get("kind")
        if kind == "multiview":
            await self._execute_multiview_mode(
                event,
                spec["image_bytes"],
                spec["mode"],
                custom_prompt=spec.get("custom_prompt"),
                do_upscale=bool(spec.get("do_upscale")),
                upscale_arg=str(spec.get("upscale_arg") or ""),
            )
            return
        if kind == "upscale":
            await self._run_direct_upscale(
                event,
                scale_factor=spec.get("scale_factor"),
                target_mp=spec.get("target_mp"),
                image_bytes=spec["image_bytes"],
            )
            return
        if kind == "video":
            generated, error = await self._generate_video(
                event,
                prompt=spec.get("prompt", ""),
                negative=spec.get("negative", ""),
                fps=spec.get("fps"),
                length=spec.get("length"),
                image_bytes=spec["image_bytes"],
            )
            if generated is None:
                if error:
                    await self._send_text_message(event, error)
                return
            ticket, video_data = generated
            temp_file = self.temp_dir / f"{uuid.uuid4().hex}.mp4"
            saved = False
            try:
                temp_file.write_bytes(video_data)
                saved = True
                await self._send_generated_video(event, temp_file)
            except Exception as error:
                ticket.finish("delivery_failed" if saved else "failed")
                logger.warning("[ComfyUI] 重画视频保存或发送失败：%s", type(error).__name__)
                await self._send_text_message(event, "视频已生成，保存或发送失败。")
                return
            ticket.finish("sent")
            return
        if kind == "image":
            task, data, error = await self._generate_image(
                event,
                spec.get("backend", "qwen"),
                spec.get("prompt", ""),
                negative=spec.get("negative"),
                width=spec.get("width"),
                height=spec.get("height"),
                scale=spec.get("scale"),
                mode=spec.get("mode", "generate"),
                lora_selection=spec.get("lora_selection"),
                reference_images=spec.get("reference_images"),
                qwen_workflow_upscale=spec.get("qwen_workflow_upscale"),
                qwen_workflow_upscale_only=bool(spec.get("qwen_workflow_upscale_only")),
                qwen_edit_has_watermark=bool(spec.get("qwen_edit_has_watermark")),
                edit_engine=spec.get("edit_engine"),
                edit_mask=spec.get("edit_mask"),
                mask_lora_defaults=spec.get("mask_lora_defaults"),
                native_mask_original=spec.get("native_mask_original"),
                post_upscale=spec.get("post_upscale"),
                edit_profile_command=spec.get("edit_profile_command"),
            )
            if data is None:
                if error:
                    await self._send_text_message(event, error)
                return
            result = await self._deliver_generated_image(event, task, data, self.default_chain)
            for notice in self._image_delivery_notices(result):
                await self._send_text_message(event, notice)
            return
        await self._send_text_message(event, "最近一次任务类型暂不支持一键重画。")

    @filter.command("重画", alias={"再来一张", "重roll", "reroll"})
    @_stop_matched_command
    async def cmd_reroll(self, event: AstrMessageEvent):
        """使用本人在当前会话最近一次成功的生成参数，换随机种子一键重画。"""
        await self._execute_reroll(event)
        event.stop_event()
        if False:
            yield

    @filter.command("测试绘图")
    @_stop_matched_command
    async def cmd_test_comfyui(self, event: AstrMessageEvent):
        """检查插件到已配置的 ComfyUI 服务的连通性。"""
        if await self._is_comfyui_available():
            yield event.plain_result("ComfyUI 在线，连接正常。")
        else:
            yield event.plain_result("ComfyUI不在线噢，快让喵蛋去开启")

    @filter.command("绘图状态")
    @_stop_matched_command
    async def cmd_image_task_status(self, event: AstrMessageEvent):
        """列出本人在当前会话的最近图片任务、当前阶段、预计剩余耗时及 ComfyUI 显存状态。"""
        owner = self._image_task_owner(event)
        queue = self._image_generation_queue
        task_id = self._image_task_arg(event)
        tasks = [queue.find(owner, task_id)] if task_id else queue.recent(owner)
        now = time.monotonic()
        lines = []
        for task in tasks:
            if task is None:
                continue
            ahead = queue.ahead(task)
            details = []
            if ahead is not None:
                details.append(f"前面还有 {ahead} 个任务")
            if task.status == "generating":
                if task.stage:
                    details.append(task.stage)
                eta_sec = None
                elapsed = (now - task.started) if task.started is not None else 0.0
                if task.progress:
                    step, total = task.progress
                    percent = int(step * 100 // total) if total > 0 else 0
                    details.append(f"{percent}% · {step}/{total}")
                    if total > 0 and step > 0 and elapsed >= 3.0 and (step / total) >= 0.08:
                        eta_sec = max(1, int(round(elapsed * (total - step) / step)))
                if eta_sec is None:
                    est = queue.estimate_duration(task.backend, task.model_group)
                    if est is not None and elapsed < est:
                        eta_sec = max(1, int(round(est - elapsed)))
                if eta_sec is not None:
                    if eta_sec >= 60:
                        details.append(f"预计剩余约 {eta_sec // 60}分{eta_sec % 60}秒")
                    else:
                        details.append(f"预计剩余约 {eta_sec}秒")
            suffix = f"（{' | '.join(details)}）" if details else ""
            lines.append(f"{task.id} · {task.backend} · {TASK_LABELS[task.status]}{suffix}")

        stats = await self.api.get_system_stats()
        if stats and stats.get("vram_total_gb", 0) > 0:
            gpu_line = (
                f"🖥️ {stats['name']} · 显存 {stats['vram_used_gb']}/{stats['vram_total_gb']} GB"
                f"（{stats['vram_percent']}%）"
            )
        else:
            gpu_line = "🖥️ ComfyUI 当前未连接或无法读取显存状态"

        if lines:
            lines.append("")
            lines.append(gpu_line)
            yield event.plain_result("\n".join(lines))
        else:
            yield event.plain_result(f"当前会话没有找到你的图片任务。\n\n{gpu_line}")
        event.stop_event()

    @filter.command("取消绘图")
    @_stop_matched_command
    async def cmd_cancel_image_task(self, event: AstrMessageEvent):
        """取消本人在当前会话尚在排队或正在 ComfyUI 中执行的任务。"""
        queue = self._image_generation_queue
        task = queue.find(self._image_task_owner(event), self._image_task_arg(event), ("queued", "generating"))
        previous = await queue.cancel(task)
        if previous == "queued":
            yield event.plain_result(f"已取消排队任务 {task.id}。")
        elif previous == "generating":
            prompt_id = task.comfy_prompt_id
            interrupted = bool(prompt_id) and await self.api.cancel_prompt(prompt_id)
            if interrupted:
                yield event.plain_result(f"已向 ComfyUI 请求中断任务 {task.id}。")
            else:
                yield event.plain_result(f"已取消本地任务 {task.id}，后端中断未确认。")
        else:
            yield event.plain_result("没有找到你正在排队或正在处理的任务。")
        event.stop_event()

    @filter.command("释放显存", alias={"清理显存", "comfy_free"})
    @_stop_matched_command
    async def cmd_free_comfy_memory(self, event: AstrMessageEvent):
        """卸载 ComfyUI 已加载模型并释放显存/内存缓存（仅管理员）。"""
        if not event.is_admin():
            yield event.plain_result("❌ 仅管理员可执行释放显存操作。")
            event.stop_event()
            return
        if await self.api.free_memory(unload_models=True, free_memory=True):
            yield event.plain_result("✅ 已请求 ComfyUI 卸载模型并释放显存与缓存。")
        else:
            yield event.plain_result("⚠️ 请求 ComfyUI 释放显存失败，请检查 ComfyUI 是否在线。")
        event.stop_event()

    @filter.command("重发图片")
    @_stop_matched_command
    async def cmd_resend_image_task(self, event: AstrMessageEvent):
        """取回或重发本人在当前会话的图片。"""
        passed, rejection = self._check_whitelist(event)
        if not passed:
            if rejection:
                yield event.plain_result(rejection)
            return
        queue = self._image_generation_queue
        owner = self._image_task_owner(event)
        task_id = self._image_task_arg(event)
        recoverable = ("delivery_failed", "result_unavailable")
        task = queue.find(owner, task_id) if task_id else queue.find(owner, None, recoverable)
        if task is not None and task.status not in recoverable:
            yield event.plain_result(f"任务 {task.id} 当前状态：{TASK_LABELS[task.status]}。")
            return
        if task is None and task_id:
            task = self._load_image_resend(owner, task_id)
        if task is None:
            yield event.plain_result("没有找到可重发的图片，或者临时文件已过期。")
            return
        if task.id in self._resending_images:
            yield event.plain_result("这张图片正在重发。")
            return
        self._resending_images.add(task.id)
        try:
            local = task.path is not None and task.path.is_file()
            data = None
            if not local:
                if not task.comfy_prompt_id:
                    yield event.plain_result("图片临时文件已过期，无法重发。")
                    return
                status, data = await self.api.get_image_result(task.comfy_prompt_id, task.comfy_output_node)
                if status == "failed":
                    task.finish("failed")
                    self._clear_image_resend(task.id)
                    yield event.plain_result("ComfyUI 历史记录显示任务失败或没有图片输出。")
                    return
                if data is None:
                    message = ("ComfyUI 历史里还没有这张图，任务可能在排队或处理中。"
                               if status == "pending" else "暂时无法从 ComfyUI 下载结果，稍后可再试。")
                    yield event.plain_result(message)
                    return
                checked, message = await self._check_output_censorship(
                    event, data, "文生图" if task.backend == "动漫" else "Qwen",
                )
                if not checked:
                    task.finish("failed")
                    self._clear_image_resend(task.id)
                    yield event.plain_result(message or "图片未通过输出检查。")
                    return
            result = await self._deliver_generated_image(event, task, data, force_qq_file=True)
        finally:
            self._resending_images.discard(task.id)
        for notice in self._image_delivery_notices(result):
            yield event.plain_result(notice)
        event.stop_event()
