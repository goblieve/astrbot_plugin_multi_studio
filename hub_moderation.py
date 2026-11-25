import asyncio
import base64
import json
import re
import time
from PIL import Image as PILImage
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.core.agent.message import ImageURLPart
from astrbot.core.agent.message import TextPart
from astrbot.core.agent.message import UserMessageSegment
from io import BytesIO
from pathlib import Path
from typing import Optional
from typing import Tuple
from .hub_common import _is_violation


class ModerationMixin:
    def _load_block_data(self):
        self.block_tags = set()
        self.output_block_tags = set()
        self.blocked_users = {}
        self.censored_groups = set()
        self.sent_messages = {}
        self.message_cache_ttl = 120

        self._load_json_set(self.output_block_tags_file, "output_block_tags")
        self._load_json_set(self.block_tags_file, "block_tags")

        if self.blocked_users_file.exists():
            try:
                with open(self.blocked_users_file, "r", encoding='utf-8') as f:
                    self.blocked_users = json.load(f)
            except Exception as e:
                logger.error(f"Error loading blocked users: {e}")

        if self.censorship_config_file.exists():
            try:
                with open(self.censorship_config_file, "r", encoding='utf-8') as f:
                    cfg = json.load(f)
                    self.censored_groups = set(cfg.get("groups", []))
            except Exception as e:
                logger.error(f"Error loading censorship config: {e}")

        if self.sent_messages_file.exists():
            try:
                with open(self.sent_messages_file, "r", encoding='utf-8') as f:
                    data = json.load(f)
                    self.sent_messages = {str(k): v for k, v in data.items()}
                    self._cleanup_expired_messages()
            except Exception as e:
                logger.error(f"Error loading sent messages: {e}")

    def _load_json_set(self, path: Path, attr: str):
        if not path.exists():
            return
        try:
            with open(path, "r", encoding='utf-8') as f:
                setattr(self, attr, set(json.load(f)))
        except Exception as e:
            logger.error(f"Error loading {attr}: {e}")

    def _cleanup_expired_messages(self):
        """清理过期的消息ID"""
        current_time = time.time()
        for group_id in list(self.sent_messages.keys()):
            valid_messages = [
                msg_data for msg_data in self.sent_messages[group_id]
                if isinstance(msg_data, dict) and
                current_time - msg_data.get('timestamp', 0) <= self.message_cache_ttl
            ]
            self.sent_messages[group_id] = valid_messages
            if not valid_messages:
                del self.sent_messages[group_id]

    @staticmethod
    def _atomic_write_json(path: Path, payload):
        try:
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "w", encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False)
            tmp.replace(path)
        except Exception as e:
            logger.error(f"Error saving {path.name}: {e}")

    def _save_block_tags(self):
        self._atomic_write_json(self.block_tags_file, list(self.block_tags))

    def _save_output_block_tags(self):
        self._atomic_write_json(self.output_block_tags_file, list(self.output_block_tags))

    def _save_blocked_users(self):
        self._atomic_write_json(self.blocked_users_file, self.blocked_users)

    def _save_censorship(self):
        self._atomic_write_json(self.censorship_config_file, {"groups": list(self.censored_groups)})

    def _save_sent_messages(self):
        self._cleanup_expired_messages()
        self._sent_messages_dirty = True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._sent_messages_dirty = False
            self._atomic_write_json(self.sent_messages_file, self.sent_messages)
            return
        task = getattr(self, "_sent_messages_flush_task", None)
        if task is None or task.done():
            self._sent_messages_flush_task = loop.create_task(self._debounced_flush_sent_messages())

    async def _debounced_flush_sent_messages(self):
        try:
            await asyncio.sleep(3.0)
            await self._flush_sent_messages()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.warning("[ComfyUI] 异步保存 sent_messages 失败：%s", type(error).__name__)

    async def _flush_sent_messages(self):
        if not getattr(self, "_sent_messages_dirty", False):
            return
        self._sent_messages_dirty = False
        self._cleanup_expired_messages()
        snapshot = {k: list(v) for k, v in self.sent_messages.items()}
        await asyncio.to_thread(self._atomic_write_json, self.sent_messages_file, snapshot)

    def _on_censorship_error(self, scope: str, error: Exception) -> Tuple[bool, str]:
        """统一处理审查链路异常，按 censorship_failure_mode 决定放行或拦截"""
        logger.error(f"[{scope}] 审查异常: {error}")
        if self.censorship_failure_mode == "fail_closed":
            return False, "⚠️ 审查服务暂不可用，请稍后再试。"
        return True, f"审查出错: {error}"

    async def _check_safety_with_llm(self, event: AstrMessageEvent, text: str) -> Tuple[bool, str]:
        """使用 AstrBot 内置 LLM 检查文本安全（用于输入审查）"""
        try:
            if not self.input_censorship_use_llm:
                return True, "LLM Disabled"

            provider_id = self.llm_provider_id
            if not provider_id:
                umo = event.unified_msg_origin
                provider_id = await self.context.get_current_chat_provider_id(umo=umo)
                if not provider_id:
                    raise RuntimeError("未配置可用的审查提供商")

            llm_resp = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=text,
                system_prompt=self.censorship_prompt,
            )

            if not llm_resp or not llm_resp.completion_text:
                raise RuntimeError("审查提供商未返回有效结果")

            result = llm_resp.completion_text.strip()
            logger.info(f"[输入审查] LLM原始响应: {result}")

            violation = _is_violation(result)
            logger.info(f"[输入审查] 判定结果: {'违规' if violation else '通过'}")

            if violation:
                return False, "AI审查拦截"
            return True, ""

        except Exception as e:
            return self._on_censorship_error("输入审查", e)

    async def _check_image_safety_with_llm(self, event: AstrMessageEvent, image_data: bytes,
                                           is_img2img_input: bool = False) -> Tuple[bool, str]:
        """使用多模态 LLM 检查图片安全"""
        try:
            provider_id = self.llm_provider_id
            if not provider_id:
                umo = event.unified_msg_origin
                provider_id = await self.context.get_current_chat_provider_id(umo=umo)
                if not provider_id:
                    raise RuntimeError("未配置可用的审查提供商")

            image_base64 = base64.b64encode(image_data).decode('utf-8')
            try:
                img = PILImage.open(BytesIO(image_data))
                mime_type = f"image/{img.format.lower()}"
            except Exception:
                mime_type = "image/png"
            image_url = f"data:{mime_type};base64,{image_base64}"

            check_type = "图生图输入" if is_img2img_input else "输出"
            user_msg = UserMessageSegment(content=[
                TextPart(text="请审查这张图片是否包含违规内容。"),
                ImageURLPart(image_url=ImageURLPart.ImageURL(url=image_url)),
            ])

            llm_resp = await self.context.llm_generate(
                chat_provider_id=provider_id,
                contexts=[user_msg],
                system_prompt=self.output_censorship_prompt,
            )

            if not llm_resp or not llm_resp.completion_text:
                raise RuntimeError("审查提供商未返回有效结果")

            result = llm_resp.completion_text.strip()
            logger.info(f"[{check_type}审查] 多模态LLM原始响应: {result}")

            violation = _is_violation(result)
            logger.info(f"[{check_type}审查] 判定结果: {'违规' if violation else '通过'}")

            if violation:
                return False, "AI审查拦截"
            return True, ""

        except Exception as e:
            return self._on_censorship_error("图片审查", e)

    def _resolve_output_censor_options(self, check_type: str) -> tuple:
        """根据 check_type 选择对应的输出审查开关"""
        if check_type == "文生图":
            return (
                self.enable_output_censorship,
                self.output_censorship_use_llm,
                self.output_censorship_use_tagger,
            )
        return (
            self.enable_img2img_output_censorship,
            self.img2img_output_censorship_use_llm,
            self.img2img_output_censorship_use_tagger,
        )

    async def _check_output_censorship(self, event: AstrMessageEvent, image_data: bytes,
                                       check_type: str = "文生图") -> Tuple[bool, str]:
        """统一的输出图片审查"""
        group_id = event.get_group_id()
        is_censorship_enabled = group_id and group_id in self.censored_groups

        is_admin = event.is_admin()
        should_bypass_censorship = is_admin and self.admin_bypass_censorship

        enable, use_llm, use_tagger = self._resolve_output_censor_options(check_type)

        if not (is_censorship_enabled and not should_bypass_censorship and enable):
            logger.info(f"[{check_type}] 未开启输出审查或已绕过")
            return True, ""

        if use_tagger and self.img2txt:
            tags_text = await self.img2txt.generate(image_data)
            if tags_text:
                logger.info(f"[{check_type}] 输出图片标签: {tags_text}")
                is_safe_simple, reason_simple = self._check_simple_tags(tags_text)
                if not is_safe_simple:
                    logger.info(f"[{check_type}] 输出图片关键词审查拦截: {reason_simple}")
                    return False, f"⚠️ 生成的图片{reason_simple}，已被审查系统拒绝。"
            else:
                logger.info(f"[{check_type}] Tagger 未返回标签，跳过 tagger 审查")
        elif use_tagger and not self.img2txt:
            logger.info(f"[{check_type}] Tagger 不可用，跳过 tagger 审查")

        if use_llm:
            is_safe, _ = await self._check_image_safety_with_llm(event, image_data, is_img2img_input=False)
            if not is_safe:
                logger.info(f"[{check_type}] 输出图片多模态LLM审查拦截")
                return False, "⚠️ 生成的图片包含敏感内容，已被AI审查系统拒绝。"

        logger.info(f"[{check_type}] 输出图片审查通过")
        return True, ""

    async def _check_img2img_input_censorship(self, event: AstrMessageEvent,
                                              image_data: bytes) -> Tuple[bool, str]:
        """图生图输入图片审查（多模态LLM）"""
        group_id = event.get_group_id()
        is_censorship_enabled = group_id and group_id in self.censored_groups
        is_admin = event.is_admin()
        should_bypass_censorship = is_admin and self.admin_bypass_censorship

        if not (is_censorship_enabled and not should_bypass_censorship and self.enable_img2img_input_censorship):
            logger.info("[图生图] 未开启输入图片审查或已绕过")
            return True, ""

        if self.img2img_input_censorship_use_llm:
            is_safe, _ = await self._check_image_safety_with_llm(event, image_data, is_img2img_input=True)
            if not is_safe:
                logger.info("[图生图] 输入图片多模态LLM审查拦截")
                return False, "⚠️ 您输入的图片包含敏感内容，已被AI审查系统拒绝。"

        logger.info("[图生图] 输入图片审查通过")
        return True, ""

    async def _run_input_text_censorship(self, event: AstrMessageEvent, positive: str) -> Tuple[Optional[str], Optional[str]]:
        """对正面提示词跑一遍（封禁检查 + block tag + LLM）输入审查

        Returns:
            (positive_or_None, message_or_None):
              - positive_or_None: 通过则返回（可能附加 sfw 后缀的）正面提示词；不通过返回 None
              - message_or_None: 不通过时的提示文本
        """
        user_id = event.get_sender_id()
        current_time = time.time()

        # 封禁期检查
        if user_id in self.blocked_users:
            expire_time = self.blocked_users[user_id]
            if current_time < expire_time:
                remaining = int(expire_time - current_time)
                return None, f"由于触发违规词，您已被禁止使用绘图功能。剩余时间: {remaining} 秒。"
            del self.blocked_users[user_id]
            self._save_blocked_users()

        group_id = event.get_group_id()
        is_censorship_enabled = group_id and group_id in self.censored_groups
        is_admin = event.is_admin()
        should_bypass_censorship = is_admin and self.admin_bypass_censorship

        if not (is_censorship_enabled and not should_bypass_censorship and self.enable_input_censorship):
            return positive, None

        # 本地 block tag
        for tag in self.block_tags:
            if tag.lower() in positive.lower():
                self.blocked_users[user_id] = current_time + 120
                self._save_blocked_users()
                return None, f"⚠️ 违规：检测到敏感词 '{tag}'。您将被禁服务 2 分钟。"

        # LLM 审查
        if self.input_censorship_use_llm:
            is_safe, _ = await self._check_safety_with_llm(event, positive)
            if not is_safe:
                self.blocked_users[user_id] = current_time + 120
                self._save_blocked_users()
                logger.info("LLM 审查拦截")
                return None, "⚠️ 您的请求包含敏感内容，已被AI审查系统拒绝。您将被禁服务 2 分钟。"

        # 自动追加 sfw
        if "sfw" not in positive.lower() and "safe" not in positive.lower():
            positive = (positive + ", sfw, safe for work").strip(", ")
        return positive, None

    def _check_simple_tags(self, tags_text: str) -> tuple:
        """精确匹配 tag token，避免子串误命中（man → woman/human）"""
        tags = [tag.strip().lower() for tag in tags_text.split(',')]
        block_lower = {kw.lower() for kw in self.output_block_tags}
        for tag in tags:
            tokens = re.split(r'[\s_]+', tag)
            for keyword in block_lower:
                # 允许 keyword 自身含空格/下划线，作为整体匹配
                kw_tokens = re.split(r'[\s_]+', keyword) if (' ' in keyword or '_' in keyword) else None
                if kw_tokens:
                    # 子序列匹配
                    n = len(kw_tokens)
                    for i in range(len(tokens) - n + 1):
                        if tokens[i:i + n] == kw_tokens:
                            return False, f"检测到敏感内容 '{keyword}'"
                elif keyword in tokens:
                    return False, f"检测到敏感内容 '{keyword}'"
        return True, ""

    async def _handle_admin_subcommand(self, event: AstrMessageEvent, text: str):
        """处理 #draw $xxx 形式的管理员子命令，返回 (handled, message_or_None)"""
        if not text.startswith('$'):
            return False, None
        if not event.is_admin():
            return True, "❌ 仅管理员可执行此操作。"

        if text.startswith('$enable_censorship'):
            group_id = event.get_group_id()
            if not group_id:
                return True, "⚠️ 此命令仅支持在群组中使用。"
            self.censored_groups.add(group_id)
            self._save_censorship()
            return True, "✅ 已在当前群组开启审查功能。"

        if text.startswith('$disable_censorship'):
            group_id = event.get_group_id()
            if not group_id:
                return True, "⚠️ 此命令仅支持在群组中使用。"
            if group_id in self.censored_groups:
                self.censored_groups.remove(group_id)
                self._save_censorship()
            return True, "✅ 已在当前群组关闭审查功能。"

        if text.startswith('$add_block_tag'):
            tags_part = text[len('$add_block_tag'):].strip()
            new_tags = self._parse_tag_list(tags_part)
            if not new_tags:
                return True, "用法: #draw $add_block_tag tag1,tag2 或 [tag1] [tag2]"
            self.block_tags.update(new_tags)
            self._save_block_tags()
            return True, f"✅ 已成功添加违规词: {', '.join(new_tags)}"

        if text.startswith('$add_output_block_tag'):
            tags_part = text[len('$add_output_block_tag'):].strip()
            new_tags = self._parse_tag_list(tags_part)
            if not new_tags:
                return True, "用法: #draw $add_output_block_tag tag1,tag2"
            self.output_block_tags.update(new_tags)
            self._save_output_block_tags()
            return True, f"✅ 已添加输出违规词: {', '.join(new_tags)}"

        if text.startswith('$remove_output_block_tag'):
            tags_part = text[len('$remove_output_block_tag'):].strip()
            rem_tags = self._parse_tag_list(tags_part)
            if not rem_tags:
                return True, "用法: #draw $remove_output_block_tag tag1,tag2"
            removed = [t for t in rem_tags if t in self.output_block_tags]
            self.output_block_tags -= set(rem_tags)
            self._save_output_block_tags()
            if removed:
                return True, f"✅ 已移除输出违规词: {', '.join(removed)}"
            return True, "⚠️ 未找到指定的输出违规词"

        if text.startswith('$remove_block_tag'):
            tags_part = text[len('$remove_block_tag'):].strip()
            rem_tags = self._parse_tag_list(tags_part)
            if not rem_tags:
                return True, "用法: #draw $remove_block_tag tag1,tag2 或 [tag1] [tag2]"
            removed = []
            for t in rem_tags:
                if t in self.block_tags:
                    self.block_tags.remove(t)
                    removed.append(t)
            self._save_block_tags()
            if removed:
                return True, f"✅ 已成功移除违规词: {', '.join(removed)}"
            return True, "⚠️ 未找到指定的违规词。"

        return False, None

    @staticmethod
    def _parse_tag_list(tags_part: str) -> list:
        raw_tags = re.split(r',|\[|\]', tags_part)
        return [t.strip() for t in raw_tags if t.strip()]

    def _check_whitelist(self, event: AstrMessageEvent) -> Tuple[bool, Optional[str]]:
        """检查会话是否在白名单内。

        群聊白名单与私聊白名单分别由 enable_group_whitelist / enable_user_whitelist 控制。
        当对应开关开启时，列表为空相当于禁用该类型会话；开关关闭则不限制该类型会话。

        Returns:
            (passed, reject_message): 通过返回 (True, None)；不通过返回 (False, message_or_None)，
            message 为 None 表示静默忽略
        """
        if self.whitelist_admin_bypass and event.is_admin():
            return True, None

        group_id = event.get_group_id()
        if group_id:
            if self.enable_group_whitelist and str(group_id) not in self.whitelist_groups:
                msg = self.whitelist_reject_message.strip() if self.whitelist_reject_message else ""
                return False, msg or None
            return True, None

        if self.enable_user_whitelist:
            user_id = event.get_sender_id()
            if not user_id or str(user_id) not in self.whitelist_users:
                msg = self.whitelist_reject_message.strip() if self.whitelist_reject_message else ""
                return False, msg or None
        return True, None
