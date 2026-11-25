import aiohttp
import asyncio
import json
import re
from astrbot.api import logger
from urllib.parse import quote
from .draw_tasks import TASK_LABELS
from .hub_common import DEFAULT_DRAW_EDIT_OPTION_ROWS, EDIT_OPTION_PLUGIN_NAME, astrbot_web
from .multiview_workflow import DEFAULT_MULTIVIEW_PROMPTS, normalize_multiview_prompts


class WebApiMixin:
    def _register_draw_edit_options_web_api(self):
        register = getattr(self.context, "register_web_api", None)
        if not callable(register) or astrbot_web is None:
            logger.warning("[ComfyUI] 当前 AstrBot 不支持插件 WebUI API，改图选项管理页不可用")
            return
        base = f"/{EDIT_OPTION_PLUGIN_NAME}/edit_options"
        try:
            register(f"{base}/state", self._webapi_draw_edit_options_state, ["GET"], "Read image edit menu options")
            register(f"{base}/save", self._webapi_save_draw_edit_options, ["POST"], "Save image edit menu options")
            register(f"{base}/validate", self._webapi_validate_draw_edit_options, ["POST"], "Validate image edit menu options")
            register(f"{base}/restore", self._webapi_restore_draw_edit_options, ["POST"], "Restore previous image edit menu options")
            register(f"{base}/diagnostics", self._webapi_draw_diagnostics, ["GET"], "Check rendering backends and timings")
            register(f"{base}/workflows", self._webapi_workflow_list, ["GET"], "List named workflows")
            register(f"{base}/workflows/save", self._webapi_workflow_save, ["POST"], "Import or replace a named workflow")
            register(f"{base}/workflows/delete", self._webapi_workflow_delete, ["POST"], "Remove an unbound workflow")
            register(f"{base}/multiview/save", self._webapi_multiview_save, ["POST"], "Save multi-view prompts")
            register(f"{base}/multiview/reset", self._webapi_multiview_reset, ["POST"], "Reset multi-view prompts to defaults")
        except Exception as error:
            logger.warning(f"[ComfyUI] 注册改图选项 WebUI API 失败：{error}")

    def _draw_lora_alias_names(self):
        configured_rows = self.draw_lora_aliases
        if isinstance(configured_rows, str):
            configured_rows = [configured_rows]
        engine_aliases = set(getattr(self._qwen_engine, "lora_aliases", {}).keys())
        aliases = []
        seen = set()
        for row in configured_rows or []:
            if not isinstance(row, str) or "=" not in row:
                continue
            alias = row.split("=", 1)[0].strip()
            key = alias.casefold()
            if alias and key in engine_aliases and key not in seen:
                aliases.append(alias)
                seen.add(key)
        return aliases

    def _edit_profiles_response(self):
        return astrbot_web.json_response({
            "profiles": self.draw_edit_command_profiles,
            "lora_aliases": self._draw_lora_alias_names(),
            "can_restore": self._previous_edit_profiles.is_file(),
            "workflows": self._workflow_library.list(),
            "mask_editor_ready": self._mask_editor.available,
            "multiview_prompts": self.multiview_prompts,
            "multiview_defaults": DEFAULT_MULTIVIEW_PROMPTS,
        })

    async def _webapi_draw_edit_options_state(self):
        async with self._edit_profiles_lock:
            return self._edit_profiles_response()

    async def _webapi_workflow_list(self):
        return astrbot_web.json_response({
            "workflows": self._workflow_library.list(),
            "mask_editor_ready": self._mask_editor.available,
        })

    async def _webapi_workflow_save(self):
        try:
            payload = await astrbot_web.request.json(default={})
            if not isinstance(payload, dict):
                raise ValueError("请求必须是 JSON 对象")
            async with self._edit_profiles_lock:
                self._workflow_library.save(
                    payload.get("name"), payload.get("graph"), payload.get("bindings"),
                    payload.get("id", ""), payload.get("kind", "qwen"),
                )
            return await self._webapi_workflow_list()
        except (ValueError, TypeError) as error:
            return astrbot_web.error_response(str(error), status_code=400)
        except OSError:
            return astrbot_web.error_response("无法保存工作流库文件", status_code=500)

    async def _webapi_workflow_delete(self):
        try:
            payload = await astrbot_web.request.json(default={})
            if not isinstance(payload, dict):
                raise ValueError("请求必须是 JSON 对象")
            workflow_id = payload.get("id")
            async with self._edit_profiles_lock:
                if any(profile.get("workflow_id") == workflow_id for profile in self.draw_edit_command_profiles):
                    return astrbot_web.error_response("该工作流仍有指令组绑定，请先解除绑定并保存。", status_code=409)
                self._workflow_library.delete(workflow_id)
            return await self._webapi_workflow_list()
        except (ValueError, TypeError) as error:
            return astrbot_web.error_response(str(error), status_code=400)
        except OSError:
            return astrbot_web.error_response("无法删除工作流库文件", status_code=500)

    def _load_multiview_prompts(self) -> dict:
        try:
            if self._multiview_prompts_file.is_file():
                raw = json.loads(self._multiview_prompts_file.read_text(encoding="utf-8"))
                return normalize_multiview_prompts(raw)
        except Exception as error:
            logger.warning("[ComfyUI] 读取多视图提示词配置失败，已使用默认值：%s", type(error).__name__)
        return normalize_multiview_prompts(DEFAULT_MULTIVIEW_PROMPTS)

    def _save_multiview_prompts(self, prompts: dict):
        normalized = normalize_multiview_prompts(prompts)
        staged = self._multiview_prompts_file.with_suffix(".tmp")
        try:
            staged.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
            staged.replace(self._multiview_prompts_file)
        finally:
            staged.unlink(missing_ok=True)
        self.multiview_prompts = normalized
        return normalized

    async def _webapi_multiview_save(self):
        try:
            payload = await astrbot_web.request.json(default={})
            if not isinstance(payload, dict):
                raise ValueError("请求体必须是 JSON 对象")
            async with self._edit_profiles_lock:
                saved = await asyncio.to_thread(self._save_multiview_prompts, payload.get("prompts"))
            return astrbot_web.json_response({"multiview_prompts": saved})
        except (ValueError, TypeError) as error:
            return astrbot_web.error_response(str(error), status_code=400)
        except OSError:
            return astrbot_web.error_response("无法保存多视图提示词配置文件", status_code=500)

    async def _webapi_multiview_reset(self):
        try:
            async with self._edit_profiles_lock:
                saved = await asyncio.to_thread(self._save_multiview_prompts, DEFAULT_MULTIVIEW_PROMPTS)
            return astrbot_web.json_response({"multiview_prompts": saved})
        except OSError:
            return astrbot_web.error_response("无法重置多视图提示词配置文件", status_code=500)

    def _check_profile_workflows(self, profiles):
        for profile in profiles:
            workflow_id = profile.get("workflow_id")
            workflow = self._workflow_library.get(workflow_id) if workflow_id else None
            if workflow_id and workflow is None:
                raise ValueError(f"指令“{profile['command']}”绑定的工作流不存在或加载失败")
            if profile.get("direct_editor") is True and (workflow is None or workflow.kind != "mask"):
                raise ValueError(f"指令“{profile['command']}”需绑定局部重绘工作流才能直接发送链接")
            defaults = profile.get("mask_lora_defaults")
            if defaults:
                catalog = workflow.engine.lora_catalog()
                for section, rows in defaults.items():
                    if set(rows) - {row["name"] for row in catalog.get(section, ())}:
                        raise ValueError(f"指令“{profile['command']}”的 LoRA 已不在对应支路，请刷新配置后重试")

    def _validate_edit_profiles(self, rows):
        profiles = self._normalize_draw_edit_command_profiles(rows)
        self._check_profile_workflows(profiles)
        aliases = {alias.casefold() for alias in self._draw_lora_alias_names()}
        for profile in profiles:
            for option in profile["options"]:
                name = option["name"]
                for lora in option["loras"]:
                    if lora["alias"].casefold() not in aliases:
                        raise ValueError(f"选项“{name}”引用了未配置的 LoRA 别名“{lora['alias']}”。")
                remainder = re.sub(r"(?<!\{)\{prompt[^{}\s]*\}(?!\})", "", option["prompt"], flags=re.I)
                if re.search(r"\{\s*prompt", remainder, flags=re.I):
                    raise ValueError(f"选项“{name}”中的占位符格式不完整，请使用 {{prompt名称}}。")
        return profiles

    async def _read_edit_profiles_request(self):
        try:
            payload = await astrbot_web.request.json(default={})
        except Exception as error:
            raise ValueError("请求体不是合法 JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return self._validate_edit_profiles(payload.get("profiles"))

    async def _webapi_validate_draw_edit_options(self):
        try:
            await self._read_edit_profiles_request()
        except ValueError as error:
            return astrbot_web.error_response(str(error), status_code=400)
        return astrbot_web.json_response({"valid": True})

    async def _webapi_save_draw_edit_options(self):
        try:
            profiles = await self._read_edit_profiles_request()
        except ValueError as error:
            return astrbot_web.error_response(str(error), status_code=400)
        async with self._edit_profiles_lock:
            return await self._apply_edit_profiles(profiles)

    async def _webapi_restore_draw_edit_options(self):
        async with self._edit_profiles_lock:
            try:
                rows = json.loads(self._previous_edit_profiles.read_text(encoding="utf-8"))
                profiles = self._validate_edit_profiles(rows)
            except FileNotFoundError:
                return astrbot_web.error_response("目前没有可恢复的上一版", status_code=409)
            except ValueError as error:
                return astrbot_web.error_response(f"上一版无法恢复：{error}", status_code=400)
            except OSError:
                return astrbot_web.error_response("读取上一版失败", status_code=500)
            return await self._apply_edit_profiles(profiles)

    async def _apply_edit_profiles(self, profiles):
        try:
            self._check_profile_workflows(profiles)
        except ValueError as error:
            return astrbot_web.error_response(str(error), status_code=400)
        if profiles == self.draw_edit_command_profiles:
            return self._edit_profiles_response()

        profile_key = "draw_edit_command_profiles"
        legacy_key = "draw_edit_options"
        previous_rows = self.config.get(profile_key, [])
        previous_legacy = self.config.get(legacy_key, DEFAULT_DRAW_EDIT_OPTION_ROWS)
        history = self._previous_edit_profiles
        staged = history.with_suffix(".tmp")
        try:
            staged.write_text(
                json.dumps(self.draw_edit_command_profiles, ensure_ascii=False),
                encoding="utf-8",
            )
            self.config[profile_key] = [
                json.dumps(profile, ensure_ascii=False, separators=(",", ":")) for profile in profiles
            ]
            self.config[legacy_key] = []
            save_async = getattr(self.config, "save_config_async", None)
            save_sync = getattr(self.config, "save_config", None)
            if callable(save_async):
                saved = await save_async()
            elif callable(save_sync):
                saved = await asyncio.to_thread(save_sync)
            else:
                raise RuntimeError("当前 AstrBot 配置对象不支持持久化保存")
            if saved is False:
                raise RuntimeError("配置保存未提交")
        except Exception as error:
            self.config[profile_key] = previous_rows
            self.config[legacy_key] = previous_legacy
            staged.unlink(missing_ok=True)
            logger.error(f"[ComfyUI] 保存改图选项失败：{type(error).__name__}")
            return astrbot_web.error_response("保存失败，请检查 AstrBot 配置日志后重试", status_code=500)

        self.draw_edit_command_profiles = profiles
        try:
            staged.replace(history)
        except OSError as error:
            logger.error(f"[ComfyUI] 更新改图上一版失败：{type(error).__name__}")
            return astrbot_web.error_response("当前指令已保存，但更新上一版失败。请刷新页面。", status_code=500)
        return self._edit_profiles_response()

    async def _webapi_draw_diagnostics(self):
        comfy_up = await self._is_comfyui_available()
        provider, _ = self._select_codex_image_provider()
        workflows = {
            "Qwen": ("enable_qwen", self._qwen_engine),
            "动漫": ("enable_txt2img", self.txt2img),
            "视频": ("enable_img2video", self._img2video_engine),
            "Tagger": ("enable_tagger", self.img2txt),
        }
        classes = {
            name: {node["class_type"] for node in engine.template.workflow.values()} if engine else set()
            for name, (_, engine) in workflows.items()
        }
        known = {}
        if comfy_up:
            semaphore = asyncio.Semaphore(6)
            async with self.api.session() as session:
                async def inspect_node(node):
                    async with semaphore:
                        try:
                            async with session.get(
                                f"{self.api.server_url}/object_info/{quote(node, safe='')}",
                                timeout=aiohttp.ClientTimeout(total=2),
                            ) as response:
                                if response.status == 200:
                                    result = await response.json()
                                    if isinstance(result, dict) and (not result or node in result):
                                        known[node] = node in result
                        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
                            return
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*(inspect_node(node) for node in set().union(*classes.values()))),
                        timeout=7,
                    )
                except asyncio.TimeoutError:
                    pass
        return astrbot_web.json_response({
            "comfyui_online": comfy_up,
            "fallback_enabled": self.codex_fallback_enabled,
            "gpt_image_available": provider is not None,
            "workflows": [
                {
                    "name": name,
                    "enabled": bool(self.config.get(key, True)),
                    "loaded": engine is not None,
                    "missing": sorted(node for node in classes[name] if known.get(node) is False),
                    "unknown": len(classes[name] - known.keys()),
                }
                for name, (key, engine) in workflows.items()
            ],
            "recent": [
                {"id": task.id, "backend": task.backend, "status": TASK_LABELS[task.status],
                 **task.timings()}
                for task in self._image_generation_queue.recent(limit=10)
            ],
        })
