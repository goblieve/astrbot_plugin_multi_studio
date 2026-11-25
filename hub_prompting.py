import asyncio
import math
import re
from fractions import Fraction
from typing import Optional
from astrbot.api import logger


class PromptingMixin:
    @staticmethod
    def _load_draw_prompt_presets(rows):
        """加载配置中的“名称=模板”列表，名称不区分大小写。"""
        if isinstance(rows, str):
            rows = [rows]
        presets = {}
        for row in rows or []:
            if not isinstance(row, str) or "=" not in row:
                continue
            name, template = row.split("=", 1)
            name, template = name.strip(), template.strip()
            if not name or not template or any(char.isspace() for char in name):
                continue
            presets.setdefault(name.casefold(), (name, template))
        return presets

    def _extract_draw_preset(self, text: str):
        pattern = re.compile(r"(?<!\S)预设=(\S+)")
        matches = list(pattern.finditer(text))
        if not matches:
            if re.search(r"(?<!\S)预设=(?=\s|$)", text):
                return text, None, "预设名称不能为空，用法：/draw 预设=名称 提示词"
            return text, None, None
        if len(matches) > 1:
            return text, None, "一次只能指定一个预设，例如：/draw 预设=分镜 提示词"
        remaining = pattern.sub(" ", text)
        remaining = re.sub(r"[ \t]+", " ", remaining).strip()
        return remaining, matches[0].group(1), None

    @staticmethod
    def _extract_draw_aspect_ratio(text: str):
        option = r"(?:比例|宽高比|长宽比|--ar|aspect[_-]?ratio)"
        pattern = re.compile(
            rf"(?<!\w){option}\s*[:=]?\s*(\d+(?:\.\d+)?)\s*[:/x×]\s*(\d+(?:\.\d+)?)(?:[,，;；、。.!！?？]\s*|(?=$|\s))",
            re.IGNORECASE,
        )
        matches = list(pattern.finditer(text))
        if not matches:
            malformed = re.search(
                rf"(?<!\w){option}\s*[:=]\s*\S*",
                text,
                re.IGNORECASE,
            )
            if malformed:
                return text, None, "长宽比格式不正确，用法：/draw 比例=16:9 提示词"
            return text, None, None
        if len(matches) > 1:
            return text, None, "一次只能指定一个长宽比，例如：/draw 比例=16:9 提示词"

        numerator = Fraction(matches[0].group(1))
        denominator = Fraction(matches[0].group(2))
        if numerator <= 0 or denominator <= 0:
            return text, None, "长宽比的两个数值都必须大于 0"
        ratio = numerator / denominator
        if not Fraction(1, 4) <= ratio <= Fraction(4, 1):
            return text, None, "长宽比范围为 1:4 到 4:1"
        ratio = ratio.limit_denominator(16)
        remaining = pattern.sub(" ", text)
        remaining = re.sub(r"[ \t]+", " ", remaining).strip()
        return remaining, ratio, None

    def _parse_draw_lora_args(self, text: str):
        option_pattern = re.compile(r"(?<!\S)--lora(?:\s+|=)(\S+)", re.IGNORECASE)
        matches = list(option_pattern.finditer(text))
        if not matches:
            if re.search(r"(?<!\S)--lora(?=\s|=|$)", text, re.IGNORECASE):
                return text, None, "用法：/draw --lora 别名[=M强度] [--lora 别名[=M强度]] 提示词"
            return text, None, None

        tokens = [match.group(1) for match in matches]
        remaining = option_pattern.sub(" ", text)
        remaining = re.sub(r"[ \t]+", " ", remaining).strip()

        off_tokens = [token for token in tokens if token.casefold() in {"off", "none"}]
        if off_tokens:
            if len(tokens) != 1:
                return remaining, None, "关闭 LoRA 时请单独使用 --lora off"
            return remaining, {}, None

        selections = {}
        for token in tokens:
            if "=" in token:
                alias, raw_strength = token.rsplit("=", 1)
                if not alias or not raw_strength:
                    return remaining, None, "LoRA 参数格式应为 别名=M强度，例如 --lora 基础=0.8"
                try:
                    strength = float(raw_strength)
                except (TypeError, ValueError, OverflowError):
                    return remaining, None, f"LoRA M 强度不是有效数字：{raw_strength}"
                if not math.isfinite(strength):
                    return remaining, None, "LoRA M 强度必须是有限数值"
            else:
                alias, strength = token, None

            alias_key = alias.casefold()
            if alias_key in selections:
                return remaining, None, f"LoRA 别名重复：{alias}"
            selections[alias_key] = strength

        available = getattr(self._qwen_engine, "lora_aliases", {})
        unknown = [alias for alias in selections if alias not in available]
        if unknown:
            names = ", ".join(sorted(available))
            return remaining, None, (
                f"未配置 LoRA 别名：{', '.join(unknown)}。"
                f"请在插件配置的 draw_lora_aliases 中添加；当前可用：{names or '无'}"
            )
        return remaining, selections, None

    def _parse_tool_lora_selection(self, lora):
        """解析 LLM Qwen 工具传入的 LoRA 别名列表；省略时保留工作流默认值。"""
        if lora is None:
            return None, None
        if isinstance(lora, str):
            items = [lora]
        elif isinstance(lora, list):
            items = lora
        else:
            return None, "LoRA 选择必须是别名字符串列表。"
        if not items:
            return None, None
        if any(not isinstance(item, str) or not item.strip() for item in items):
            return None, "LoRA 列表中的每项都必须是非空别名字符串。"
        if any(any(char.isspace() for char in item.strip()) for item in items):
            return None, "LoRA 别名不能包含空格；强度写成 别名=0.8。"
        _, selection, error = self._parse_draw_lora_args(
            " ".join(f"--lora={item.strip()}" for item in items)
        )
        return selection, error

    def _parse_params(self, text: str, default_negative: Optional[str] = None) -> tuple:
        """解析用户输入的参数"""
        params = {
            'positive': '',
            'negative': self.default_negative if default_negative is None else default_negative,
            'chain': self.default_chain,
            'width': None,
            'height': None,
            'scale': None,
        }

        chain_pattern = r'(?:chain|转发|合并转发)\s*[:=]?\s*(true|false|是|否|开|关)'
        m = re.search(chain_pattern, text, re.IGNORECASE)
        if m:
            params['chain'] = m.group(1).lower() in ['true', '是', '开']
            text = re.sub(chain_pattern, '', text, flags=re.IGNORECASE).strip()

        scale_pattern = r'(?:scale|倍率|超分|放大)\s*[:=]?\s*(\d+(?:\.\d+)?)'
        m = re.search(scale_pattern, text, re.IGNORECASE)
        if m:
            params['scale'] = float(m.group(1))
            text = re.sub(scale_pattern, '', text, flags=re.IGNORECASE).strip()

        width_pattern = r'(?:\s+|^)(?:宽|宽度|w|width|x)\s*[:=]?\s*(\d+)'
        m = re.search(width_pattern, text, re.IGNORECASE)
        if m:
            params['width'] = int(m.group(1))
            text = re.sub(width_pattern, '', text, flags=re.IGNORECASE).strip()

        height_pattern = r'(?:\s+|^)(?:高|高度|h|height|y)\s*[:=]?\s*(\d+)'
        m = re.search(height_pattern, text, re.IGNORECASE)
        if m:
            params['height'] = int(m.group(1))
            text = re.sub(height_pattern, '', text, flags=re.IGNORECASE).strip()

        positive_aliases = r'(?:正面|正向|正面提示词|正向提示词)'
        negative_aliases = r'(?:负面|反向|负面提示词|反向提示词)'
        new_format_pattern = (
            rf'({positive_aliases})\s*[:=]?\s*[\[{{]([^\]}}]+?)[\]}}]'
            rf'|({negative_aliases})\s*[:=]?\s*[\[{{]([^\]}}]+?)[\]}}]'
        )
        matches = list(re.finditer(new_format_pattern, text, re.IGNORECASE))

        if matches:
            for match in matches:
                if match.group(1):
                    params['positive'] = match.group(2).strip()
                elif match.group(3):
                    params['negative'] = match.group(4).strip()

            if not params['positive']:
                remaining = re.sub(new_format_pattern, '', text, flags=re.IGNORECASE).strip()
                if remaining:
                    params['positive'] = remaining
        else:
            parts = text.split('|')
            params['positive'] = parts[0].strip()
            if len(parts) > 1:
                params['negative'] = parts[1].strip()

        return (params['positive'], params['negative'], params['chain'],
                params['width'], params['height'], params['scale'])

    @staticmethod
    def _extract_polish_prefix(text: str) -> tuple[str, bool]:
        """检测指令文本开头是否包含“润色”前缀（仅带前缀时才启用副脑 LLM 扩写）。"""
        raw = str(text or "").strip()
        if not raw:
            return "", False
        match = re.match(r"^(?:润色|--?(?:polish|enhance))(?:\s*[:：]\s*|\s+)(.+)$", raw, re.IGNORECASE | re.DOTALL)
        if not match:
            return raw, False
        return match.group(1).strip(), True

    async def _enhance_prompt_with_llm(self, event, prompt: str, *, mode: str = "qwen") -> tuple[str, bool]:
        """调用副脑 LLM 对简短提示词进行定向润色扩写；失败或超时时回退原提示词。"""
        text = str(prompt or "").strip()
        if not text:
            return text, False
        system_prompts = {
            "qwen": (
                "你是专业 AI 绘图提示词润色助手。将用户的简短描述扩写为自然、具体、可直接用于文生图的中文画面描述，"
                "合理补充主体外貌与服饰、姿势神态、场景环境、构图视角、光线与色彩氛围以及材质细节。"
                "严格忠实于用户原意，不凭空增加无关人物或情节，不堆砌空泛画质词。只输出润色后的提示词本身，不要任何解释、标题或引号。"
            ),
            "edit": (
                "你是专业 AI 改图指令润色助手。将用户的简短改图要求整理为清晰、可执行、局部可控的中文图像编辑指令，"
                "明确目标对象、修改后的外观材质与光影融合要求，并强调保留未要求修改的主体身份、姿势与背景。"
                "只输出润色后的编辑指令本身，不要任何解释、标题或引号。"
            ),
            "sd": (
                "You are a Danbooru-style anime prompt assistant. Expand the user's short idea into clean, high-quality "
                "comma-separated English tags/phrases for Stable Diffusion anime generation (subject, outfit, pose, expression, "
                "background, lighting, composition). Stay strictly faithful to the user's intent. Output tags only, no explanations or quotes."
            ),
            "video": (
                "你是专业图生视频运镜提示词助手。将用户的简短想法扩写为流畅自然的动态与镜头描述（主体动作、微表情、衣发飘动、光影变化与运镜节奏），"
                "适合图生视频模型执行，不凭空新增无关人物或突兀转场。只输出润色后的提示词本身，不要任何解释、标题或引号。"
            ),
        }
        system_prompt = system_prompts.get(mode, system_prompts["qwen"])
        try:
            provider_id = str(getattr(self, "prompt_enhance_provider_id", "") or "").strip()
            if not provider_id:
                provider_id = str(getattr(self, "llm_provider_id", "") or "").strip()
            if not provider_id:
                provider_id = await self.context.get_current_chat_provider_id(umo=event.unified_msg_origin)
            if not provider_id:
                logger.warning("[ComfyUI] 副脑提示词润色失败：未找到可用的 LLM 提供商")
                return text, False
            response = await asyncio.wait_for(
                self.context.llm_generate(
                    chat_provider_id=provider_id,
                    system_prompt=system_prompt,
                    prompt=text,
                ),
                timeout=15.0,
            )
            enhanced = str(getattr(response, "completion_text", "") or "").strip()
            enhanced = enhanced.strip("`\"'“”").strip()
            if not enhanced or len(enhanced) > 2000:
                return text, False
            return enhanced, True
        except Exception as error:
            logger.warning("[ComfyUI] 副脑提示词润色异常（已回退原提示词）：%s", type(error).__name__)
            return text, False

    @staticmethod
    def _image_task_owner(event):
        return (str(event.unified_msg_origin), str(event.get_sender_id()))

    @staticmethod
    def _image_task_arg(event):
        parts = event.message_str.strip().split(maxsplit=1)
        return parts[1].strip() if len(parts) > 1 else ""
