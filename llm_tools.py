from astrbot.core.agent.tool import FunctionTool
from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor


class ComfyUITool(FunctionTool):
    """通过 call 接口执行，避免插件加载器再次绑定已绑定的实例方法。"""

    def __init__(self, callback, background, **kwargs):
        super().__init__(**kwargs)
        self._callback = callback
        self._background = background

    async def call(self, context, **kwargs):
        if not self._background:
            return await self._callback(context.context.event, **kwargs)
        # 权限包装器可能丢失后台标记，在权限检查后的 call 内交给原生执行器。
        task = FunctionTool(
            name=self.name, description=self.description, parameters=self.parameters,
            handler=self._callback, is_background_task=True,
        )
        async for result in FunctionToolExecutor.execute(task, context, **kwargs):
            if result is not None:
                return result

COMPLETION_FEEDBACK_HINT = (
    "文字反馈可选：按用户需求和会话语境决定是否补充一句；无需反馈时可不发消息。"
    "media_sent=true 仅表示媒体已发送，不代表禁止文字反馈；不要重复发送同一媒体。"
    "若选择让用户收到文字反馈，使用 send_message_to_user；仅输出内部总结不会发送到聊天。"
    "依据 status 和 message 说明实际结果，不把已提交当作成功，不在未查看媒体时评价画面或视频内容。"
    "不要自动再次生成；delivery_failed 表示已有产物但发送失败，可处理现有文件。"
    "file_path 是内部产物路径，除非用户索要或后续操作需要，不主动展示路径、任务 ID 或实现细节。"
)

IMAGE_COMPLETION_FEEDBACK_HINT = (
    "图片生成或编辑成功且 media_sent=true 时，图片已经发送到当前会话；不要再调用 send_message_to_user 发送完成通知、复述画面，或提到“没有重复发图”。默认不补充文字。"
    "只有用户明确要求文字说明，或工具返回了需要用户处理的 warning 时，才发送必要的简短说明。"
    "失败时依据 status 和 message 简要说明实际情况；delivery_failed 表示图片已生成但未成功发送，不要自动再次生成。"
    "不要评价未查看的图片内容；file_path 是内部路径，除非用户索要或后续操作需要，不主动展示路径或任务 ID。"
)


def create_tools(plugin):
    prompt = {
        "type": "string", "minLength": 1,
        "description": "把简短或口语化需求整理成完整、具体、可直接用于生图的画面描述。按需求补足相关的主体外观与服饰、动作姿势、场景、构图、光线、色彩和材质细节，使画面元素协调一致。严格保留用户指定的主体数量、年龄、身份、动作、风格、文字、比例和位置关系；不编造无依据的关键人物、道具、情节、标识或文字。不确定的细节保持开放，不堆叠空泛画质词。中文或英文均可。",
    }
    edit_prompt = {
        "type": "string", "minLength": 1,
        "description": "把简短的改图要求整理成具体可执行的编辑提示，明确目标对象或区域、要改成什么样，并补足相关的颜色、材质、纹理、细节及与原图光线和透视融合的要求。结合图片说明需要保留的身份特征、姿势、构图、背景和其他未要求修改的部分；不新增未要求的人物、道具、文字或情节，不扩大修改范围。多图时用 <image1>、<image2> 等标明各图用途，编号按最终输入顺序。中文或英文均可。",
    }
    negative = {"type": "string", "description": "可选的不希望出现的内容。未明确需要时省略，不机械套用通用负面词，也不要否定用户要求的主体或效果。"}
    size = {
        "width": {"type": "integer", "description": "可选输出宽度（像素）。仅在用户指定尺寸或需要特定画幅时填写；否则省略，不随意设置为 512。"},
        "height": {"type": "integer", "description": "可选输出高度（像素）。仅在用户指定尺寸或需要特定画幅时填写；否则省略，沿用工作流。"},
        "scale": {"type": "number", "description": "仅当用户要求放大时填写，例如 2。否则省略，不默认追加放大操作。"},
    }
    image_index = {"type": "integer", "minimum": 1, "default": 1, "description": "当前消息及引用消息中的图片序号，从 1 开始；未指定时使用第一张。"}
    image_path = {
        "type": "string",
        "description": "已知存在的代码执行器图片路径，本地或当前会话沙箱均可。使用工具实际返回的路径，不猜测、不传 URL；相对路径按执行器工作区解析。指定后不使用消息图片序号。",
    }
    configured_loras = sorted(getattr(plugin._qwen_engine, "lora_aliases", {}))
    lora = {
        "type": "array",
        "items": {"type": "string"},
        "maxItems": 16,
        "description": (
            "可选，只在用户明确指定 Qwen /draw LoRA 时传入别名列表；普通别名启用对应项，"
            "别名=0.8 会把该 LoRA 的 M（strength_model）和 C（strength_clip）都设为 0.8；"
            "[\"off\"] 关闭工作流中全部 LoRA。用户未指定时省略此字段以保留工作流默认开关。"
            + (" 当前可用别名：" + ", ".join(configured_loras) + "。" if configured_loras else "")
        ),
    }

    def tool(name, description, handler, properties, required, active, background=True, completion_hint=None):
        if background:
            description += (
                " 后台提交返回任务 ID，不代表已完成，不要重复提交。"
                "完成结果回传 status、file_path、media_sent、message 和 feedback_hint；"
            )
            description += completion_hint or "媒体会发送到当前会话，额外文字反馈可选，按实际结果和反馈建议处理。"
        return ComfyUITool(
            name=name, description=description, callback=handler, active=active,
            background=background,
            parameters={"type": "object", "properties": properties, "required": required, "additionalProperties": False},
        )

    return [
        tool("comfyui_generate_image",
             "自然语言文生图工具：无论用户要求写实、动漫或二次元画风，都使用 Qwen 工作流；保留用户给出的主体、数量、动作、构图与文字，不自行改用动漫/SD 工作流。动漫/SD 工作流仅由用户显式输入 /draw_sd 指令触发。需要依据现有图片时调用 comfyui_edit_image；本工具不使用输入图片。若用户明确指定 LoRA（包括 /draw --lora 别名 或 --lora 别名=0.8），必须把别名放入 lora 列表；off 表示关闭全部。用户未指定时省略该字段。",
             plugin.tool_generate_image, {"prompt": prompt, "negative": negative, "lora": lora, **size}, ["prompt"], plugin._qwen_engine is not None, completion_hint=IMAGE_COMPLETION_FEEDBACK_HINT),
        tool("comfyui_edit_image",
             "参考图编辑、合成及参考图驱动的生成。需要修改已有图、保留主体或参考多图时使用，动漫图片也适用。输入来自当前/引用消息、已知的执行器图片路径，或 Synology Chat 中同一用户在当前会话最近上传的一张图；缺图时请用户提供图片，不编造输入，也不改成纯文生图。若用户明确指定 LoRA（包括 `/draw --lora 别名` 或 `--lora 别名=0.8`），必须把别名放入 lora 列表；`off` 表示关闭全部。用户未指定时省略该字段。",
             plugin.tool_edit_image, {"prompt": edit_prompt, "negative": negative, "image_indices": {
                 "type": "array", "items": {"type": "integer", "minimum": 1},
                 "description": "可选消息图片序号列表，按给定顺序选择。没有文件路径且省略或留空时使用全部消息图片。最终最多 16 张参考图。",
             }, "image_paths": {
                 "type": "array", "items": {"type": "string"}, "maxItems": 16,
                 "description": "已知存在的执行器图片路径列表，不猜测路径。单独提供时只用文件图；与 image_indices 同时提供时，选中的消息图在前，文件图在后；最终顺序依次对应 <image1> 等。",
             }, "lora": lora, **size}, ["prompt"], plugin._qwen_engine is not None, completion_hint=IMAGE_COMPLETION_FEEDBACK_HINT),
        tool("comfyui_image_to_video",
             "将一张已有图片生成为视频。可使用当前/引用消息图片、已知的执行器图片路径，或 Synology Chat 中同一用户在当前会话最近上传的一张图；缺图时请用户提供，不调用文生图来擅自补图。主要描述期望的运动与镜头变化，保持用户要求的主体。",
             plugin.tool_image_to_video, {"prompt": {"type": "string", "description": "运动或镜头指令，可为空；不擅自增加角色、剧情或转场。"}, "negative": negative, "image_index": image_index, "image_path": image_path,
                 "fps": {"type": "number", "description": "仅在用户要求帧率时填写；否则省略，沿用工作流。"},
                 "length": {"type": "number", "description": "用户要求的视频秒数；未指定时省略。总帧数受配置限制。"}}, [], plugin._img2video_engine is not None),
        tool("comfyui_image_tagger",
             "提取一张已有图片的 Danbooru 风格标签，支持 Synology Chat 当前会话最近上传的图片，直接返回给 LLM，可用于组织绘图提示词。仅在用户要求识别标签、提取提示词或确实需要这些标签时调用，不作为每次绘图的固定前置步骤。标签是识别结果，不等于可靠的文字识别、精确空间关系或完整图像描述；缺图时请用户提供图片。",
             plugin.tool_image_tagger, {"image_index": image_index, "image_path": image_path}, [], plugin.img2txt is not None, background=False),
    ]
