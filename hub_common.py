import re

try:
    from astrbot.api import web as astrbot_web
except Exception:  # Older AstrBot versions may not expose plugin page APIs.
    astrbot_web = None

DRAW_ALIASES = ('draw',)
SD_DRAW_ALIASES = ('draw_sd',)
CONTINUE_EDIT_ALIASES = ('继续改', '继续编辑', '接着改')
CUTOUT_COMMAND_ALIASES = ('抠图', '去背景', '抠背景', 'rmbg')
UPSCALE_COMMAND_ALIASES = ('放大', '高清修复', '超分', 'upscale')
MULTIVIEW_COMMAND_ALIASES = ('生成多视图', '多视图', '三视图', 'multiview')
DEFAULT_DRAW_LORA_ALIASES = [
    "NSFW基础=Qwen-Image-2.1 NSFW Image EditV2.safetensors",
    "骆驼趾=Qwen21 - Cameltoe.safetensors",
    "内阴=qwen2_1-femaledVaginus-inniepussy-v1_0.safetensors",
    "贫乳=Qwen21 - Flat Chested v1.1.safetensors",
    "胸部大小=breasts_slider_qwen-inage2.1_v1.safetensors",
    "二转三=SinoX_QW2.1_Anything2Real_v2.0_r64.safetensors",
    "连裤袜=QWEN2.1-Opaque_pantyhose_c1-st6000.safetensors",
]
DEFAULT_DRAW_EDIT_OPTION_ROWS = [
    "去除路人=仅移除画面中的路人和无关人物，完整保留主要人物或主体、服装、姿势、构图、背景和光线；自然补全被遮挡的区域。",
    "去除背景=移除原有背景并替换为干净的纯白背景；保持主体完整，保留边缘细节、姿势、服装和受光，不要改变主体本身。",
    "去除主体=移除画面中的主要人物或主体，并依据周围环境自然补全其遮挡区域；保留其余物体、背景、构图和光线。",
    "去水印=仅移除画面中的可见水印（包括半透明叠加水印、角落水印和重复平铺水印），根据周围区域的纹理、颜色、光线与透视自然修复被遮挡部分；保持人物、其他主体、构图、背景、色彩、清晰度及所有非水印文字原样不变，不改动水印以外的区域，不新增内容。",
]
EDIT_OPTION_PLUGIN_NAME = "astrbot_plugin_multi_studio"
EDIT_OPTION_TTL_SECONDS = 600
EDIT_OPTION_MAX_PENDING = 32
IMG2VIDEO_ALIASES = ('img2video', '图生视频', '生视频', 'i2v')

# 用户输入图片下载限制
MAX_INPUT_IMAGE_BYTES = 20 * 1024 * 1024
MASK_IMAGE_FILE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
INPUT_IMAGE_TIMEOUT = 30
RECENT_FILE_TTL = 10 * 60  # 不带引用的“局部重绘”可使用最近 10 分钟内发送的图片文件

GENERATED_MEDIA_TTL_SECONDS = 7 * 24 * 60 * 60
GENERATED_MEDIA_CLEANUP_INTERVAL = 60 * 60
GENERATED_MEDIA_FILENAME = re.compile(r"[0-9a-f]{32}\.(?:png|webp|avif|mp4)")
IMAGE_RESEND_ID = re.compile(r"[0-9a-f]{12}")
IMAGE_RESEND_FILENAME = re.compile(r"[0-9a-f]{12}\.delivery\.json")
COMFY_PROMPT_ID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")
QQ_IMAGE_FILE_THRESHOLD = 20 * 1024 * 1024

# Discord/Telegram 单文件上限
PLATFORM_FILE_SIZE_LIMIT = 10 * 1024 * 1024
SIZE_LIMITED_PLATFORMS = ("discord", "telegram")

LLM_VIOLATION_PATTERNS = [
    re.compile(r'\byes\b', re.IGNORECASE),
    re.compile(r'\bviolation\b', re.IGNORECASE),
    re.compile(r'\bnsfw\b', re.IGNORECASE),
]
LLM_VIOLATION_LITERALS = ('违规', '不安全')


def _is_violation(text: str) -> bool:
    """判断 LLM 回复是否表示违规"""
    cleaned = text.strip()
    # 中文 \b 不工作，所以单独处理"是"
    if cleaned in ('是', 'Yes', 'YES'):
        return True
    for pattern in LLM_VIOLATION_PATTERNS:
        if pattern.search(cleaned):
            return True
    return any(literal in cleaned for literal in LLM_VIOLATION_LITERALS)


def _strip_command_prefix(text: str, aliases: tuple) -> str:
    """剥离命令前缀（包含 / 或 # 等可选前缀），未命中则返回原文"""
    for cmd in aliases:
        pattern = rf'^[\s/#]?{re.escape(cmd)}\s+'
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            return text[match.end():]
        if re.match(rf'^[\s/#]?{re.escape(cmd)}$', text, re.IGNORECASE):
            return ""
    return text
