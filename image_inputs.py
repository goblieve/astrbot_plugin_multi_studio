import asyncio
import uuid
from io import BytesIO
from pathlib import Path

from PIL import Image
from astrbot.api import logger
from astrbot.core.computer.computer_client import get_booter
from astrbot.core.tools.computer_tools.util import workspace_root
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path


def _read_image(path, max_bytes):
    if not path.is_file():
        raise ValueError(f"图片文件不存在：{path}")
    if path.stat().st_size > max_bytes:
        raise ValueError("图片超过每张 20MB 的限制")
    with path.open("rb") as stream:
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("图片超过每张 20MB 的限制")
    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise ValueError(f"文件不是可读取的图片：{path.name}") from error
    return data


async def load_executor_images(context, event, paths, temp_dir, max_bytes, timeout):
    settings = context.get_config(umo=event.unified_msg_origin).get("provider_settings", {})
    if settings.get("computer_use_require_admin", True) and event.role != "admin":
        raise ValueError("代码执行器文件访问仅对管理员开放，请检查 AstrBot 的执行器权限设置")
    runtime = settings.get("computer_use_runtime", "local")
    if runtime == "none":
        raise ValueError("AstrBot 代码执行器已禁用，不能读取执行器图片路径")
    workspace = workspace_root(event.unified_msg_origin)
    host_image_roots = (Path(get_astrbot_temp_path()).resolve(), Path(temp_dir).resolve())
    booter = None
    images = []
    for value in paths:
        if not isinstance(value, str) or not value.strip() or "://" in value:
            raise ValueError("请提供代码执行器中的图片文件路径，而不是 URL")
        path = Path(value).expanduser()
        temporary = None
        try:
            if runtime == "local":
                local_path = (path if path.is_absolute() else workspace / path).resolve()
            elif path.is_absolute() and any(path.resolve().is_relative_to(root) for root in host_image_roots) and path.is_file():
                # AstrBot 下载文件和本插件生成的图片都保存在宿主机。
                local_path = path.resolve()
            else:
                if booter is None:
                    booter = await asyncio.wait_for(get_booter(context, event.unified_msg_origin), timeout)
                temporary = Path(temp_dir) / f"executor_{uuid.uuid4().hex}.image"
                await asyncio.wait_for(booter.download_file(value, str(temporary)), timeout)
                local_path = temporary
            images.append(await asyncio.to_thread(_read_image, local_path, max_bytes))
        except ValueError:
            raise
        except Exception as error:
            logger.error(f"[ComfyUI] 执行器图片读取失败：{error}")
            raise ValueError(f"无法读取执行器图片：{value}，请确认文件存在且属于当前执行环境") from error
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return images
