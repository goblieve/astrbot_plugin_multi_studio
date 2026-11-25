"""Multi-view / character sheet generation engine and prompt presets."""

import asyncio
import json
from pathlib import Path
from typing import Optional

from astrbot.api import logger

from .qwen_image import VisualLoRAControls
from .workflow import randomize_seeds


MULTIVIEW_COMMAND_ALIASES = ("生成多视图", "多视图", "三视图", "multiview")

MULTIVIEW_MODE_LABELS = {
    1: "人物表情（偏动漫/偏浮夸）",
    2: "人物表情（偏写实）",
    3: "动作差分（12张/24姿势）",
    4: "多角度生成（12个机位）",
    5: "动作/多视图单独生成（自定义单图）",
    6: "多角度里图生成（12个机位）",
}

DEFAULT_MULTIVIEW_PROMPTS = {
    "three_view_prompt": (
        "以这张图为【人物/角色】唯一的造型依据，生成她的正面、正侧面、背面三视图：三个等宽的全身立姿视图横向并列，"
        "从左到右依次为正面、正侧面、背面，三格共用整片平整的单一浅色底色背景与同一套光照，人物的头顶与脚底落在同一条水平基线上，"
        "身高、头身比例、体型与镜头距离完全一致，并保持与原图相同的身材比例与体态；三个视图一律双臂自然下垂并略微离开身体、"
        "双脚并拢站立、目视前方，侧视那格只呈现侧脸轮廓且头部不转向镜头，背面那格完整露出后脑、背面发型、头饰与衣物背面的款式及结构；"
        "严格保留原图的面部身份与五官、发型与发色及发饰、眼睛与妆容、上衣、下装、鞋袜及每一件配饰的款式、颜色与数量；"
        "图中看不到的背面与侧面细节按同一套服装结构补齐；保持与输入图一致的媒介、质感与色调（若原图为真人摄影，则保持真实人物皮肤肌理、发丝与真实布料材质，避免3D塑料假人感），画面中不出现任何文字。"
    ),
    "mode1_prompt": (
        "做一张四列两行的表情差分网格图：同一个人物、同一角度、同一景别（胸部以上）、同一背景与同一光照；"
        "八格从左到右、从上到下依次为：正面平静、微笑、大笑、害羞脸红、惊讶张嘴、生气皱眉、闭眼笑、侧头看向镜头；"
        "每一格只改变表情与头部轻微的朝向，发型、服装、配饰、瞳色与画风在八格中完全一致，格与格之间留出均等的浅色间隔；"
        "人物面部特征取自<image1>，保持与输入图一致的媒介、质感与色调。"
    ),
    "mode2_prompt": (
        "做一张四列两行的表情差分网格图：同一个人物、同一角度、同一景别（胸部以上）、同一背景与同一光照；"
        "八格从左到右、从上到下依次为：平静中性、嘴角上扬的微笑、露齿轻笑、眉心收拢的不悦、眉毛上挑眼睛睁大的惊讶、"
        "一侧眉毛扬起并微微眯眼的疑惑、眉毛内端上抬眼角下垂的难过、双眼闭合的放松；每一格的表情都要明显可辨，"
        "靠眉毛的形态、眼睑的开合程度、嘴角的走向与下颌的松紧来拉开差别，同时控制在日常自然的情绪幅度内——"
        "不吐舌、不翻白眼、不做颜艺、不加漫画符号、不夸张拉扯五官；每一格只改变表情与眼神，"
        "头部朝向、发型、服装、配饰、瞳色与整体质感在八格中完全一致，格与格之间留出均等的浅色间隔；"
        "人物面部特征取自<image1>，保持与输入图一致的真实皮肤肌理、发丝质感与色调。"
    ),
    "mode3_prompts": [
        "做一张左右两格的全身动作差分图：同一个人物、同一机位、同一景别（全身）、同一背景与同一光照；左格为踮起脚尖、双臂举过头顶向上伸展；右格为蹲在地上、双手撑在身前的地面、抬头看向镜头；面部与真实皮肤肌理以<image1>中的脸为准（严禁引入<image1>的背景），发型、服装、鞋袜、配饰及全身比例严格参照<image2>中的三视图；每格中只有一个完整人物，头顶与脚底不被裁切，两格之间留出均等的浅色间隔，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，画面里看不到地面、投影与任何道具，人物保持<image1>原本的真实皮肤与织物材质，避免3D塑料假人感。",
        "做一张左右两格的全身动作差分图：同一个人物、同一机位、同一景别（全身）、同一背景与同一光照；左格为向前迈步、双臂随步伐自然前后摆动；右格为坐在地面、双腿屈起、双手环抱膝盖、看向镜头；面部与真实皮肤肌理以<image1>中的脸为准（严禁引入<image1>的背景），发型、服装、鞋袜、配饰及全身比例严格参照<image2>中的三视图；每格中只有一个完整人物，头顶与脚底不被裁切，两格之间留出均等的浅色间隔，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，画面里看不到地面、投影与任何道具，人物保持<image1>原本的真实皮肤与织物材质，避免3D塑料假人感。",
        "做一张左右两格的全身动作差分图：同一个人物、同一机位、同一景别（全身）、同一背景与同一光照；左格为单脚站立、另一条腿屈膝抬起、双臂向两侧张开保持平衡；右格为双膝跪地、双手交叠放在身前、上身微微前倾；面部与真实皮肤肌理以<image1>中的脸为准（严禁引入<image1>的背景），发型、服装、鞋袜、配饰及全身比例严格参照<image2>中的三视图；每格中只有一个完整人物，头顶与脚底不被裁切，两格之间留出均等的浅色间隔，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，画面里看不到地面、投影与任何道具，人物保持<image1>原本的真实皮肤与织物材质，避免3D塑料假人感。",
        "做一张左右两格的全身动作差分图：同一个人物、同一机位、同一景别（全身）、同一背景与同一光照；左格为双臂交叉抱在胸前、重心落在一侧腿上、微微偏头；右格为身体侧向、回过头越过肩膀看向镜头、一只手轻轻拢住耳边的头发；面部与真实皮肤肌理以<image1>中的脸为准（严禁引入<image1>的背景），发型、服装、鞋袜、配饰及全身比例严格参照<image2>中的三视图；每格中只有一个完整人物，头顶与脚底不被裁切，两格之间留出均等的浅色间隔，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，画面里看不到地面、投影与任何道具，人物保持<image1>原本的真实皮肤与织物材质，避免3D塑料假人感。",
        "做一张左右两格的全身动作差分图：同一个人物、同一机位、同一景别（全身）、同一背景与同一光照；左格为双手叉腰、双脚分开与肩同宽、抬头挺胸；右格为俯身趴卧在地面、双肘撑起上半身、双小腿在身后轻轻翘起、看向镜头；面部与真实皮肤肌理以<image1>中的脸为准（严禁引入<image1>的背景），发型、服装、鞋袜、配饰及全身比例严格参照<image2>中的三视图；每格中只有一个完整人物，头顶与脚底不被裁切，两格之间留出均等的浅色间隔，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，画面里看不到地面、投影与任何道具，人物保持<image1>原本的真实皮肤与织物材质，避免3D塑料假人感。",
        "做一张左右两格的全身动作差分图：同一个人物、同一机位、同一景别（全身）、同一背景与同一光照；左格为低头看向自己的手、另一只手自然垂下；右格为跪坐在地上、一只手抬起在脸侧、另一只手放在腿上；面部与真实皮肤肌理以<image1>中的脸为准（严禁引入<image1>的背景），发型、服装、鞋袜、配饰及全身比例严格参照<image2>中的三视图；每格中只有一个完整人物，头顶与脚底不被裁切，两格之间留出均等的浅色间隔，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，画面里看不到地面、投影与任何道具，人物保持<image1>原本的真实皮肤与织物材质，避免3D塑料假人感。",
        "做一张左右两格的全身动作差分图：同一个人物、同一机位、同一景别（全身）、同一背景与同一光照；左格为一只手举高挥手、另一只手自然垂下、身体微微倾向一侧；右格为侧坐在地面、一只手撑地、双腿偏向一侧收拢；面部与真实皮肤肌理以<image1>中的脸为准（严禁引入<image1>的背景），发型、服装、鞋袜、配饰及全身比例严格参照<image2>中的三视图；每格中只有一个完整人物，头顶与脚底不被裁切，两格之间留出均等的浅色间隔，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，画面里看不到地面、投影与任何道具，人物保持<image1>原本的真实皮肤与织物材质，避免3D塑料假人感。",
        "做一张左右两格的全身动作差分图：同一个人物、同一机位、同一景别（全身）、同一背景与同一光照；左格为双手在身前轻握、上身微向前倾、双脚一前一后站立；右格为单膝跪地、一只手搭在抬起的膝盖上、另一只手自然下垂；面部与真实皮肤肌理以<image1>中的脸为准（严禁引入<image1>的背景），发型、服装、鞋袜、配饰及全身比例严格参照<image2>中的三视图；每格中只有一个完整人物，头顶与脚底不被裁切，两格之间留出均等的浅色间隔，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，画面里看不到地面、投影与任何道具，人物保持<image1>原本的真实皮肤与织物材质，避免3D塑料假人感。",
        "做一张左右两格的全身动作差分图：同一个人物、同一机位、同一景别（全身）、同一背景与同一光照；左格为向一侧转身迈步、双臂随转身微微展开；右格为背对镜头奔跑中回头、身体侧转、一条腿向后抬起；面部与真实皮肤肌理以<image1>中的脸为准（严禁引入<image1>的背景），发型、服装、鞋袜、配饰及全身比例严格参照<image2>中的三视图；每格中只有一个完整人物，头顶与脚底不被裁切，两格之间留出均等的浅色间隔，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，画面里看不到地面、投影与任何道具，人物保持<image1>原本的真实皮肤与织物材质，避免3D塑料假人感。",
        "做一张左右两格的全身动作差分图：同一个人物、同一机位、同一景别（全身）、同一背景与同一光照；左格为双手撑地、双膝着地、身体伏低、抬头看向镜头；右格为身体侧转、回头看向镜头、一只手向前平伸、掌心朝下；面部与真实皮肤肌理以<image1>中的脸为准（严禁引入<image1>的背景），发型、服装、鞋袜、配饰及全身比例严格参照<image2>中的三视图；每格中只有一个完整人物，头顶与脚底不被裁切，两格之间留出均等的浅色间隔，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，画面里看不到地面、投影与任何道具，人物保持<image1>原本的真实皮肤与织物材质，避免3D塑料假人感。",
        "做一张左右两格的全身动作差分图：同一个人物、同一机位、同一景别（全身）、同一背景与同一光照；左格为身体侧转、向前倾身、一只手向前伸出像在抓取、另一只手向后摆动；右格为背对镜头跪坐、回头看向镜头、双手放在膝上；面部与真实皮肤肌理以<image1>中的脸为准（严禁引入<image1>的背景），发型、服装、鞋袜、配饰及全身比例严格参照<image2>中的三视图；每格中只有一个完整人物，头顶与脚底不被裁切，两格之间留出均等的浅色间隔，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，画面里看不到地面、投影与任何道具，人物保持<image1>原本的真实皮肤与织物材质，避免3D塑料假人感。",
        "做一张左右两格的全身动作差分图：同一个人物、同一机位、同一景别（全身）、同一背景与同一光照；左格为背对镜头站立、双手交握在身后、肩膀略微下沉；右格为身体侧转蹲下、一只手撑在膝上、回头看向镜头；面部与真实皮肤肌理以<image1>中的脸为准（严禁引入<image1>的背景），发型、服装、鞋袜、配饰及全身比例严格参照<image2>中的三视图；每格中只有一个完整人物，头顶与脚底不被裁切，两格之间留出均等的浅色间隔，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，画面里看不到地面、投影与任何道具，人物保持<image1>原本的真实皮肤与织物材质，避免3D塑料假人感。",
    ],
    "mode4_prompts": [
        "生成她的面部近景特写：镜头正对面部、取景从头顶到锁骨，五官、眼神与表情清晰；面部身份与真实皮肤纹理以<image1>中的脸为准（严禁引入<image1>的背景），发型、发色、头饰及领口结构严格参照<image2>中的三视图；整张画面只呈现这一个人物，面部居中，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，画面里看不到任何额外背景元素，人物保持<image1>原本的真实皮肤与发丝质感，避免3D塑料假人感。",
        "生成她的正面全身像：镜头正对人物、从头顶到脚底完整入镜，身体正对镜头、站姿自然；面部身份与真实皮肤肌理以<image1>中的脸为准（严禁引入<image1>的背景），发型、服装、鞋袜、配饰及全身比例严格参照<image2>中的正视图；整张画面只呈现这一个人物，居中完整入镜，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，画面里看不到地面、投影与任何道具，人物保持<image1>原本的真实皮肤与织物材质，避免3D塑料假人感。",
        "生成她的正侧面全身像：镜头位于人物正左方、与腰部等高水平拍摄，从头顶到脚底完整入镜，身体呈90度正侧面轮廓、脸部也呈完整的正侧脸轮廓，头部不转向镜头；侧脸五官比例与真实皮肤肌理对照<image1>中的脸（严禁引入<image1>的背景），侧面发型、头饰、服装廓形、鞋袜与全身比例严格参照<image2>中的侧视图；整张画面只呈现这一个人物，居中完整入镜，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，画面里看不到地面、投影与任何道具，人物保持<image1>原本的真实皮肤与织物材质，避免3D塑料假人感。",
        "生成她的正背面全身像：镜头位于人物正后方、与腰部等高水平拍摄，从头顶到脚底完整入镜，身体完全背对镜头，完整露出后脑、背面发型、头饰背面、上衣与下装的背部结构及鞋跟，头部保持背向镜头、不回头；背面造型、发色、服装款式与全身比例严格参照<image2>中的背视图（严禁引入<image1>的背景）；整张画面只呈现这一个人物，居中完整入镜，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，画面里看不到地面、投影与任何道具，人物保持<image1>原本的真实发丝与织物材质，避免3D塑料假人感。",
        "从人物头顶正上方垂直向下拍：镜头位于头顶正上方、垂直朝下，画面正中是头顶的发旋与发型顶面，头发与头饰围绕头顶向四周展开，只能看到额前刘海的上沿与鼻尖的一点点前端，看不到正脸五官；头顶下方是向两侧展开的双肩、锁骨与上胸的上表面，身体其余部分被肩胸遮挡在正下方；发色、发型、头饰与肩部服装结构严格参照<image2>中的三视图（严禁引入<image1>的背景）；整张画面只呈现这一个人物，头顶与双肩居中，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，画面里看不到地面、投影与任何道具，人物保持<image1>原本的真实发丝与织物材质，避免3D塑料假人感。",
        "生成她的四十五度侧面近景特写：镜头从人物左前方四十五度方向拍摄、取景从头顶到肩部，脸部呈现四分之三侧向轮廓，一侧脸颊完整展开、另一侧脸颊轻微收窄，鼻梁与下巴的侧向线条清楚；面部五官与真实皮肤肌理以<image1>中的脸为准（严禁引入<image1>的背景）并在四十五度角度下自然呈现，发型、头饰与领口结构严格参照<image2>中的三视图；整张画面只呈现这一个人物，头部居中，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，人物保持<image1>原本的真实皮肤与发丝质感，避免3D塑料假人感。",
        "生成她的左侧半身像：从人物左侧拍摄、取景从头顶到腰胯，身体侧向、面部微微转向镜头；面部五官与真实皮肤肌理以<image1>中的脸为准（严禁引入<image1>的背景），发型、头饰与上半身服装的侧面结构严格参照<image2>中的三视图；整张画面只呈现这一个人物，居中完整入镜，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，人物保持<image1>原本的真实皮肤与织物材质，避免3D塑料假人感。",
        "生成她的背面半身像：从人物正后方拍摄、取景从头顶到腰胯，完整露出后脑、背面发型、头饰背面与上半身服装的背部结构，头部保持背向镜头、不回头；背面造型严格参照<image2>中的背视图（严禁引入<image1>的背景）；整张画面只呈现这一个人物，居中完整入镜，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，人物保持<image1>原本的真实发丝与织物材质，避免3D塑料假人感。",
        "生成她的正面四十五度全身像：镜头从人物左前方四十五度方向拍摄、从头顶到脚底完整入镜，身体朝左前方斜向站立、面部朝向镜头；面部五官与真实皮肤肌理以<image1>中的脸为准（严禁引入<image1>的背景），发型、服装、鞋袜、配饰及全身比例严格参照<image2>中的三视图，同时能看到正面与侧面的服装衔接；整张画面只呈现这一个人物，居中完整入镜，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，画面里看不到地面、投影与任何道具，人物保持<image1>原本的真实皮肤与织物材质，避免3D塑料假人感。",
        "生成她的背面四十五度全身像：镜头从人物左后方四十五度方向拍摄、从头顶到脚底完整入镜，身体背向镜头并带四十五度侧转，同时呈现背面与侧面的发型、服装与身形衔接，头部保持背向或仅露出侧后脸颊轮廓；造型、服装结构与全身比例严格参照<image2>中的背视图与侧视图（严禁引入<image1>的背景）；整张画面只呈现这一个人物，居中完整入镜，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，画面里看不到地面、投影与任何道具，人物保持<image1>原本的真实发丝与织物材质，避免3D塑料假人感。",
        "从人物斜上方四十五度往下拍全身像：机位在人物头顶斜上方、向下四十五度俯视拍摄，从头顶到脚底完整入镜；能看到发顶、额头以及向下延伸的肩部、躯干与双腿，人物呈现上近下远的俯视透视，头部略微抬起看向镜头；面部五官与真实皮肤肌理以<image1>中的脸为准（严禁引入<image1>的背景），发型、头饰、服装、鞋袜与配饰严格参照<image2>中的三视图；整张画面只呈现这一个人物，居中完整入镜，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，画面里看不到地面、投影与任何道具，人物保持<image1>原本的真实皮肤与织物材质，避免3D塑料假人感。",
        "生成她的正侧脸特写：镜头位于人物正左方、与头部等高水平拍摄，取景从头顶到肩部上沿，头部与视线完全朝向画面正右方，丝毫不转向镜头；画面中只出现左半边脸与左眼，右半边脸、右眼与右侧鼻翼被鼻梁完全遮挡而不可见；额头、眉骨、鼻梁、鼻尖、上下唇与下巴连成清晰完整的90度正侧面轮廓线，左耳、耳饰与鬓角发丝位于头部中央偏左的位置，脑后完整展露后脑勺与背面发型的侧向厚度；五官比例与真实皮肤肌理对照<image1>中的脸（严禁引入<image1>的背景），侧面发型、头饰与领口结构严格参照<image2>中的侧视图；整张画面只呈现这一个人物，头部居中，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，人物保持<image1>原本的真实皮肤与发丝质感，避免3D塑料假人感。",
    ],
    "mode5_prompt": (
        "生成她的正面四十五度全身像：镜头从人物左前方四十五度方向拍摄、从头顶到脚底完整入镜，身体朝左前方斜向站立、面部朝向镜头；"
        "面部五官与真实皮肤肌理以<image1>中的脸为准（严禁引入<image1>的背景），发型、服装、鞋袜、配饰及全身比例严格参照<image2>中的三视图，"
        "同时能看到正面与侧面的服装衔接；整张画面只呈现这一个人物，居中完整入镜，背景是整片平整的单一底色、与<image2>的背景完全一致，"
        "完全剔除<image1>中的原始环境背景，画面里看不到地面、投影与任何道具，人物保持<image1>原本的真实皮肤与织物材质，避免3D塑料假人感。"
    ),
    "mode6_prompts": [
        "生成她的面部近景特写：镜头正对面部、取景从头顶到锁骨，五官、眼神与表情清晰；面部身份与真实皮肤纹理以<image1>中的脸为准（严禁引入<image1>的背景），发型、发色、头饰及领口结构严格参照<image2>中的三视图；整张画面只呈现这一个人物，面部居中，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，画面里看不到任何额外背景元素，人物保持<image1>原本的真实皮肤与发丝质感，避免3D塑料假人感。",
        "生成她的头部到胸部正面特写：镜头正对人物、取景从头顶到胸部下缘与上腹部上沿之间，头部居中偏上，胸部为画面视觉重点特写区域，胸廓轮廓、胸部体积、领口与上衣贴合关系清晰；面部身份与真实皮肤肌理以<image1>中的脸为准（严禁引入<image1>的背景），发型、发色、头饰及胸部服装结构严格参照<image2>中的三视图；整张画面只呈现这一个人物，头部与胸部居中，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，画面里看不到任何额外背景元素，人物保持<image1>原本的真实皮肤、发丝与织物材质，避免3D塑料假人感。",
        "生成她的站姿下裆部特写：人物保持自然站姿，镜头从人物正前方略低位置拍摄，取景从腰部下缘到大腿上段，下裆、腹股沟与大腿根部位于画面中心并被重点特写；面部身份与真实皮肤肌理以<image1>中的脸为准（严禁引入<image1>的背景，若面部局部入镜则自然呈现），下裆周围皮肤、下装轮廓、腰带或下摆、袜子及腿部比例严格参照<image2>中的三视图；整张画面只呈现这一个人物，下裆区域居中，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，画面里看不到任何额外背景元素，人物保持<image1>原本的真实皮肤与织物材质，避免3D塑料假人感。",
        "生成她的臀部特写：人物保持自然站姿，镜头位于人物正后方、与臀部等高水平拍摄，取景从腰胯到膝盖上方，臀部位于画面中心，臀部轮廓、臀部体积与背面服装贴合关系清晰；头部保持背向镜头、不回头，背面造型、发色、服装款式与身形比例严格参照<image2>中的背视图（严禁引入<image1>的背景）；整张画面只呈现这一个人物，臀部居中，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，画面里看不到任何额外背景元素，人物保持<image1>原本的真实皮肤、发丝与织物材质，避免3D塑料假人感。",
        "生成她的正面低角度仰拍站姿像：镜头从人物正面偏下方向向上仰拍，机位位于人物脚前偏下、向上仰视拍摄，人物保持自然站姿，从脚底到头顶完整入镜；由于仰拍透视，画面中心自下而上依次可见裆部、胸部与脸部，裆部和胸部离镜头更近、透视更明显，脸部仍位于画面上方且清晰可辨；面部身份与真实皮肤肌理以<image1>中的脸为准（严禁引入<image1>的背景），发型、发色、头饰、服装、袜子、配饰及全身比例严格参照<image2>中的三视图；整张画面只呈现这一个人物，人物居中，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，画面里看不到任何额外背景元素，人物保持<image1>原本的真实皮肤、发丝、织物与鞋履材质，避免3D塑料假人感。",
        "生成她的四十五度侧面近景特写：镜头从人物左前方四十五度方向拍摄、取景从头顶到肩部，脸部呈现四分之三侧向轮廓，一侧脸颊完整展开、另一侧脸颊轻微收窄，鼻梁与下巴的侧向线条清楚；面部五官与真实皮肤肌理以<image1>中的脸为准（严禁引入<image1>的背景）并在四十五度角度下自然呈现，发型、头饰与领口结构严格参照<image2>中的三视图；整张画面只呈现这一个人物，头部居中，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，人物保持<image1>原本的真实皮肤与发丝质感，避免3D塑料假人感。",
        "生成她的左侧半身像：从人物左侧拍摄、取景从头顶到腰胯，身体侧向、面部微微转向镜头；面部五官与真实皮肤肌理以<image1>中的脸为准（严禁引入<image1>的背景），发型、头饰与上半身服装的侧面结构严格参照<image2>中的三视图；整张画面只呈现这一个人物，居中完整入镜，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，人物保持<image1>原本的真实皮肤与织物材质，避免3D塑料假人感。",
        "生成她的背面半身像：从人物正后方拍摄、取景从头顶到腰胯，完整露出后脑、背面发型、头饰背面与上半身服装的背部结构，头部保持背向镜头、不回头；背面造型严格参照<image2>中的背视图（严禁引入<image1>的背景）；整张画面只呈现这一个人物，居中完整入镜，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，人物保持<image1>原本的真实发丝与织物材质，避免3D塑料假人感。",
        "生成她的坐姿脚部特写：人物坐在与<image2>背景完全一致的同色平整表面上，双腿向镜头方向伸展，脚部伸向镜头，镜头从小腿下段到脚尖或脚底取景，脚掌、脚趾或鞋袜细节清晰，腿部由近及远形成强烈透视；面部身份与真实皮肤肌理以<image1>中的脸为准（严禁引入<image1>的背景，若面部未入镜则不作为取景重点），发型、服装、袜子、鞋履、配饰及身体比例严格参照<image2>中的三视图；整张画面只呈现这一个人物，脚部居中且最靠近镜头，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，地面只作为同色平整表面的延伸，不出现具体环境、纹理、投影与任何道具，人物保持<image1>原本的真实皮肤与织物、鞋履材质，避免3D塑料假人感。",
        "生成她的正面坐姿像：人物坐在与<image2>背景完全一致的同色平整表面上，双腿自然分开、两腿向两侧展开，从头顶到脚底完整入镜，脸部正对或轻微转向镜头；面部身份与真实皮肤肌理以<image1>中的脸为准（严禁引入<image1>的背景），发型、发色、头饰、服装、袜子、配饰及全身比例严格参照<image2>中的三视图；整张画面只呈现这一个人物，人物居中，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，地面只作为同色平整表面的延伸，不出现具体环境、纹理、投影与任何道具，人物保持<image1>原本的真实皮肤、发丝、织物与鞋履材质，避免3D塑料假人感。",
        "生成她的跪姿背面像：人物双膝着地呈跪姿，身体背对镜头、臀部朝向镜头，取景从后脑或肩背到膝盖后侧及小腿后侧，腰臀区域居中，臀部轮廓与背面服装贴合关系清晰；头部保持背向镜头、不回头；背面造型、发色、服装款式与身形比例严格参照<image2>中的背视图（严禁引入<image1>的背景）；人物坐在与<image2>背景完全一致的同色平整表面上，只呈现同色地面作为跪姿支撑，不出现具体环境、纹理、投影与任何道具；整张画面只呈现这一个人物，人物居中，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，画面里看不到任何额外背景元素，人物保持<image1>原本的真实发丝、皮肤与织物材质，避免3D塑料假人感。",
        "生成她的坐姿分开腿阴道特写：人物坐在与<image2>背景完全一致的同色平整表面上，双腿分开，镜头从正前方近距离拍摄，取景从大腿内侧、下腹下缘到大腿下段，阴部位于画面中心；画面中可见同一人物的手指位于阴部两侧，用手指轻轻掰开阴唇，露出阴道口与内侧细节，手部位置、指节和阴影真实自然；下装参照<image2>中的款式与材质，必要时被自然撩起或褪至大腿上端，不改变服装材质与整体风格；面部身份与真实皮肤肌理以<image1>中的脸为准（严禁引入<image1>的背景，若面部未入镜则不作为取景重点），发型、服装、袜子及身体比例严格参照<image2>中的三视图；整张画面只呈现这一个人物，双手为同一人物的手，不出现多余手指、多余手或第二个人物；阴部居中，背景是整片平整的单一底色、与<image2>的背景完全一致，完全剔除<image1>中的原始环境背景，地面只作为同色平整表面的延伸，不出现具体环境、纹理、投影与任何道具，人物保持<image1>原本的真实皮肤、私密部位细节与织物材质，避免3D塑料假人感。",
    ],
}


def normalize_multiview_prompts(value) -> dict:
    """校验并规范化多视图 6 种模式的提示词配置。"""
    if not isinstance(value, dict):
        value = {}
    result = {}
    for key in ("three_view_prompt", "mode1_prompt", "mode2_prompt", "mode5_prompt"):
        raw = value.get(key)
        text = str(raw).strip() if isinstance(raw, str) and raw.strip() else DEFAULT_MULTIVIEW_PROMPTS[key]
        if len(text) > 4000:
            raise ValueError(f"{key} 长度不能超过 4000 个字符")
        result[key] = text
    for key in ("mode3_prompts", "mode4_prompts", "mode6_prompts"):
        raw_list = value.get(key)
        defaults = DEFAULT_MULTIVIEW_PROMPTS[key]
        if not isinstance(raw_list, list) or len(raw_list) != 12:
            result[key] = list(defaults)
            continue
        cleaned = []
        for idx, item in enumerate(raw_list):
            text = str(item).strip() if isinstance(item, str) and item.strip() else defaults[idx]
            if len(text) > 4000:
                raise ValueError(f"{key} 第 {idx + 1} 条提示词不能超过 4000 个字符")
            cleaned.append(text)
        result[key] = cleaned
    return result


def _inpaint_crop_inputs(
    image_ref: list,
    mask_ref: list,
    *,
    extend_factor: float = 1.8,
    expand_pixels: int = 8,
    blend_pixels: int = 16,
    target_size: int = 1024,
) -> dict:
    return {
        "image": image_ref,
        "mask": mask_ref,
        "downscale_algorithm": "bilinear",
        "upscale_algorithm": "bicubic",
        "preresize": False,
        "preresize_mode": "ensure minimum resolution",
        "preresize_min_width": 1024,
        "preresize_min_height": 1024,
        "preresize_max_width": 16384,
        "preresize_max_height": 16384,
        "mask_fill_holes": True,
        "mask_expand_pixels": expand_pixels,
        "mask_invert": False,
        "mask_blend_pixels": blend_pixels,
        "mask_hipass_filter": 0.1,
        "extend_for_outpainting": False,
        "extend_up_factor": 1.0,
        "extend_down_factor": 1.0,
        "extend_left_factor": 1.0,
        "extend_right_factor": 1.0,
        "context_from_mask_extend_factor": extend_factor,
        "output_resize_to_target_size": True,
        "output_target_width": target_size,
        "output_target_height": target_size,
        "output_padding": "32",
    }


def _add_orig_face_reference_nodes(workflow: dict) -> list:
    """使用 YOLO 自动检测 <image1> 原图中的人脸并裁出 1024x1024 高清正脸特写，供 ReActor 提取高精度源人脸特征。"""
    workflow.update(
        {
            "face_det_loader": {
                "class_type": "UltralyticsDetectorProvider",
                "inputs": {"model_name": "bbox/face_yolov9c.pt"},
            },
            "ref_orig_face_mask": {
                "class_type": "BboxDetectorCombined_v2",
                "inputs": {
                    "bbox_detector": ["face_det_loader", 0],
                    "image": ["568", 0],
                    "threshold": 0.25,
                    "dilation": 16,
                },
            },
            "ref_orig_face_crop": {
                "class_type": "InpaintCropImproved",
                "inputs": _inpaint_crop_inputs(
                    ["568", 0],
                    ["ref_orig_face_mask", 0],
                    extend_factor=1.8,
                    expand_pixels=8,
                    blend_pixels=16,
                    target_size=1024,
                ),
            },
        }
    )
    return ["ref_orig_face_crop", 1]


def _add_reactor_face_swap_nodes(
    workflow: dict,
    input_image_ref: list,
    source_face_ref: list,
    *,
    two_faces: bool = False,
) -> list:
    """方案 B1：先统一转为 3 通道 RGB（避免 Qwen VAE 的 4 通道 RGBA 在无脸跳过的背面图与已换脸的 3 通道正面图拼批时通道数不一致），再经 ReActor 5 点关键点换脸 + GPEN-BFR-1024 面部还原。"""
    workflow.update(
        {
            "reactor_rgba2rgb": {
                "class_type": "ImageRGBA2RGB",
                "inputs": {"image": input_image_ref},
            },
            "reactor_boost": {
                "class_type": "ReActorFaceBoost",
                "inputs": {
                    "enabled": True,
                    "boost_model": "GPEN-BFR-1024.onnx",
                    "interpolation": "Lanczos",
                    "visibility": 0.85,
                    "codeformer_weight": 0.5,
                    "restore_with_main_after": False,
                },
            },
            "reactor_swap": {
                "class_type": "ReActorFaceSwap",
                "inputs": {
                    "enabled": True,
                    "input_image": ["reactor_rgba2rgb", 0],
                    "source_image": source_face_ref,
                    "swap_model": "inswapper_128.onnx",
                    "facedetection": "retinaface_resnet50",
                    "face_restore_model": "none",
                    "face_restore_visibility": 0.85,
                    "codeformer_weight": 0.5,
                    "detect_gender_input": "no",
                    "detect_gender_source": "no",
                    "input_faces_index": "0,1" if two_faces else "0",
                    "source_faces_index": "0",
                    "console_log_level": 1,
                    "face_boost": ["reactor_boost", 0],
                },
            },
        }
    )
    return ["reactor_swap", 0]


def build_multiview_workflow(
    image_filename: str,
    mode: int,
    prompts_config: dict,
    *,
    custom_prompt: Optional[str] = None,
) -> tuple[dict, str]:
    """按选定模式（1~6）与提示词配置构建精简、无冗余分支的 ComfyUI API 工作流。

    Returns:
        (workflow_dict, output_node_id)
    """
    if mode not in (1, 2, 3, 4, 5, 6):
        raise ValueError("多视图生成模式必须为 1 至 6")
    cfg = normalize_multiview_prompts(prompts_config)

    # 基础加载器与采样公共节点（输入图缩放提升为 2.0 MP 以保留真实皮肤与材质细节）
    workflow = {
        "476": {
            "class_type": "LoadImage",
            "inputs": {"image": image_filename},
        },
        "568": {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {
                "image": ["476", 0],
                "upscale_method": "lanczos",
                "megapixels": 2.0,
                "resolution_steps": 1,
            },
        },
        "479": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "qwen_image_2.1_turbo_int8_convrot.safetensors",
                "weight_dtype": "default",
            },
        },
        "480": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen3vl_8b_int8_convrot.safetensors",
                "type": "qwen_image",
                "device": "default",
            },
        },
        "481": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"},
        },
        "918": {
            "class_type": "VisualLoRALoader",
            "inputs": {
                "model": ["479", 0],
                "lora_data": json.dumps(
                    [
                        {
                            "strength_model": 1.0,
                            "strength_clip": 1.0,
                            "enabled": True,
                            "name": "/Qwenimage21/Qwen-Image-2.1 NSFW Image EditV2.safetensors",
                        },
                        {
                            "strength_model": -1.5,
                            "strength_clip": -1.5,
                            "enabled": True,
                            "name": "/Qwenimage21/breasts_slider_qwen-inage2.1_v1.safetensors",
                        },
                    ],
                    ensure_ascii=False,
                ),
            },
        },
    }

    if mode in (1, 2):
        # 方式 1 / 2：单阶段直接生成 16:9 4MP 八格表情网格图（无需先跑三视图）
        prompt_text = (custom_prompt or "").strip() or (
            cfg["mode1_prompt"] if mode == 1 else cfg["mode2_prompt"]
        )
        workflow.update(
            {
                "13": {
                    "class_type": "ResolutionSelector",
                    "inputs": {
                        "aspect_ratio": "16:9 (Widescreen)",
                        "megapixels": 4.0,
                        "multiple": 32,
                    },
                },
                "816": {
                    "class_type": "EmptyLatentImage",
                    "inputs": {
                        "width": ["13", 0],
                        "height": ["13", 1],
                        "batch_size": 1,
                    },
                },
                "798": {
                    "class_type": "QwenImage21Cache",
                    "inputs": {"model": ["918", 0], "device": "auto", "dtype": "default"},
                },
                "800": {
                    "class_type": "TextEncodeQwenImage21",
                    "inputs": {
                        "clip": ["480", 0],
                        "images.image_1": ["568", 0],
                        "vae": ["481", 0],
                        "prompt": prompt_text,
                        "negative_prompt": "",
                        "resolution": 1024,
                    },
                },
                "806": {
                    "class_type": "CFGGuider",
                    "inputs": {
                        "model": ["798", 0],
                        "positive": ["800", 0],
                        "negative": ["800", 1],
                        "cfg": 1.0,
                    },
                },
                "807": {
                    "class_type": "ManualSigmas",
                    "inputs": {
                        "sigmas": "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0",
                    },
                },
                "809": {
                    "class_type": "KSamplerSelect",
                    "inputs": {"sampler_name": "euler"},
                },
                "810": {
                    "class_type": "RandomNoise",
                    "inputs": {"noise_seed": 1},
                },
                "812": {
                    "class_type": "SamplerCustomAdvanced",
                    "inputs": {
                        "noise": ["810", 0],
                        "guider": ["806", 0],
                        "sampler": ["809", 0],
                        "sigmas": ["807", 0],
                        "latent_image": ["816", 0],
                    },
                },
                "803": {
                    "class_type": "VAEDecode",
                    "inputs": {"samples": ["812", 0], "vae": ["481", 0]},
                },
                "728": {
                    "class_type": "SaveImage",
                    "inputs": {"images": ["803", 0], "filename_prefix": "hub_multiview_emotion"},
                },
            }
        )
        randomize_seeds(workflow)
        return workflow, "728"

    # 先从 <image1> 原图自动检测并裁出 1024x1024 高清正脸特写，供出图后 ReActor 换脸提取高精度源人脸特征
    orig_face_ref = _add_orig_face_reference_nodes(workflow)

    # 方式 3 / 4 / 5：第一阶段先用原图（编码分辨率提升至 1344 以保留更多正/侧脸细节）生成 2:3 4MP 三视图（节点 483）
    workflow.update(
        {
            "826": {
                "class_type": "ResolutionSelector",
                "inputs": {
                    "aspect_ratio": "2:3 (Portrait Photo)",
                    "megapixels": 4.0,
                    "multiple": 32,
                },
            },
            "482": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": ["826", 0],
                    "height": ["826", 1],
                    "batch_size": 1,
                },
            },
            "486": {
                "class_type": "QwenImage21Cache",
                "inputs": {"model": ["918", 0], "device": "auto", "dtype": "default"},
            },
            "699": {
                "class_type": "TextEncodeQwenImage21",
                "inputs": {
                    "clip": ["480", 0],
                    "images.image_1": ["568", 0],
                    "vae": ["481", 0],
                    "prompt": cfg["three_view_prompt"],
                    "negative_prompt": "",
                    "resolution": 1344,
                },
            },
            "767": {
                "class_type": "CFGGuider",
                "inputs": {
                    "model": ["486", 0],
                    "positive": ["699", 0],
                    "negative": ["699", 1],
                    "cfg": 1.0,
                },
            },
            "768": {
                "class_type": "KSamplerSelect",
                "inputs": {"sampler_name": "euler"},
            },
            "769": {
                "class_type": "ManualSigmas",
                "inputs": {
                    "sigmas": "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0",
                },
            },
            "766": {
                "class_type": "RandomNoise",
                "inputs": {"noise_seed": 1},
            },
            "765": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["766", 0],
                    "guider": ["767", 0],
                    "sampler": ["768", 0],
                    "sigmas": ["769", 0],
                    "latent_image": ["482", 0],
                },
            },
            "483": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["765", 0], "vae": ["481", 0]},
            },
            "483_rgb": {
                "class_type": "ImageRGBA2RGB",
                "inputs": {"image": ["483", 0]},
            },
        }
    )

    if mode == 5:
        # 方式 5：生图阶段将 <image1> 与 <image2> 三视图参考编码分辨率提升至 1344（A），出图后再经 ReActor 关键点换脸（B1）
        prompt_text = (custom_prompt or "").strip() or cfg["mode5_prompt"]
        workflow.update(
            {
                "893": {
                    "class_type": "ResolutionSelector",
                    "inputs": {
                        "aspect_ratio": "2:3 (Portrait Photo)",
                        "megapixels": 4.0,
                        "multiple": 32,
                    },
                },
                "816": {
                    "class_type": "EmptyLatentImage",
                    "inputs": {
                        "width": ["893", 0],
                        "height": ["893", 1],
                        "batch_size": 1,
                    },
                },
                "798": {
                    "class_type": "QwenImage21Cache",
                    "inputs": {"model": ["918", 0], "device": "auto", "dtype": "default"},
                },
                "800": {
                    "class_type": "TextEncodeQwenImage21",
                    "inputs": {
                        "clip": ["480", 0],
                        "images.image_1": ["568", 0],
                        "images.image_2": ["483", 0],
                        "vae": ["481", 0],
                        "prompt": prompt_text,
                        "negative_prompt": "",
                        "resolution": 1344,
                    },
                },
                "806": {
                    "class_type": "CFGGuider",
                    "inputs": {
                        "model": ["798", 0],
                        "positive": ["800", 0],
                        "negative": ["800", 1],
                        "cfg": 1.0,
                    },
                },
                "807": {
                    "class_type": "ManualSigmas",
                    "inputs": {
                        "sigmas": "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0",
                    },
                },
                "809": {
                    "class_type": "KSamplerSelect",
                    "inputs": {"sampler_name": "euler"},
                },
                "810": {
                    "class_type": "RandomNoise",
                    "inputs": {"noise_seed": 2},
                },
                "812": {
                    "class_type": "SamplerCustomAdvanced",
                    "inputs": {
                        "noise": ["810", 0],
                        "guider": ["806", 0],
                        "sampler": ["809", 0],
                        "sigmas": ["807", 0],
                        "latent_image": ["816", 0],
                    },
                },
                "803": {
                    "class_type": "VAEDecode",
                    "inputs": {"samples": ["812", 0], "vae": ["481", 0]},
                },
            }
        )
        swapped_ref = _add_reactor_face_swap_nodes(
            workflow,
            ["803", 0],
            orig_face_ref,
            two_faces=False,
        )
        workflow.update(
            {
                "505": {
                    "class_type": "ImageConcanate",
                    "inputs": {
                        "image1": ["483_rgb", 0],
                        "image2": swapped_ref,
                        "direction": "right",
                        "match_image_size": True,
                    },
                },
                "728": {
                    "class_type": "SaveImage",
                    "inputs": {"images": ["505", 0], "filename_prefix": "hub_multiview_custom"},
                },
            }
        )
        randomize_seeds(workflow)
        return workflow, "728"

    # 方式 3（动作差分 12 张，1:1 4MP） / 方式 4 & 6（多角度 12 张，2:3 4MP）
    prompts_12 = (
        cfg["mode3_prompts"]
        if mode == 3
        else (cfg["mode6_prompts"] if mode == 6 else cfg["mode4_prompts"])
    )
    aspect = "1:1 (Square)" if mode == 3 else "2:3 (Portrait Photo)"
    workflow.update(
        {
            "782": {
                "class_type": "easy promptList",
                "inputs": {
                    "prompt_1": prompts_12[0],
                    "prompt_2": prompts_12[1],
                    "prompt_3": prompts_12[2],
                    "prompt_4": prompts_12[3],
                    "prompt_5": prompts_12[4],
                },
            },
            "783": {
                "class_type": "easy promptList",
                "inputs": {
                    "prompt_1": prompts_12[5],
                    "prompt_2": prompts_12[6],
                    "prompt_3": prompts_12[7],
                    "prompt_4": prompts_12[8],
                    "prompt_5": prompts_12[9],
                    "optional_prompt_list": ["782", 0],
                },
            },
            "868": {
                "class_type": "easy promptList",
                "inputs": {
                    "prompt_1": prompts_12[10],
                    "prompt_2": prompts_12[11],
                    "prompt_3": "",
                    "prompt_4": "",
                    "prompt_5": "",
                    "optional_prompt_list": ["783", 0],
                },
            },
            "792": {
                "class_type": "ResolutionSelector",
                "inputs": {
                    "aspect_ratio": aspect,
                    "megapixels": 4.0,
                    "multiple": 32,
                },
            },
            "816": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": ["792", 0],
                    "height": ["792", 1],
                    "batch_size": 1,
                },
            },
            "798": {
                "class_type": "QwenImage21Cache",
                "inputs": {"model": ["918", 0], "device": "auto", "dtype": "default"},
            },
            "800": {
                "class_type": "TextEncodeQwenImage21",
                "inputs": {
                    "clip": ["480", 0],
                    "images.image_1": ["568", 0],
                    "images.image_2": ["483", 0],
                    "vae": ["481", 0],
                    "prompt": ["868", 1],
                    "negative_prompt": "",
                    "resolution": 1344,
                },
            },
            "806": {
                "class_type": "CFGGuider",
                "inputs": {
                    "model": ["798", 0],
                    "positive": ["800", 0],
                    "negative": ["800", 1],
                    "cfg": 1.0,
                },
            },
            "807": {
                "class_type": "ManualSigmas",
                "inputs": {
                    "sigmas": "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0",
                },
            },
            "809": {
                "class_type": "KSamplerSelect",
                "inputs": {"sampler_name": "euler"},
            },
            "810": {
                "class_type": "RandomNoise",
                "inputs": {"noise_seed": 2},
            },
            "812": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["810", 0],
                    "guider": ["806", 0],
                    "sampler": ["809", 0],
                    "sigmas": ["807", 0],
                    "latent_image": ["816", 0],
                },
            },
            "803": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["812", 0], "vae": ["481", 0]},
            },
        }
    )

    # 方案 B1：每张分图生成后经过 ReActor 5 点关键点换脸 + GPEN-BFR-1024 面部还原（方式 3 同时换左右 2 张脸，背面与 90° 大侧脸自动跳过）
    swapped_list_ref = _add_reactor_face_swap_nodes(
        workflow,
        ["803", 0],
        orig_face_ref,
        two_faces=(mode == 3),
    )
    workflow["808"] = {
        "class_type": "easy imageListToImageBatch",
        "inputs": {"images": swapped_list_ref},
    }

    batch_node_ids = ["829", "830", "831", "832", "833", "834", "835", "836", "837", "838", "869", "870"]
    for idx, nid in enumerate(batch_node_ids):
        workflow[nid] = {
            "class_type": "ImageFromBatch",
            "inputs": {"image": ["808", 0], "batch_index": idx, "length": 1},
        }
    workflow.update(
        {
            "862": {
                "class_type": "ImageConcatMulti",
                "inputs": {
                    "inputcount": 4,
                    "direction": "right",
                    "match_image_size": True,
                    "image_1": ["829", 0],
                    "image_2": ["830", 0],
                    "image_3": ["831", 0],
                    "image_4": ["832", 0],
                },
            },
            "863": {
                "class_type": "ImageConcatMulti",
                "inputs": {
                    "inputcount": 4,
                    "direction": "right",
                    "match_image_size": True,
                    "image_1": ["833", 0],
                    "image_2": ["834", 0],
                    "image_3": ["835", 0],
                    "image_4": ["836", 0],
                },
            },
            "865": {
                "class_type": "ImageConcatMulti",
                "inputs": {
                    "inputcount": 4,
                    "direction": "right",
                    "match_image_size": True,
                    "image_1": ["837", 0],
                    "image_2": ["838", 0],
                    "image_3": ["869", 0],
                    "image_4": ["870", 0],
                },
            },
            "866": {
                "class_type": "ImageConcatMulti",
                "inputs": {
                    "inputcount": 3,
                    "direction": "down",
                    "match_image_size": True,
                    "image_1": ["862", 0],
                    "image_2": ["863", 0],
                    "image_3": ["865", 0],
                },
            },
            "727": {
                "class_type": "SaveImage",
                "inputs": {"images": ["483_rgb", 0], "filename_prefix": "hub_multiview_threeview"},
            },
            "728": {
                "class_type": "SaveImage",
                "inputs": {"images": ["866", 0], "filename_prefix": "hub_multiview_sheet12"},
            },
        }
    )
    randomize_seeds(workflow)
    return workflow, ("727", "728")


async def generate_multiview_images(
    api,
    image_bytes: bytes,
    mode: int,
    prompts_config: dict,
    *,
    custom_prompt: Optional[str] = None,
    image_extension: str = "png",
    on_submitted_callback=None,
    on_result_reference=None,
    on_progress=None,
) -> Optional[list[bytes]]:
    """上传单张参考图并按指定模式（1~6）执行多视图生成工作流。

    方式 3 / 4 / 6 返回 2 张图：[首轮三视图, 12张分镜拼图]；
    方式 1 / 2 / 5 返回 1 张图：[成品图]。
    """
    filename = await api.upload_image(f"hub_multiview.{image_extension}", image_bytes)
    if not filename:
        raise ValueError("多视图参考图上传失败")
    workflow, output_nodes = build_multiview_workflow(
        filename, mode, prompts_config, custom_prompt=custom_prompt
    )
    # 模式 3 / 4 / 6 需要生成 1 张三视图 + 12 张分图 + ReActor 关键点换脸，给予充足的等待超时
    max_wait = max(api.timeout, 900) if mode in (3, 4, 6) else api.timeout
    if isinstance(output_nodes, (list, tuple)):
        return await api.queue_and_wait_images(
            workflow,
            list(output_nodes),
            max_wait=max_wait,
            on_submitted_callback=on_submitted_callback,
            on_result_reference=on_result_reference,
            on_progress=on_progress,
        )
    single = await api.queue_and_wait_image(
        workflow,
        max_wait=max_wait,
        on_submitted_callback=on_submitted_callback,
        output_node=output_nodes,
        on_result_reference=on_result_reference,
        on_progress=on_progress,
    )
    return [single] if single else None


async def generate_multiview_image(
    api,
    image_bytes: bytes,
    mode: int,
    prompts_config: dict,
    *,
    custom_prompt: Optional[str] = None,
    image_extension: str = "png",
    on_submitted_callback=None,
    on_result_reference=None,
    on_progress=None,
) -> Optional[bytes]:
    """兼容旧接口：返回最后一张成品图。"""
    results = await generate_multiview_images(
        api, image_bytes, mode, prompts_config,
        custom_prompt=custom_prompt, image_extension=image_extension,
        on_submitted_callback=on_submitted_callback,
        on_result_reference=on_result_reference,
        on_progress=on_progress,
    )
    return results[-1] if results else None
