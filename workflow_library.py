"""Named ComfyUI workflows used by the QQ edit menus."""

import copy
import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from astrbot.api import logger

from .masked_workflow import MASK_ROLES, MaskedWorkflow
from .qwen_image import QwenImage
from .workflow import WorkflowTemplate

WORKFLOW_ID = re.compile(r"[0-9a-f]{32}")
QWEN_ROLES = ("positive", "negative", "width", "height", "latent", "output_image")
MAX_LIBRARY_SIZE = 24
MAX_GRAPH_BYTES = 1024 * 1024


@dataclass
class LibraryWorkflow:
    id: str
    name: str
    kind: str
    graph: dict = field(repr=False)
    engine: QwenImage | MaskedWorkflow = field(repr=False)


class WorkflowLibrary:
    def __init__(self, directory: Path, api, aliases):
        self.directory = directory
        self.api = api
        self.aliases = aliases
        self.entries = {}
        directory.mkdir(exist_ok=True)
        for path in sorted(directory.glob("*.json")):
            if not WORKFLOW_ID.fullmatch(path.stem):
                continue
            try:
                if path.stat().st_size > MAX_GRAPH_BYTES + 8192:
                    raise ValueError("workflow file too large")
                item = json.loads(path.read_text(encoding="utf-8"))
                entry = self._build(path.stem, item["name"], item["graph"], item.get("kind", "qwen"))
                self.entries[entry.id] = entry
            except (OSError, KeyError, TypeError, ValueError) as error:
                logger.warning("[ComfyUI] 工作流库条目 %s 加载失败：%s", path.stem, type(error).__name__)

    def list(self):
        return [
            {
                "id": item.id, "name": item.name, "nodes": len(item.graph), "kind": item.kind,
                **({"mask_loras": item.engine.lora_catalog()} if isinstance(item.engine, MaskedWorkflow) else {}),
            }
            for item in sorted(self.entries.values(), key=lambda item: item.name.casefold())
        ]

    def get(self, workflow_id):
        return self.entries.get(workflow_id)

    @staticmethod
    def inspect(graph):
        if not isinstance(graph, dict) or not graph or "nodes" in graph:
            raise ValueError("请选择 ComfyUI 导出的 API 格式 JSON")
        try:
            if len(json.dumps(graph, ensure_ascii=False, allow_nan=False).encode("utf-8")) > MAX_GRAPH_BYTES:
                raise ValueError("工作流 JSON 最大 1 MiB")
        except RecursionError as error:
            raise ValueError("工作流 JSON 嵌套过深") from error
        if len(graph) > 400:
            raise ValueError("工作流最多 400 个节点")
        for node_id, node in graph.items():
            if (not isinstance(node_id, str) or not isinstance(node, dict)
                    or not isinstance(node.get("class_type"), str)
                    or not isinstance(node.get("inputs"), dict)
                    or not isinstance(node.get("_meta", {}), dict)
                    or not isinstance(node.get("_meta", {}).get("hub", {}), dict)):
                raise ValueError(f"节点 {node_id} 的结构或 _meta.hub 无效")
        return graph

    @staticmethod
    def _required_roles(kind):
        if kind == "qwen":
            return QWEN_ROLES
        if kind == "mask":
            return MASK_ROLES
        raise ValueError("未知工作流类型")

    @classmethod
    def _bind(cls, graph, roles, kind):
        required = cls._required_roles(kind)
        if not isinstance(roles, dict) or set(roles) != set(required):
            raise ValueError("请完成所选工作流类型的全部节点绑定")
        revised = copy.deepcopy(graph)
        for node in revised.values():
            hub = node.get("_meta", {}).get("hub", {})
            for role in (*QWEN_ROLES, *MASK_ROLES):
                hub.pop(role, None)
        used = set()
        for role in required:
            target = roles[role]
            if not isinstance(target, dict):
                raise ValueError(f"缺少绑定：{role}")
            node_id, key = target.get("node"), target.get("input")
            node = revised.get(node_id) if isinstance(node_id, str) else None
            if node is None or (role != "output" and (not isinstance(key, str) or key not in node["inputs"])):
                raise ValueError(f"绑定 {role} 必须指向实际存在的节点输入")
            if role == "output":
                key = None
            if (node_id, key) in used:
                raise ValueError("不同用途不能绑定同一个输入")
            used.add((node_id, key))
            node.setdefault("_meta", {}).setdefault("hub", {})[role] = key
        return revised

    def _build(self, workflow_id, name, graph, kind):
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 40 or any(c in name for c in "\n\r"):
            raise ValueError("工作流名称需为 1 至 40 个字符")
        self.inspect(graph)
        template = WorkflowTemplate.from_graph(graph, required=self._required_roles(kind), name=name.strip())
        if kind == "mask":
            for role in ("image", "mask", "positive"):
                if not isinstance(template.get(graph, role), str):
                    raise ValueError(f"局部重绘的 {role} 必须绑定到文件名或文字字段")
            return LibraryWorkflow(workflow_id, name.strip(), kind, graph, MaskedWorkflow(self.api, template, self.aliases))
        engine = QwenImage(self.api, None, self.aliases, template=template)
        for role in ("latent", "output_image"):
            link = template.get(graph, role)
            if not (isinstance(link, list) and len(link) == 2 and isinstance(link[0], str)
                    and link[0] in graph and isinstance(link[1], int) and link[1] >= 0):
                raise ValueError(f"绑定 {role} 必须是已有的节点连线")
        return LibraryWorkflow(workflow_id, name.strip(), kind, graph, engine)

    def save(self, name, graph, roles, workflow_id, kind="qwen"):
        if workflow_id and (not isinstance(workflow_id, str) or workflow_id not in self.entries):
            raise ValueError("找不到要替换的工作流")
        if workflow_id and self.entries[workflow_id].kind != kind:
            raise ValueError("替换时需保持工作流类型一致，切换类型请新建条目")
        if not workflow_id and len(self.entries) >= MAX_LIBRARY_SIZE:
            raise ValueError(f"工作流库最多 {MAX_LIBRARY_SIZE} 项")
        workflow_id = workflow_id or uuid.uuid4().hex
        graph = self._bind(self.inspect(graph), roles, kind)
        entry = self._build(workflow_id, name, graph, kind)
        if any(item.name.casefold() == entry.name.casefold() and item.id != workflow_id
               for item in self.entries.values()):
            raise ValueError("工作流名称已存在")
        path = self.directory / f"{entry.id}.json"
        staged = path.with_suffix(".tmp")
        try:
            staged.write_text(
                json.dumps({"name": entry.name, "graph": entry.graph, "kind": entry.kind}, ensure_ascii=False),
                encoding="utf-8",
            )
            staged.replace(path)
        finally:
            staged.unlink(missing_ok=True)
        self.entries[entry.id] = entry
        return entry

    def delete(self, workflow_id):
        if not isinstance(workflow_id, str) or workflow_id not in self.entries:
            raise ValueError("找不到工作流")
        (self.directory / f"{workflow_id}.json").unlink(missing_ok=True)
        del self.entries[workflow_id]
