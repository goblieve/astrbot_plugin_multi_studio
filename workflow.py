import copy
import json
import random
from pathlib import Path


class WorkflowTemplate:
    def __init__(self, path: str, required=()):
        self.path = Path(path)
        with self.path.open(encoding="utf-8-sig") as stream:
            self.workflow = json.load(stream)
        self._validate_bindings(required)

    @classmethod
    def from_graph(cls, graph, required=(), name="workflow"):
        template = cls.__new__(cls)
        template.path = Path(name)
        template.workflow = copy.deepcopy(graph)
        template._validate_bindings(required)
        return template

    def _validate_bindings(self, required):
        if not isinstance(self.workflow, dict) or "nodes" in self.workflow:
            raise ValueError(f"{self.path.name} 必须是 ComfyUI API 格式工作流")
        self.bindings = {}
        for node_id, node in self.workflow.items():
            if not isinstance(node, dict) or not isinstance(node.get("class_type"), str) or not isinstance(node.get("inputs"), dict):
                raise ValueError(f"{self.path.name} 的节点 {node_id} 无效")
            metadata = node.get("_meta", {})
            if not isinstance(metadata, dict) or not isinstance(metadata.get("hub", {}), dict):
                raise ValueError(f"{self.path.name} 的节点 {node_id} 的 _meta.hub 必须是对象")
            for role, field in metadata.get("hub", {}).items():
                if not isinstance(field, str) and not (role == "output" and field is None):
                    raise ValueError(f"{self.path.name} 的绑定 {role} 必须指定输入字段；仅 output 可使用 null")
                if role in self.bindings:
                    raise ValueError(f"{self.path.name} 的绑定 {role} 重复")
                if field is not None and field not in node["inputs"]:
                    raise ValueError(f"{self.path.name} 的绑定 {role} 指向不存在的字段 {field}")
                self.bindings[role] = (node_id, field)
        missing = set(required) - self.bindings.keys()
        if missing:
            raise ValueError(f"{self.path.name} 缺少绑定：{', '.join(sorted(missing))}")

    def copy(self):
        return copy.deepcopy(self.workflow)

    def node_id(self, role):
        return self.bindings[role][0]

    def get(self, workflow, role):
        if role not in self.bindings:
            return None
        node_id, field = self.bindings[role]
        return workflow[node_id]["inputs"][field]

    def set(self, workflow, role, value):
        node_id, field = self.bindings[role]
        workflow[node_id]["inputs"][field] = value


def randomize_seeds(workflow):
    seed = random.randint(1, 999999999999999)
    for node in workflow.values():
        for field in ("seed", "noise_seed"):
            if isinstance(node["inputs"].get(field), (int, float)):
                node["inputs"][field] = seed
                seed += 1
