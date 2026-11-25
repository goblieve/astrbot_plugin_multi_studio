from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
import asyncio
import time
import uuid


TASK_LABELS = {
    "queued": "排队中", "generating": "处理中", "ready": "已生成，等待发送",
    "sending": "发送中", "sent": "已发送", "delivery_failed": "发送未确认", "result_unavailable": "可尝试取回",
    "failed": "失败", "cancelled": "已取消",
}
FINISHED = {"sent", "delivery_failed", "result_unavailable", "failed", "cancelled"}


def infer_model_group(backend: str) -> str:
    """将后端名称归类为显存模型组，用于队列同模型亲和调度，减少大模型频繁换出换入。"""
    text = str(backend or "").strip().casefold()
    if not text:
        return "other"
    if any(k in text for k in ("qwen", "放大", "局部重绘", "多视图", "改图")):
        return "qwen"
    if "视频" in text or "video" in text or "wan" in text:
        return "video"
    if "自动遮罩" in text or "sam" in text:
        return "sam3"
    if "标签" in text or "tagger" in text:
        return "tagger"
    if "抠图" in text or "cutout" in text:
        return "cutout"
    if "sd" in text or "文生图" in text:
        return "sd"
    return text


@dataclass(eq=False)
class ImageTask:
    owner: tuple[str, str]
    backend: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "queued"
    created: float = field(default_factory=time.monotonic)
    started: float | None = None
    generated: float | None = None
    send_started: float | None = None
    finished: float | None = None
    path: Path | None = None
    chain: bool = False
    comfy_prompt_id: str | None = None
    comfy_output_node: str | None = None
    progress: tuple[int, int] | None = None
    stage: str = ""
    model_group: str = ""
    skips: int = 0

    def __post_init__(self):
        if not self.model_group:
            self.model_group = infer_model_group(self.backend)

    def finish(self, status):
        self.status = status
        self.finished = time.monotonic()

    def timings(self):
        return {
            "queue": round(self.started - self.created, 2) if self.started is not None else None,
            "generation": round(self.generated - self.started, 2) if self.generated is not None else None,
            "send": round(self.finished - self.send_started, 2)
            if self.finished is not None and self.send_started is not None else None,
        }


class LocalImageGenerationQueue:
    ANTI_STARVATION_SECONDS = 180.0
    MAX_AFFINITY_SKIPS = 2

    def __init__(self):
        self._condition = asyncio.Condition()
        self._tickets = []
        self._history = OrderedDict()
        self._last_model_group: str = ""
        self._last_activity: float | None = None
        self._vram_dirty: bool = False

    def _prune(self):
        cutoff = time.monotonic() - 7 * 86400
        completed = [task for task in self._history.values() if task.status in FINISHED]
        for task in completed:
            if task.finished < cutoff:
                self._history.pop(task.id, None)
        completed = [task for task in completed if task.id in self._history]
        for task in completed[:-300]:
            self._history.pop(task.id, None)

    def _reorder_for_affinity(self):
        """在防饥饿约束下优先连续执行同模型组任务，避免 Qwen / Wan 等大模型反复加载卸载。"""
        if len(self._tickets) < 2 or not self._last_model_group:
            return
        head = self._tickets[0]
        if head.status != "queued" or head.model_group == self._last_model_group:
            return
        if head.skips >= self.MAX_AFFINITY_SKIPS or (time.monotonic() - head.created) >= self.ANTI_STARVATION_SECONDS:
            return
        for idx in range(1, len(self._tickets)):
            candidate = self._tickets[idx]
            if candidate.status == "queued" and candidate.model_group == self._last_model_group:
                for skipped in self._tickets[:idx]:
                    skipped.skips += 1
                self._tickets.pop(idx)
                self._tickets.insert(0, candidate)
                return

    async def enqueue(self, owner, backend, *, model_group: str = ""):
        ticket = ImageTask(owner, backend, model_group=model_group)
        async with self._condition:
            self._prune()
            self._history[ticket.id] = ticket
            self._tickets.append(ticket)
            return ticket, len(self._tickets) - 1

    async def wait_turn(self, ticket):
        async with self._condition:
            await self._condition.wait_for(
                lambda: ticket.status == "cancelled" or (self._tickets and self._tickets[0] is ticket)
            )
            if ticket.status == "cancelled":
                return False
            ticket.started = time.monotonic()
            ticket.status = "generating"
            self._last_model_group = ticket.model_group
            self._last_activity = ticket.started
            self._vram_dirty = True
            return True

    async def release(self, ticket):
        async with self._condition:
            if ticket in self._tickets:
                self._tickets.remove(ticket)
                self._last_activity = time.monotonic()
                self._reorder_for_affinity()
                self._condition.notify_all()

    def idle_seconds_needing_free(self) -> float | None:
        if self._tickets or not self._vram_dirty or self._last_activity is None:
            return None
        return max(0.0, time.monotonic() - self._last_activity)

    def mark_vram_freed(self):
        self._vram_dirty = False

    async def cancel(self, ticket):
        async with self._condition:
            if ticket is None or ticket.status not in ("queued", "generating") or ticket not in self._tickets:
                return None
            previous = ticket.status
            ticket.finish("cancelled")
            if previous == "queued":
                self._tickets.remove(ticket)
                self._condition.notify_all()
            return previous

    def ahead(self, ticket):
        return self._tickets.index(ticket) if ticket.status == "queued" and ticket in self._tickets else None

    def count_active(self, owner) -> int:
        return sum(
            1 for task in self._tickets
            if task.owner == owner and task.status in ("queued", "generating")
        )

    def estimate_duration(self, backend: str, model_group: str = "") -> float | None:
        """根据近期已完成同类型任务的耗时估算单任务总耗时（秒）。"""
        same_backend = [
            task.generated - task.started
            for task in self._history.values()
            if task.backend == backend and task.started is not None and task.generated is not None
            and task.generated > task.started
        ]
        if same_backend:
            recent = same_backend[-8:]
            return sum(recent) / len(recent)
        group = model_group or infer_model_group(backend)
        same_group = [
            task.generated - task.started
            for task in self._history.values()
            if task.model_group == group and task.started is not None and task.generated is not None
            and task.generated > task.started
        ]
        if same_group:
            recent = same_group[-8:]
            return sum(recent) / len(recent)
        return None

    def recent(self, owner=None, limit=5):
        self._prune()
        return [
            task for task in reversed(self._history.values())
            if owner is None or task.owner == owner
        ][:limit]

    def find(self, owner, task_id, statuses=()):
        if task_id:
            self._prune()
            task = self._history.get(task_id.casefold())
            return task if task and task.owner == owner and (not statuses or task.status in statuses) else None
        return next(
            (task for task in self.recent(owner, limit=len(self._history)) if task.status in statuses),
            None,
        )
