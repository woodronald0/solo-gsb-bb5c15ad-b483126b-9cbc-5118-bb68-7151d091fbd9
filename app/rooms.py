"""会议室批量分配核心逻辑：房间 upsert/查询与预约的全局确定性房间分配。

- 房间（meeting room）以客户端提供的 ``room_id`` 唯一标识，带容量（座位数）
  与一组**去重字典序排序**的设备标签；``upsert_room`` 按 room_id 插入或更新。
- 批量分配（batch assign）：提交一组 active 预约及各自所需容量与设备标签，
  结合**已有分配**（本批未涉及的预约视为固定占用）与预约时段做全局求解：
  同一房间不得分给时间半开区间 ``[start, end)`` 重叠的两条预约；
  房间容量须满足预约所需容量，且房间设备标签须**包含**预约要求的全部标签。
- 仅当整批都存在可行解时，才在单个 ``BEGIN IMMEDIATE`` 事务内把本批的
  分配结果（对已有分配是原子替换）持久化；无解则抛出
  :class:`BatchAssignmentInfeasibleError`（API 映射为 409，列出无法分配的
  预约），事务整体回滚，**不改动任何已有分配**。
- 结果确定性：在全部可行解中优先**总空余座位最少**
  （sum(room.capacity - required_capacity)，仅计本批预约；等价于最小化
  所选房间容量之和）；总空余并列时取按 (reservation_id 升序、room_id 字典序)
  的完整分配向量字典序最小解。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from . import db
from .reservations import (
    ReservationNotFoundError,
    _canon,
    _parse,
)

_ROOM_COLUMNS = "room_id, capacity, equipment_key, created_at, updated_at"


class RoomNotFoundError(Exception):
    """指定 room_id 的会议室不存在。"""

    def __init__(self, room_id: str) -> None:
        self.room_id = room_id
        super().__init__(f"meeting room {room_id!r} not found")


class BatchAssignmentInfeasibleError(Exception):
    """整批预约不存在可行的房间分配。"""

    def __init__(self, unassignable: list[dict[str, object]]) -> None:
        # 每项形如 {"reservation_id": ..., "required_capacity": ...,
        # "required_equipment": [...], "reason": ...}
        self.unassignable = unassignable
        ids = ", ".join(f"#{item['reservation_id']}" for item in unassignable)
        super().__init__(
            f"meeting room batch assignment infeasible for reservation(s): {ids}"
        )


@dataclass(frozen=True)
class Room:
    room_id: str
    capacity: int
    equipment: tuple[str, ...]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class BatchRequest:
    reservation_id: int
    required_capacity: int
    required_equipment: tuple[str, ...]


@dataclass(frozen=True)
class AssignmentView:
    """分配查询结果：分配行 + 预约窗口与房间快照。"""

    id: int
    reservation_id: int
    room_id: str
    created_at: datetime
    reservation_request_id: str
    start_utc: datetime
    end_utc: datetime
    status: str
    room_capacity: int
    room_equipment: tuple[str, ...]


def normalize_equipment(tags: Iterable[str]) -> tuple[str, ...]:
    """设备标签去重 + 字典序排序。"""
    return tuple(sorted(set(tags)))


def _row_to_room(row: sqlite3.Row) -> Room:
    return Room(
        room_id=row["room_id"],
        capacity=row["capacity"],
        equipment=tuple(json.loads(row["equipment_key"])),
        created_at=_parse(row["created_at"]),
        updated_at=_parse(row["updated_at"]),
    )


def upsert_room(
    room_id: str,
    capacity: int,
    equipment_tags: Iterable[str],
    db_path: str | None = None,
) -> tuple[Room, bool]:
    """按 ``room_id`` 创建或更新会议室，返回 ``(房间, 是否新建)``。

    整个 upsert 在单个 ``BEGIN IMMEDIATE`` 事务内完成：room_id 已存在则更新
    容量与设备标签（``updated_at`` 推进、``created_at`` 保留），否则插入新行。
    设备标签去重排序后存储。
    """
    equipment = normalize_equipment(equipment_tags)
    equipment_key = json.dumps(list(equipment), ensure_ascii=False)

    path = db.ensure_initialized(db_path)
    conn = db.connect(path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            now_s = _canon(datetime.now(timezone.utc))
            row = conn.execute(
                f"SELECT {_ROOM_COLUMNS} FROM meeting_rooms WHERE room_id = ?",
                (room_id,),
            ).fetchone()
            created = row is None
            if created:
                conn.execute(
                    "INSERT INTO meeting_rooms"
                    " (room_id, capacity, equipment_key, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (room_id, capacity, equipment_key, now_s, now_s),
                )
                created_at_s = now_s
            else:
                conn.execute(
                    "UPDATE meeting_rooms"
                    " SET capacity = ?, equipment_key = ?, updated_at = ?"
                    " WHERE room_id = ?",
                    (capacity, equipment_key, now_s, room_id),
                )
                created_at_s = row["created_at"]
            conn.execute("COMMIT")
            return Room(
                room_id=room_id,
                capacity=capacity,
                equipment=equipment,
                created_at=_parse(created_at_s),
                updated_at=_parse(now_s),
            ), created
        except BaseException:
            # 无活动事务时 rollback 为 no-op；任何失败都保证整体回滚
            conn.rollback()
            raise
    finally:
        conn.close()


def get_room(room_id: str, db_path: str | None = None) -> Room:
    """按 ``room_id`` 查询单个会议室；不存在 → :class:`RoomNotFoundError`。"""
    path = db.ensure_initialized(db_path)
    conn = db.connect(path)
    try:
        row = conn.execute(
            f"SELECT {_ROOM_COLUMNS} FROM meeting_rooms WHERE room_id = ?",
            (room_id,),
        ).fetchone()
        if row is None:
            raise RoomNotFoundError(room_id)
        return _row_to_room(row)
    finally:
        conn.close()


def list_rooms(db_path: str | None = None) -> list[Room]:
    """列出全部会议室，按 ``room_id`` 字典序升序。"""
    path = db.ensure_initialized(db_path)
    conn = db.connect(path)
    try:
        rows = conn.execute(
            f"SELECT {_ROOM_COLUMNS} FROM meeting_rooms ORDER BY room_id ASC"
        ).fetchall()
        return [_row_to_room(row) for row in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------- 批量求解


def _overlaps(a_start: str, a_end: str, b_start: str, b_end: str) -> bool:
    """半开区间 ``[a_start, a_end)`` 与 ``[b_start, b_end)`` 是否重叠。"""
    return a_start < b_end and b_start < a_end


@dataclass
class _SolveInputs:
    # reservation_id -> 预约窗口（定宽 UTC 文本）
    windows: dict[int, tuple[str, str]]
    # reservation_id -> 本批需求
    requests: dict[int, BatchRequest]
    # room_id -> Room
    rooms: dict[str, Room]
    # room_id -> 固定占用 [(reservation_id, start, end)]（本批未涉及的 active 预约）
    fixed: dict[str, list[tuple[int, str, str]]]
    # 候选集（硬阻断项为空）：reservation_id -> [room_id ...]（容量/设备过滤后）
    candidates: dict[int, list[str]]
    # 硬阻断原因（非 active / 无任何满足容量与设备的房间）
    blocked: dict[int, str]


def _load_solve_inputs(
    conn: sqlite3.Connection, requests: list[BatchRequest]
) -> _SolveInputs:
    """在调用方事务内加载求解所需的全部现状数据。

    本批重复的 reservation_id 由 API 层（Pydantic）拦截；这里同样防御。
    """
    req_by_id = {req.reservation_id: req for req in requests}
    if len(req_by_id) != len(requests):
        raise ValueError("duplicate reservation_id in batch assignment request")

    ids = tuple(req_by_id)
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        "SELECT id, start_utc, end_utc, status FROM reservations"
        f" WHERE id IN ({placeholders})",
        ids,
    ).fetchall()
    found = {row["id"]: row for row in rows}
    missing = [rid for rid in req_by_id if rid not in found]
    if missing:
        # 404 优先于 409：先确认引用的预约都存在
        raise ReservationNotFoundError(missing[0])

    windows: dict[int, tuple[str, str]] = {}
    blocked: dict[int, str] = {}
    for rid, row in found.items():
        if row["status"] != "active":
            # 已取消预约不参与时段占用，也不能成为新分配的目标
            blocked[rid] = "reservation_not_active"
            continue
        windows[rid] = (row["start_utc"], row["end_utc"])

    room_rows = conn.execute(
        f"SELECT {_ROOM_COLUMNS} FROM meeting_rooms ORDER BY room_id ASC"
    ).fetchall()
    rooms = {row["room_id"]: _row_to_room(row) for row in room_rows}

    # 固定占用：本批不涉及的 active 预约所持有的现有分配。
    # 本批涉及预约的旧分配已被视为可替换，故不计入固定占用；
    # 持久化阶段会先删除这些旧行再写入新解（同一事务）。
    occ_rows = conn.execute(
        "SELECT a.reservation_id, a.room_id, r.start_utc, r.end_utc, r.status"
        " FROM meeting_room_assignments a"
        " JOIN reservations r ON r.id = a.reservation_id"
        f" WHERE a.reservation_id NOT IN ({placeholders})",
        ids,
    ).fetchall()
    fixed: dict[str, list[tuple[int, str, str]]] = {rid: [] for rid in rooms}
    for row in occ_rows:
        if row["status"] != "active":
            # 已取消预约不占用房间（其历史分配行保留用于查询）
            continue
        fixed.setdefault(row["room_id"], []).append(
            (row["reservation_id"], row["start_utc"], row["end_utc"]))

    # 候选集：仅按容量、设备做硬过滤（与固定占用的时段冲突在搜索中判定）
    candidates: dict[int, list[str]] = {}
    for rid, req in req_by_id.items():
        if rid in blocked:
            candidates[rid] = []
            continue
        wanted = set(req.required_equipment)
        eligible = [
            room.room_id for room in rooms.values()
            if room.capacity >= req.required_capacity
            and wanted.issubset(room.equipment)
        ]  # rooms.values() 已按 room_id 升序
        candidates[rid] = eligible
        if not eligible:
            blocked[rid] = "no_room_matching_capacity_or_equipment"

    return _SolveInputs(
        windows=windows,
        requests=req_by_id,
        rooms=rooms,
        fixed=fixed,
        candidates=candidates,
        blocked=blocked,
    )


def _fixed_blocked_rooms(
    inputs: _SolveInputs, rid: int
) -> frozenset[str]:
    """与 ``rid`` 时段重叠的固定占用所在房间集合（这些房间对 rid 不可用）。"""
    start, end = inputs.windows[rid]
    blocked: set[str] = set()
    for room_id, occupancy in inputs.fixed.items():
        for _oid, ostart, oend in occupancy:
            if _overlaps(start, end, ostart, oend):
                blocked.add(room_id)
                break
    return frozenset(blocked)


def _allowed_rooms(
    inputs: _SolveInputs,
    rid: int,
    assignment: dict[int, str],
    fixed_blocked: dict[int, frozenset[str]],
) -> list[str]:
    """当前部分分配下 ``rid`` 仍可选的房间。

    候选集（容量/设备）依次排除：与固定占用时段冲突的房间、本批中已放置且
    与 rid 时段重叠的预约所占房间。返回列表保持候选集的 room_id 升序。
    """
    start, end = inputs.windows[rid]
    blocked: set[str] = set(fixed_blocked.get(rid, ()))
    for other_id, other_room in assignment.items():  # 本批已放置预约
        if other_room not in blocked:
            ostart, oend = inputs.windows[other_id]
            if _overlaps(start, end, ostart, oend):
                blocked.add(other_room)
    return [room_id for room_id in inputs.candidates[rid]
            if room_id not in blocked]


def _max_matching_size(
    left_nodes: list[int], allowed: dict[int, list[str]]
) -> int:
    """二分图最大匹配基数（左侧预约 → 右侧房间），Kuhn 增广路算法。"""
    match_room_to_node: dict[str, int] = {}

    def augment(node: int, seen_rooms: set[str]) -> bool:
        for room_id in allowed[node]:
            if room_id in seen_rooms:
                continue
            seen_rooms.add(room_id)
            holder = match_room_to_node.get(room_id)
            if holder is None or augment(holder, seen_rooms):
                match_room_to_node[room_id] = node
                return True
        return False

    size = 0
    for node in left_nodes:  # left_nodes 已按 id 升序，匹配结果确定
        if augment(node, set()):
            size += 1
    return size


def _search(inputs: _SolveInputs) -> dict[int, str] | None:
    """确定性全局求解：返回最优分配 ``{reservation_id: room_id}``，无解返回 ``None``。

    目标：总空余座位 ``sum(room.capacity - required_capacity)`` 最小
    （每预约的 required_capacity 为常数，等价于最小化所用房间容量之和）；
    并列时取按预约 id 升序排列的 room_id 分配向量字典序最小解。

    存在硬阻断预约（非 active / 无满足容量设备的房间）时整批直接无解：
    阻断项不可分配，其他预约的可解性不影响整批失败。

    搜索按预约 id 升序**顺序**分支（不使用 MRV），以支持字典序前缀剪枝；
    可行性由扫描线 + 二分匹配的必要条件在每个节点快速判定：
    任一时刻同时进行的未分配预约两两重叠（构成团），必须能各自匹配到
    **不同的**可用房间（含排除前缀已占房间），匹配数不足即剪枝。
    """
    if inputs.blocked:
        return None
    order = sorted(inputs.requests)
    total_required = sum(inputs.requests[rid].required_capacity for rid in order)
    fixed_blocked = {rid: _fixed_blocked_rooms(inputs, rid) for rid in order}

    # 扫描线事件点：未分配预约集合只在某条预约的开始时刻发生变化
    instants = sorted({inputs.windows[rid][0] for rid in order})

    def matching_ok(
        remaining: list[int], assignment: dict[int, str]
    ) -> tuple[bool, dict[int, list[str]]]:
        """扫描线必要条件：每个时刻重叠的预约团须能匹配到不同可用房间。"""
        allowed: dict[int, list[str]] = {}
        for rid in remaining:
            opts = _allowed_rooms(inputs, rid, assignment, fixed_blocked)
            if not opts:
                return False, allowed
            allowed[rid] = opts
        for instant in instants:
            clique = [
                rid for rid in remaining
                if inputs.windows[rid][0] <= instant < inputs.windows[rid][1]
            ]
            if len(clique) <= 1:
                continue
            if _max_matching_size(clique, allowed) < len(clique):
                return False, allowed
        return True, allowed

    best: dict[int, str] | None = None
    best_cost: int | None = None

    def vector_prefix(assignment: dict[int, str], k: int) -> tuple[str, ...]:
        return tuple(assignment[order[i]] for i in range(k))

    def dfs(assignment: dict[int, str], used_capacity: int, k: int) -> None:
        nonlocal best, best_cost
        if k == len(order):
            cost = used_capacity - total_required
            vec = tuple(assignment[rid] for rid in order)
            if best is None or (cost, vec) < (
                    best_cost, tuple(best[rid] for rid in order)):
                best, best_cost = dict(assignment), cost
            return

        remaining = order[k:]
        ok, _allowed = matching_ok(remaining, assignment)
        if not ok:
            return  # 必要条件不满足：当前前缀必死

        # 成本下界：剩余预约各至少占其当前可用房间中的最小容量
        lower_bound = used_capacity + sum(
            min(inputs.rooms[r].capacity for r in _allowed[rid])
            for rid in remaining) - total_required
        if best_cost is not None:
            prefix = vector_prefix(assignment, k)
            incumbent_prefix = tuple(best[rid] for rid in order[:k])
            if lower_bound > best_cost:
                return  # 成本必劣
            if prefix > incumbent_prefix and lower_bound >= best_cost:
                # 字典序前缀已更大，且成本不可能严格更优 → 必劣
                return

        rid = order[k]
        # 分支顺序：容量小（空余少）优先；同容量 room_id 字典序优先。
        # 首片完整叶子因此即一个高质量初始可行解，后续靠成本/字典序剪枝。
        for room_id in sorted(_allowed[rid],
                              key=lambda x: (inputs.rooms[x].capacity, x)):
            assignment[rid] = room_id
            dfs(assignment, used_capacity + inputs.rooms[room_id].capacity, k + 1)
            del assignment[rid]

    dfs({}, 0, 0)
    return best


def _diagnose_unassignable(
    inputs: _SolveInputs,
) -> list[dict[str, object]]:
    """无解时给出确定性的“无法分配”预约清单。

    - 硬阻断（预约非 active、或没有任何房间满足容量/设备）必列；
    - 其余按 reservation_id 升序做贪心放置（固定占用与已放置预约为约束，
      每次取空余最少的可行房间），放置失败的预约视为受争抢所阻，一并列出。

    仅在穷举判定无解后调用，因此贪心失败不会误伤真正可行的批次。
    """
    items: list[dict[str, object]] = []

    def _record(rid: int, reason: str) -> None:
        req = inputs.requests[rid]
        items.append({
            "reservation_id": rid,
            "required_capacity": req.required_capacity,
            "required_equipment": list(req.required_equipment),
            "reason": reason,
        })

    for rid in sorted(inputs.blocked):
        _record(rid, inputs.blocked[rid])

    # 贪心仅处理硬阻断之外的预约；按 reservation_id 升序
    active_ids = sorted(r for r in inputs.requests if r not in inputs.blocked)
    fixed_blocked = {rid: _fixed_blocked_rooms(inputs, rid) for rid in active_ids}
    greedy: dict[int, str] = {}
    for rid in active_ids:
        opts = _allowed_rooms(inputs, rid, greedy, fixed_blocked)
        if not opts:
            _record(rid, "no_available_room")
            continue
        room_id = min(opts, key=lambda x: (inputs.rooms[x].capacity, x))
        greedy[rid] = room_id

    items.sort(key=lambda item: int(item["reservation_id"]))
    return items


def batch_assign_rooms(
    requests: Iterable[tuple[int, int, Iterable[str]]],
    db_path: str | None = None,
) -> dict[str, object]:
    """批量分配会议室，整批可解才在单个 ``BEGIN IMMEDIATE`` 事务内持久化。

    ``requests`` 为 ``(reservation_id, required_capacity, required_equipment)``
    序列（API 层已保证非空、reservation_id 不重复）。返回::

        {"assignments": {reservation_id: room_id},
         "total_spare_seats": int}

    无解 → :class:`BatchAssignmentInfeasibleError`（409，含无法分配清单），
    已有分配保持原样；预约不存在 → :class:`ReservationNotFoundError`（404）。
    """
    batch = [
        BatchRequest(
            reservation_id=rid,
            required_capacity=cap,
            required_equipment=normalize_equipment(eq),
        )
        for rid, cap, eq in requests
    ]
    if not batch:
        raise ValueError("batch assignment request must contain at least one item")

    path = db.ensure_initialized(db_path)
    conn = db.connect(path)
    try:
        # BEGIN IMMEDIATE：读取现状（房间、预约、既有分配）+ 求解 + 写入
        # 在同一事务内完成，并发批次被串行化，后到者看到先到者已提交的占用。
        conn.execute("BEGIN IMMEDIATE")
        try:
            now_s = _canon(datetime.now(timezone.utc))
            inputs = _load_solve_inputs(conn, batch)
            solution = _search(inputs)
            if solution is None:
                # 整批无解：先在事务内构造诊断，再回滚，不改动任何已有分配
                unassignable = _diagnose_unassignable(inputs)
                raise BatchAssignmentInfeasibleError(unassignable)

            # 可解：原子替换本批预约的分配行（先删后插，同事务）。
            # 本批之外的固定占用行完全不触碰。
            ids = tuple(req.reservation_id for req in batch)
            placeholders = ",".join("?" for _ in ids)
            conn.execute(
                f"DELETE FROM meeting_room_assignments"
                f" WHERE reservation_id IN ({placeholders})",
                ids,
            )
            conn.executemany(
                "INSERT INTO meeting_room_assignments"
                " (reservation_id, room_id, created_at) VALUES (?, ?, ?)",
                [(rid, solution[rid], now_s) for rid in sorted(solution)],
            )
            total_spare = sum(
                inputs.rooms[room_id].capacity
                - inputs.requests[rid].required_capacity
                for rid, room_id in solution.items())
            conn.execute("COMMIT")
            return {
                "assignments": {rid: solution[rid] for rid in sorted(solution)},
                "total_spare_seats": total_spare,
                # 窗口与房间容量取自本事务读到的快照，避免提交后再查产生竞态
                "windows": dict(inputs.windows),
                "requests": dict(inputs.requests),
                "rooms": dict(inputs.rooms),
            }
        except BaseException:
            # 无活动事务时 rollback 为 no-op；任何失败都保证整体回滚
            conn.rollback()
            raise
    finally:
        conn.close()


# ---------------------------------------------------------------- 分配查询


def _row_to_assignment(row: sqlite3.Row) -> AssignmentView:
    return AssignmentView(
        id=row["id"],
        reservation_id=row["reservation_id"],
        room_id=row["room_id"],
        created_at=_parse(row["created_at"]),
        reservation_request_id=row["request_id"],
        start_utc=_parse(row["res_start"]),
        end_utc=_parse(row["res_end"]),
        status=row["status"],
        room_capacity=row["room_capacity"],
        room_equipment=tuple(json.loads(row["room_equipment"])),
    )


def list_assignments(
    reservation_id: int | None = None,
    room_id: str | None = None,
    *,
    active_only: bool = False,
    db_path: str | None = None,
) -> list[AssignmentView]:
    """查询会议室分配，按 ``reservation_id`` 升序返回。

    - ``reservation_id`` 给定但预约不存在 → :class:`ReservationNotFoundError`；
    - ``room_id`` 给定但房间不存在 → :class:`RoomNotFoundError`；
    - ``active_only=True`` 仅返回 active 预约的分配（已取消预约的历史分配
      默认仍返回，``status`` 字段标明其状态）。
    """
    path = db.ensure_initialized(db_path)
    conn = db.connect(path)
    try:
        if reservation_id is not None:
            exists = conn.execute(
                "SELECT 1 FROM reservations WHERE id = ?", (reservation_id,))
            if exists.fetchone() is None:
                raise ReservationNotFoundError(reservation_id)
        if room_id is not None:
            exists = conn.execute(
                "SELECT 1 FROM meeting_rooms WHERE room_id = ?", (room_id,))
            if exists.fetchone() is None:
                raise RoomNotFoundError(room_id)

        sql = (
            "SELECT a.id, a.reservation_id, a.room_id, a.created_at,"
            " r.request_id, r.start_utc AS res_start, r.end_utc AS res_end,"
            " r.status, m.capacity AS room_capacity, m.equipment_key AS room_equipment"
            " FROM meeting_room_assignments a"
            " JOIN reservations r ON r.id = a.reservation_id"
            " JOIN meeting_rooms m ON m.room_id = a.room_id"
        )
        clauses: list[str] = []
        params: list[object] = []
        if reservation_id is not None:
            clauses.append("a.reservation_id = ?")
            params.append(reservation_id)
        if room_id is not None:
            clauses.append("a.room_id = ?")
            params.append(room_id)
        if active_only:
            clauses.append("r.status = 'active'")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY a.reservation_id ASC, a.id ASC"
        rows = conn.execute(sql, params).fetchall()
        return [_row_to_assignment(row) for row in rows]
    finally:
        conn.close()
