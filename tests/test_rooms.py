"""会议室批量分配（meeting rooms）测试。

覆盖：房间 upsert（201/200/字段回显/设备去重排序）、查询（单房间/列表/
404）、批量分配的容量与设备硬约束、半开区间同房互斥（相邻可同房）、
总空余最少 + (reservation_id, room_id) 字典序的确定性取解、
已有分配作为固定占用、本批旧分配被替换、无解 409 列出无法分配项且
不改动已有分配、404/422、分配查询过滤与 active_only、并发仅一个赢家、
旧版库无损建新表。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app import db as db_module
from app import rooms as room_core
from app.main import app


@pytest.fixture()
def db_path(tmp_path, monkeypatch) -> str:
    """每个测试用独立的 SQLite 文件。"""
    path = str(tmp_path / "reservations.db")
    monkeypatch.setenv("RESERVATIONS_DB_PATH", path)
    return path


@pytest.fixture()
def client(db_path) -> TestClient:
    return TestClient(app)


def _utc(h, mi=0, day=9):
    return f"2026-03-{day:02d}T{h:02d}:{mi:02d}:00Z"


def _reservation(client, request_id, start, end, participants=("alice",)):
    resp = client.post("/api/v1/reservations", json={
        "request_id": request_id,
        "start_utc": start,
        "end_utc": end,
        "participant_ids": list(participants),
    })
    assert resp.status_code == 201, resp.text
    return resp.json()["reservation_id"]


def _room(client, room_id, capacity, equipment=(), path_id=None):
    resp = client.put(f"/api/v1/meeting-rooms/{path_id or room_id}", json={
        "room_id": room_id,
        "capacity": capacity,
        "equipment_tags": list(equipment),
    })
    return resp


def _assign(client, items):
    return client.post("/api/v1/meeting-room-assignments/batch",
                       json={"reservations": items})


def _assignment_map(resp) -> dict[int, str]:
    assert resp.status_code == 200, resp.text
    return {a["reservation_id"]: a["room_id"] for a in resp.json()["assignments"]}


# ----------------------------------------------------------------- 房间管理


def test_upsert_room_create_and_update(client):
    resp = _room(client, "room-a", 10, ["whiteboard", "projector", "whiteboard"])
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["room_id"] == "room-a"
    assert data["capacity"] == 10
    # 去重 + 字典序
    assert data["equipment_tags"] == ["projector", "whiteboard"]
    created_at = data["created_at"]
    assert data["updated_at"] == created_at

    resp = _room(client, "room-a", 12, ["tv"])
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["capacity"] == 12
    assert data["equipment_tags"] == ["tv"]
    assert data["created_at"] == created_at
    assert data["updated_at"] >= created_at


def test_get_room_and_list_sorted(client):
    _room(client, "zeta", 8)
    _room(client, "alpha", 20, ["projector"])
    _room(client, "mid", 4)

    resp = client.get("/api/v1/meeting-rooms/alpha")
    assert resp.status_code == 200
    assert resp.json()["capacity"] == 20

    assert client.get("/api/v1/meeting-rooms/nope").status_code == 404

    resp = client.get("/api/v1/meeting-rooms")
    assert resp.status_code == 200
    assert [r["room_id"] for r in resp.json()["rooms"]] == ["alpha", "mid", "zeta"]


def test_room_validation_errors(client):
    # 容量 < 1
    resp = _room(client, "x", 0)
    assert resp.status_code == 422
    # 空设备标签
    resp = client.put("/api/v1/meeting-rooms/x", json={
        "room_id": "x", "capacity": 4, "equipment_tags": ["  "]})
    assert resp.status_code == 422
    # 请求体 room_id 为空（min_length）
    resp = client.put("/api/v1/meeting-rooms/x", json={
        "room_id": "", "capacity": 4, "equipment_tags": []})
    assert resp.status_code == 422
    # 路径与请求体 room_id 不一致
    resp = _room(client, "body-id", 4, path_id="path-id")
    assert resp.status_code == 422


def test_upsert_room_does_not_touch_assignments(client):
    rid = _reservation(client, "r1", _utc(13), _utc(14))
    _room(client, "r", 10)
    assert _assign(client, [{
        "reservation_id": rid, "required_capacity": 4,
        "required_equipment": []}]).status_code == 200
    # 更新房间容量不改变既有分配行
    _room(client, "r", 8)
    resp = client.get("/api/v1/meeting-room-assignments",
                      params={"reservation_id": rid})
    assert resp.status_code == 200
    assert resp.json()["assignments"][0]["room_id"] == "r"


# ----------------------------------------------------------------- 批量分配


def test_batch_assign_capacity_and_equipment_gating(client):
    rid = _reservation(client, "r1", _utc(13), _utc(14))
    _room(client, "small", 4)
    _room(client, "big", 20, ["projector"])

    # 容量不足：small=4 < 需要 6，big 满足容量但缺 projector
    resp = _assign(client, [{
        "reservation_id": rid, "required_capacity": 6,
        "required_equipment": ["projector", "whiteboard"]}])
    assert resp.status_code == 409, resp.text
    data = resp.json()
    assert data["unassignable"] == [{
        "reservation_id": rid,
        "required_capacity": 6,
        "required_equipment": ["projector", "whiteboard"],
        "reason": "no_room_matching_capacity_or_equipment",
    }]
    # 无解时无任何分配落库
    assert client.get("/api/v1/meeting-room-assignments").json()["assignments"] == []

    # 补齐设备后可解：唯一可行房间 big，空余 20-6=14
    _room(client, "big", 20, ["projector", "whiteboard"])
    resp = _assign(client, [{
        "reservation_id": rid, "required_capacity": 6,
        "required_equipment": ["projector", "whiteboard"]}])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total_spare_seats"] == 14
    item = body["assignments"][0]
    assert item["room_id"] == "big"
    assert item["spare_seats"] == 14
    assert item["start_utc"] == _utc(13)
    assert item["end_utc"] == _utc(14)


def test_batch_assign_overlapping_split_adjacent_share(client):
    r1 = _reservation(client, "r1", _utc(13), _utc(14), ["a"])
    r2 = _reservation(client, "r2", _utc(13, 30), _utc(14, 30), ["b"])
    r3 = _reservation(client, "r3", _utc(14, 30), _utc(15), ["c"])  # 与 r2 首尾相接
    _room(client, "ra", 10)
    _room(client, "rb", 10)

    # r1 与 r2 重叠：必须分房；字典序取解 r1->ra, r2->rb；
    # r3 与 r2 相邻（半开区间不重叠）可同房，但与 r1 也不相交 → 两者皆可，
    # 空余并列时按分配向量字典序最小：r3 选 ra
    body = _assign(client, [
        {"reservation_id": r1, "required_capacity": 2, "required_equipment": []},
        {"reservation_id": r2, "required_capacity": 2, "required_equipment": []},
        {"reservation_id": r3, "required_capacity": 2, "required_equipment": []},
    ])
    assert _assignment_map(body) == {r1: "ra", r2: "rb", r3: "ra"}
    assert body.json()["total_spare_seats"] == 8 + 8 + 8


def test_batch_assign_minimizes_total_spare(client):
    # 三条互不重叠的预约：应紧凑复用小房间，而非分散到大房间
    r1 = _reservation(client, "r1", _utc(9), _utc(10), ["a"])
    r2 = _reservation(client, "r2", _utc(11), _utc(12), ["b"])
    r3 = _reservation(client, "r3", _utc(13), _utc(14), ["c"])
    _room(client, "big", 100)
    _room(client, "small", 6)

    body = _assign(client, [
        {"reservation_id": r1, "required_capacity": 4, "required_equipment": []},
        {"reservation_id": r2, "required_capacity": 4, "required_equipment": []},
        {"reservation_id": r3, "required_capacity": 4, "required_equipment": []},
    ])
    # 全部落 small（不重叠可同房）：总空余 3*(6-4)=6，远小于用 big
    assert _assignment_map(body) == {r1: "small", r2: "small", r3: "small"}
    assert body.json()["total_spare_seats"] == 6


def test_batch_assign_tiebreak_lexicographic(client):
    # 空余并列（两房容量相同、需求相同）：重叠的两条按预约 id 升序取最小 room_id
    r1 = _reservation(client, "r1", _utc(13), _utc(14), ["a"])
    r2 = _reservation(client, "r2", _utc(13, 15), _utc(14, 15), ["b"])
    _room(client, "room-b", 10)
    _room(client, "room-a", 10)

    body = _assign(client, [
        {"reservation_id": r2, "required_capacity": 2, "required_equipment": []},
        # 请求顺序故意打乱：结果与请求顺序无关
        {"reservation_id": r1, "required_capacity": 2, "required_equipment": []},
    ])
    mapping = _assignment_map(body)
    assert mapping == {r1: "room-a", r2: "room-b"}
    # 响应列表按 reservation_id 升序
    ids = [a["reservation_id"] for a in body.json()["assignments"]]
    assert ids == sorted(ids)


def test_batch_assign_respects_existing_fixed_assignment(client):
    # r0 不在批内、已分到 small，与本批 r1 重叠 → small 对 r1 不可用
    r0 = _reservation(client, "r0", _utc(13), _utc(14), ["x"])
    r1 = _reservation(client, "r1", _utc(13, 30), _utc(14, 30), ["y"])
    _room(client, "small", 6)
    _room(client, "big", 100)
    assert _assign(client, [{
        "reservation_id": r0, "required_capacity": 2,
        "required_equipment": []}]).status_code == 200

    body = _assign(client, [{
        "reservation_id": r1, "required_capacity": 2, "required_equipment": []}])
    # small 被固定占用；尽管 small 空余更少，r1 只能去 big
    assert _assignment_map(body) == {r1: "big"}
    # 固定分配行原封不动
    rows = client.get("/api/v1/meeting-room-assignments").json()["assignments"]
    assert {(r["reservation_id"], r["room_id"]) for r in rows} == {
        (r0, "small"), (r1, "big")}


def test_batch_assign_replaces_own_prior_assignment(client):
    # 本批涉及预约的旧分配视为可替换：r1 旧占 big，重新批量时应释放并紧凑到 small
    r1 = _reservation(client, "r1", _utc(13), _utc(14), ["a"])
    r2 = _reservation(client, "r2", _utc(13), _utc(14), ["b"])
    _room(client, "a-small", 6)
    _room(client, "z-big", 100)
    assert _assignment_map(_assign(client, [
        {"reservation_id": r1, "required_capacity": 2, "required_equipment": []},
        {"reservation_id": r2, "required_capacity": 2, "required_equipment": []},
    ])) == {r1: "a-small", r2: "z-big"}

    # 本批涉及预约的旧分配视为可替换：重新批量时两条重叠预约的旧占用
    # 不阻止新解；确定性解与首次相同，每预约仍只有一行分配
    body = _assign(client, [
        {"reservation_id": r2, "required_capacity": 2, "required_equipment": []},
        {"reservation_id": r1, "required_capacity": 2, "required_equipment": []},
    ])
    assert _assignment_map(body) == {r1: "a-small", r2: "z-big"}
    rows = client.get("/api/v1/meeting-room-assignments").json()["assignments"]
    assert sorted(r["reservation_id"] for r in rows) == [r1, r2]


def test_batch_assign_infeasible_leaves_existing_untouched(client):
    r1 = _reservation(client, "r1", _utc(13), _utc(14), ["a"])
    r2 = _reservation(client, "r2", _utc(13, 15), _utc(14, 15), ["b"])
    r3 = _reservation(client, "r3", _utc(13, 30), _utc(14, 30), ["c"])
    _room(client, "only", 10)
    # 先成功分配 r1
    assert _assignment_map(_assign(client, [
        {"reservation_id": r1, "required_capacity": 2, "required_equipment": []}
    ])) == {r1: "only"}

    # r2 与 r3 相互重叠且都与固定占用 r1 重叠：单间放不下三条
    resp = _assign(client, [
        {"reservation_id": r2, "required_capacity": 2, "required_equipment": []},
        {"reservation_id": r3, "required_capacity": 2, "required_equipment": []},
    ])
    assert resp.status_code == 409, resp.text
    unassignable = resp.json()["unassignable"]
    ids = {u["reservation_id"] for u in unassignable}
    assert ids == {r2, r3}
    assert all(u["reason"] == "no_available_room" for u in unassignable)
    # 列表按 reservation_id 升序
    assert [u["reservation_id"] for u in unassignable] == sorted(ids)

    # 已有分配（r1）原样保留，失败批不写入任何行
    rows = client.get("/api/v1/meeting-room-assignments").json()["assignments"]
    assert [(r["reservation_id"], r["room_id"]) for r in rows] == [(r1, "only")]


def test_batch_assign_404_for_missing_reservation(client):
    _room(client, "r", 10)
    resp = _assign(client, [{
        "reservation_id": 999, "required_capacity": 2, "required_equipment": []}])
    assert resp.status_code == 404
    assert "999" in resp.json()["detail"]


def test_batch_assign_canceled_reservation_is_unassignable(client):
    rid = _reservation(client, "r1", _utc(13), _utc(14))
    _room(client, "r", 10)
    client.delete(f"/api/v1/reservations/{rid}?expected_version=1")
    resp = _assign(client, [{
        "reservation_id": rid, "required_capacity": 2, "required_equipment": []}])
    assert resp.status_code == 409, resp.text
    item = resp.json()["unassignable"][0]
    assert item["reservation_id"] == rid
    assert item["reason"] == "reservation_not_active"


def test_batch_assign_validation_errors(client):
    rid = _reservation(client, "r1", _utc(13), _utc(14))
    _room(client, "r", 10)

    # 空批次
    assert _assign(client, []).status_code == 422
    # reservation_id 重复
    resp = _assign(client, [
        {"reservation_id": rid, "required_capacity": 2, "required_equipment": []},
        {"reservation_id": rid, "required_capacity": 2, "required_equipment": []},
    ])
    assert resp.status_code == 422
    # required_capacity < 1
    resp = _assign(client, [{
        "reservation_id": rid, "required_capacity": 0,
        "required_equipment": []}])
    assert resp.status_code == 422
    # 空设备标签
    resp = _assign(client, [{
        "reservation_id": rid, "required_capacity": 2,
        "required_equipment": [""]}])
    assert resp.status_code == 422


# ----------------------------------------------------------------- 分配查询


def test_list_assignments_filters_and_active_only(client):
    r1 = _reservation(client, "r1", _utc(13), _utc(14), ["a"])
    r2 = _reservation(client, "r2", _utc(15), _utc(16), ["b"])
    _room(client, "ra", 6, ["tv"])
    _room(client, "rb", 8)
    _assign(client, [
        {"reservation_id": r1, "required_capacity": 4, "required_equipment": []},
        {"reservation_id": r2, "required_capacity": 3, "required_equipment": []},
    ])
    # r2 落 ra（空余 3 < rb 空余 5）；r1 与 r2 不重叠也落 ra
    rows = client.get("/api/v1/meeting-room-assignments").json()["assignments"]
    assert [r["reservation_id"] for r in rows] == [r1, r2]
    assert rows[0]["room_equipment_tags"] == ["tv"]
    assert rows[0]["room_capacity"] == 6
    assert rows[0]["status"] == "active"

    # 按预约过滤
    rows = client.get(
        "/api/v1/meeting-room-assignments",
        params={"reservation_id": r2}).json()["assignments"]
    assert [r["reservation_id"] for r in rows] == [r2]

    # 按房间过滤
    rows = client.get(
        "/api/v1/meeting-room-assignments",
        params={"room_id": "ra"}).json()["assignments"]
    assert {r["reservation_id"] for r in rows} == {r1, r2}
    assert client.get(
        "/api/v1/meeting-room-assignments",
        params={"room_id": "nope"}).status_code == 404
    assert client.get(
        "/api/v1/meeting-room-assignments",
        params={"reservation_id": 999}).status_code == 404

    # 取消 r2：默认仍返回历史分配（status=canceled），active_only 则隐藏
    client.delete(f"/api/v1/reservations/{r2}?expected_version=1")
    rows = client.get("/api/v1/meeting-room-assignments").json()["assignments"]
    assert {r["reservation_id"] for r in rows} == {r1, r2}
    canceled = [r for r in rows if r["reservation_id"] == r2][0]
    assert canceled["status"] == "canceled"
    rows = client.get(
        "/api/v1/meeting-room-assignments",
        params={"active_only": "true"}).json()["assignments"]
    assert [r["reservation_id"] for r in rows] == [r1]


def test_canceled_reservation_does_not_block_room(client):
    # 已取消预约的历史分配不占房：新的重叠预约可以分到同一房间
    r1 = _reservation(client, "r1", _utc(13), _utc(14), ["a"])
    r2 = _reservation(client, "r2", _utc(13, 30), _utc(14, 30), ["b"])
    _room(client, "only", 6)
    _assign(client, [{
        "reservation_id": r1, "required_capacity": 2, "required_equipment": []}])
    client.delete(f"/api/v1/reservations/{r1}?expected_version=1")
    body = _assign(client, [{
        "reservation_id": r2, "required_capacity": 2, "required_equipment": []}])
    assert _assignment_map(body) == {r2: "only"}


# ----------------------------------------------------------------- 并发/迁移


def test_concurrent_batches_only_one_wins_for_one_room(client, db_path):
    """两批都想在重叠时段独占唯一房间：一个成功，另一个 409，无脏写。"""
    # 直接走核心层 + 线程池，确保跨连接的 BEGIN IMMEDIATE 串行化
    from app import reservations as reservation_core
    from datetime import datetime, timezone

    def _u(h, mi=0):
        return datetime(2026, 3, 9, h, mi, tzinfo=timezone.utc)

    r1, _ = reservation_core.create_reservation(
        "r1", _u(13), _u(14), ["a"], db_path=db_path)
    r2, _ = reservation_core.create_reservation(
        "r2", _u(13, 30), _u(14, 30), ["b"], db_path=db_path)
    room_core.upsert_room("only", 6, [], db_path=db_path)

    barrier_results: list[object] = []

    def worker(rid: int):
        try:
            return ("ok", room_core.batch_assign_rooms(
                [(rid, 2, [])], db_path=db_path)["assignments"])
        except room_core.BatchAssignmentInfeasibleError as exc:
            return ("infeasible", [u["reservation_id"] for u in exc.unassignable])

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(worker, [r1.id, r2.id]))

    statuses = [o[0] for o in outcomes]
    assert statuses.count("ok") == 1
    assert statuses.count("infeasible") == 1
    # 唯一成功的那条独占 only
    rows = room_core.list_assignments(db_path=db_path)
    assert len(rows) == 1
    assert rows[0].room_id == "only"


def test_init_creates_room_tables_losslessly(tmp_path, monkeypatch):
    """旧版库（无会议室表）初始化后新增两表，既有数据不动。"""
    import sqlite3

    path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(path)
    try:
        conn.executescript("""\
CREATE TABLE reservations (
    id INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT NOT NULL UNIQUE,
    start_utc TEXT NOT NULL, end_utc TEXT NOT NULL,
    participants_key TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE reservation_participants (
    reservation_id INTEGER NOT NULL REFERENCES reservations(id) ON DELETE CASCADE,
    participant_id TEXT NOT NULL, start_utc TEXT NOT NULL, end_utc TEXT NOT NULL,
    PRIMARY KEY (reservation_id, participant_id));
""")
        conn.execute(
            "INSERT INTO reservations"
            " (request_id, start_utc, end_utc, participants_key, created_at)"
            " VALUES ('old', '2026-03-09T13:00:00.000000Z',"
            " '2026-03-09T14:00:00.000000Z', '[\"alice\"]',"
            " '2026-03-01T00:00:00.000000Z')")
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setenv("RESERVATIONS_DB_PATH", path)
    db_module.init_db(path)
    db_module.init_db(path)  # 幂等

    conn = sqlite3.connect(path)
    try:
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"meeting_rooms", "meeting_room_assignments"} <= names
        assert conn.execute(
            "SELECT request_id, participants_key FROM reservations"
        ).fetchone() == ("old", "[\"alice\"]")
    finally:
        conn.close()

    # 新表立即可用
    room, created = room_core.upsert_room("r", 10, ["tv"], db_path=path)
    assert created is True
    assert room.room_id == "r"
