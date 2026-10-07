from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MatpowerCase:
    name: str
    base_mva: float
    bus: np.ndarray
    gen: np.ndarray
    branch: np.ndarray
    gencost: np.ndarray

    @property
    def n_bus(self) -> int:
        return int(self.bus.shape[0])

    @property
    def n_gen(self) -> int:
        return int(self.gen.shape[0])

    @property
    def n_branch(self) -> int:
        return int(self.branch.shape[0])


def _strip_comments(text: str) -> str:
    return "\n".join(line.split("%", 1)[0] for line in text.splitlines())


def _matrix(text: str, field: str) -> np.ndarray:
    match = re.search(rf"mpc\.{re.escape(field)}\s*=\s*\[(.*?)\];", text, re.S)
    if not match:
        raise ValueError(f"MATPOWER matrix mpc.{field} was not found")
    rows: list[list[float]] = []
    for raw_row in match.group(1).split(";"):
        raw_row = raw_row.strip()
        if not raw_row:
            continue
        values = [float(value) for value in raw_row.replace("\n", " ").split()]
        rows.append(values)
    if not rows:
        raise ValueError(f"MATPOWER matrix mpc.{field} is empty")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"MATPOWER matrix mpc.{field} has ragged rows")
    return np.asarray(rows, dtype=np.float64)


def load_matpower_case(path: str | Path) -> MatpowerCase:
    source = Path(path)
    text = _strip_comments(source.read_text(encoding="utf-8"))
    base_match = re.search(r"mpc\.baseMVA\s*=\s*([^;]+);", text)
    if not base_match:
        raise ValueError("mpc.baseMVA was not found")
    return MatpowerCase(
        name=source.stem,
        base_mva=float(base_match.group(1).strip()),
        bus=_matrix(text, "bus"),
        gen=_matrix(text, "gen"),
        branch=_matrix(text, "branch"),
        gencost=_matrix(text, "gencost"),
    )


def canonical_bus_order(case: MatpowerCase) -> np.ndarray:
    """Topology-aware deterministic BFS order, starting from the slack bus."""
    ids = case.bus[:, 0].astype(np.int64)
    id_to_pos = {int(bus_id): pos for pos, bus_id in enumerate(ids)}
    adjacency: list[list[int]] = [[] for _ in range(case.n_bus)]
    for src_id, dst_id in case.branch[:, :2].astype(np.int64):
        if int(src_id) not in id_to_pos or int(dst_id) not in id_to_pos:
            continue
        src, dst = id_to_pos[int(src_id)], id_to_pos[int(dst_id)]
        adjacency[src].append(dst)
        adjacency[dst].append(src)
    slack_candidates = np.flatnonzero(case.bus[:, 1].astype(np.int64) == 3)
    starts = list(slack_candidates) + list(np.argsort(ids))
    seen: set[int] = set()
    order: list[int] = []
    for start in starts:
        start = int(start)
        if start in seen:
            continue
        queue = [start]
        seen.add(start)
        while queue:
            node = queue.pop(0)
            order.append(node)
            for neighbor in sorted(adjacency[node], key=lambda x: int(ids[x])):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
    return np.asarray(order, dtype=np.int64)


def case_to_static_arrays(case: MatpowerCase) -> dict[str, np.ndarray]:

    order = canonical_bus_order(case)
    old_to_new = np.empty(case.n_bus, dtype=np.int64)
    old_to_new[order] = np.arange(case.n_bus)
    bus = case.bus[order]
    ids = case.bus[:, 0].astype(np.int64)
    id_to_old = {int(bus_id): pos for pos, bus_id in enumerate(ids)}

    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_attr: list[list[float]] = []
    degree = np.zeros(case.n_bus, dtype=np.float32)
    for row in case.branch:
        src_old = id_to_old[int(row[0])]
        dst_old = id_to_old[int(row[1])]
        src, dst = int(old_to_new[src_old]), int(old_to_new[dst_old])
        rate_a = row[5] / case.base_mva if row[5] > 0 else 10.0
        attr = [row[2], row[3], row[4], rate_a, row[8] if row[8] else 1.0, row[9] / 180.0]
        edge_src.extend([src, dst])
        edge_dst.extend([dst, src])
        edge_attr.extend([attr, attr])
        degree[src] += 1.0
        degree[dst] += 1.0

    bus_type = bus[:, 1].astype(np.int64)
    type_one_hot = np.stack([(bus_type == value).astype(np.float32) for value in (1, 2, 3, 4)], axis=1)
    static_bus = np.column_stack(
        [
            bus[:, 9] / 500.0,
            bus[:, 11],
            bus[:, 12],
            np.log1p(degree) / 5.0,
            type_one_hot,
        ]
    ).astype(np.float32)

    gen_bus_old = np.asarray([id_to_old[int(bus_id)] for bus_id in case.gen[:, 0]], dtype=np.int64)
    gen_bus = old_to_new[gen_bus_old]

    cost_features = []
    for cost in case.gencost:
        coefficients = cost[4 : 4 + int(cost[3])]
        padded = np.pad(coefficients, (max(0, 3 - len(coefficients)), 0))[-3:]
        cost_features.append([padded[0], padded[1] / 200.0, padded[2] / 1e5])
    gen_limits = np.column_stack(
        [
            case.gen[:, 9] / case.base_mva,
            case.gen[:, 8] / case.base_mva,
            case.gen[:, 4] / case.base_mva,
            case.gen[:, 3] / case.base_mva,
            case.gen[:, 5],
            case.gen[:, 1] / case.base_mva,
            case.gen[:, 2] / case.base_mva,
            np.asarray(cost_features, dtype=np.float32),
        ]
    ).astype(np.float32)

    gen_limits = np.nan_to_num(gen_limits, nan=0.0, posinf=10.0, neginf=-10.0)
    return {
        "bus_order": order,
        "static_bus": static_bus,
        "edge_index": np.asarray([edge_src, edge_dst], dtype=np.int64),
        "edge_attr": np.asarray(edge_attr, dtype=np.float32),
        "gen_bus": gen_bus.astype(np.int64),
        "gen_limits": gen_limits,
    }
