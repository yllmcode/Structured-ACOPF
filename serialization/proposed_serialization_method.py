"""
The fixed MATPOWER case supplies topology, equipment parameters, and limits.
Each scenario contains canonical, semantically tagged per-unit loads and the
corresponding IPOPT target.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .matpower import MatpowerCase, canonical_bus_order


@dataclass(frozen=True)
class SerializedSample:
    instruction: str
    input_text: str
    output_text: str
    case: str
    source_index: int


def _number(value: float) -> str:
    value = float(value)
    if not np.isfinite(value):
        value = 10.0 if value > 0 else -10.0
    rendered = f"{value:.4f}".rstrip("0").rstrip(".")
    return "0" if rendered in {"-0", ""} else rendered


def _orders(case: MatpowerCase) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bus_order = canonical_bus_order(case)
    bus_rank = np.empty(case.n_bus, dtype=np.int64)
    bus_rank[bus_order] = np.arange(case.n_bus)
    id_to_pos = {int(bus_id): pos for pos, bus_id in enumerate(case.bus[:, 0])}
    gen_order = np.asarray(
        sorted(
            range(case.n_gen),
            key=lambda index: (
                int(bus_rank[id_to_pos[int(case.gen[index, 0])]]),
                index,
            ),
        ),
        dtype=np.int64,
    )
    branch_order = np.asarray(
        sorted(
            range(case.n_branch),
            key=lambda index: (
                min(
                    int(bus_rank[id_to_pos[int(case.branch[index, 0])]]),
                    int(bus_rank[id_to_pos[int(case.branch[index, 1])]]),
                ),
                max(
                    int(bus_rank[id_to_pos[int(case.branch[index, 0])]]),
                    int(bus_rank[id_to_pos[int(case.branch[index, 1])]]),
                ),
                index,
            ),
        ),
        dtype=np.int64,
    )
    return bus_order, gen_order, branch_order


def _linear_cost(case: MatpowerCase, index: int) -> float:
    row = case.gencost[index]
    coefficients = row[4 : 4 + int(row[3])]
    padded = np.pad(coefficients, (max(0, 3 - len(coefficients)), 0))[-3:]
    return float(padded[1])


def _branch_terminal_power(
    case: MatpowerCase,
    vm_pu: np.ndarray,
    va_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:


    bus_position = {
        int(bus_id): index for index, bus_id in enumerate(case.bus[:, 0])
    }
    voltage = np.asarray(vm_pu, dtype=np.float64) * np.exp(
        1j * np.asarray(va_rad, dtype=np.float64)
    )
    pf = np.zeros(case.n_branch, dtype=np.float64)
    qf = np.zeros(case.n_branch, dtype=np.float64)
    pt = np.zeros(case.n_branch, dtype=np.float64)
    qt = np.zeros(case.n_branch, dtype=np.float64)
    for index, branch in enumerate(case.branch):
        status = float(branch[10]) if branch.shape[0] > 10 else 1.0
        if status == 0.0:
            continue
        resistance, reactance, charging = map(float, branch[2:5])
        series = status / complex(resistance, reactance)
        line_charging = status * charging
        ratio = float(branch[8]) if branch.shape[0] > 8 and branch[8] else 1.0
        shift = float(branch[9]) if branch.shape[0] > 9 else 0.0
        tap = ratio * np.exp(1j * np.deg2rad(shift))
        ytt = series + 1j * line_charging / 2.0
        yff = ytt / (tap * np.conj(tap))
        yft = -series / np.conj(tap)
        ytf = -series / tap
        from_bus = bus_position[int(branch[0])]
        to_bus = bus_position[int(branch[1])]
        vf, vt = voltage[from_bus], voltage[to_bus]
        sf = vf * np.conj(yff * vf + yft * vt)
        st = vt * np.conj(ytf * vf + ytt * vt)
        pf[index], qf[index] = float(sf.real), float(sf.imag)
        pt[index], qt[index] = float(st.real), float(st.imag)
    return pf, qf, pt, qt


def serialize_sample(
    case: MatpowerCase,
    arrays: dict[str, np.ndarray],
    row: int,
    *,
    source_index: int | None = None,
) -> SerializedSample:


    if case.name not in {"case9", "case30", "case118"}:
        raise ValueError("This release contains only IEEE case9, case30, and case118")
    if source_index is None:
        source_index = int(
            arrays.get("source_index", np.arange(len(arrays["pd_pu"])))[row]
        )
    bus_order, gen_order, branch_order = _orders(case)
    bus_ids = case.bus[:, 0].astype(np.int64)

    def values(fields: list[object]) -> str:
        return "|".join(
            str(value) if isinstance(value, str) else _number(float(value))
            for value in fields
        )

    input_lines = [
        f"<GRID:case={case.name}|baseMVA={_number(case.base_mva)}|static={case.name}>",
        (
            f"<GLOBAL:PdSum={_number(np.sum(arrays['pd_pu'][row]))}"
            f"|QdSum={_number(np.sum(arrays['qd_pu'][row]))}>"
        ),
        "<BUS:id|type|Pd|Qd>",
    ]
    for index in bus_order:
        input_lines.append(
            values(
                [
                    str(int(bus_ids[index])),
                    str(int(case.bus[index, 1])),
                    arrays["pd_pu"][row, index],
                    arrays["qd_pu"][row, index],
                ]
            )
        )
    input_lines.append("<GEN:id|bus|Pmax|Pmin|Qmax|Qmin|c1>")
    for index in gen_order:
        gen = case.gen[index]
        input_lines.append(
            values(
                [
                    str(int(index + 1)),
                    str(int(gen[0])),
                    gen[8] / case.base_mva,
                    gen[9] / case.base_mva,
                    gen[3] / case.base_mva,
                    gen[4] / case.base_mva,
                    _linear_cost(case, int(index)),
                ]
            )
        )
    input_lines.append("<BRANCH>")
    for index in branch_order:
        input_lines.append(
            values(
                [
                    str(int(case.branch[index, 0])),
                    str(int(case.branch[index, 1])),
                ]
            )
        )

    output_lines = [
        f"<OBJECTIVE:cost>{_number(arrays['objective'][row])}",
        "<GEN_STATE:id|Pg|Qg>",
    ]
    for index in gen_order:
        output_lines.append(
            values(
                [
                    f"G{index + 1}",
                    arrays["pg_pu"][row, index],
                    arrays["qg_pu"][row, index],
                ]
            )
        )
    output_lines.append("<BUS_STATE:id|Vm|Va>")
    for index in bus_order:
        output_lines.append(
            values(
                [
                    str(int(bus_ids[index])),
                    arrays["vm_pu"][row, index],
                    arrays["va_rad"][row, index],
                ]
            )
        )
    pf, qf, pt, qt = _branch_terminal_power(
        case,
        arrays["vm_pu"][row],
        arrays["va_rad"][row],
    )
    output_lines.append("<BRANCH_STATE:id|Pf|Qf|Pt|Qt>")
    for index in branch_order:
        output_lines.append(
            values(
                [
                    f"L{index + 1}",
                    pf[index],
                    qf[index],
                    pt[index],
                    qt[index],
                ]
            )
        )

    return SerializedSample(
        instruction=(
            "Predict the complete AC optimal power flow state. Return only the "
            "target serialization and preserve every component identifier."
        ),
        input_text="\n".join(input_lines),
        output_text="\n".join(output_lines),
        case=case.name,
        source_index=int(source_index),
    )


def parse_target(text: str, case: MatpowerCase) -> dict[str, np.ndarray | float]:


    lines = [line.strip() for line in text.splitlines() if line.strip()]
    expected = 4 + case.n_gen + case.n_bus + case.n_branch
    if len(lines) != expected:
        raise ValueError(f"expected {expected} target rows, received {len(lines)}")
    objective_match = re.fullmatch(r"<OBJECTIVE:cost>(.+)", lines[0])
    if not objective_match or lines[1] != "<GEN_STATE:id|Pg|Qg>":
        raise ValueError("invalid objective or generator-state header")
    bus_header = 2 + case.n_gen
    if lines[bus_header] != "<BUS_STATE:id|Vm|Va>":
        raise ValueError("invalid bus-state header")
    branch_header = bus_header + 1 + case.n_bus
    if lines[branch_header] != "<BRANCH_STATE:id|Pf|Qf|Pt|Qt>":
        raise ValueError("invalid branch-state header")

    pg = np.full(case.n_gen, np.nan)
    qg = np.full(case.n_gen, np.nan)
    vm = np.full(case.n_bus, np.nan)
    va = np.full(case.n_bus, np.nan)
    pf = np.full(case.n_branch, np.nan)
    qf = np.full(case.n_branch, np.nan)
    pt = np.full(case.n_branch, np.nan)
    qt = np.full(case.n_branch, np.nan)
    bus_position = {
        int(bus_id): index for index, bus_id in enumerate(case.bus[:, 0])
    }
    for line in lines[2:bus_header]:
        identity, pg_value, qg_value = line.split("|")
        index = int(identity[1:]) - 1
        pg[index], qg[index] = float(pg_value), float(qg_value)
    for line in lines[bus_header + 1 : branch_header]:
        identity, vm_value, va_value = line.split("|")
        index = bus_position[int(identity)]
        vm[index], va[index] = float(vm_value), float(va_value)
    for line in lines[branch_header + 1 :]:
        identity, pf_value, qf_value, pt_value, qt_value = line.split("|")
        index = int(identity[1:]) - 1
        pf[index], qf[index] = float(pf_value), float(qf_value)
        pt[index], qt[index] = float(pt_value), float(qt_value)
    if not all(
        np.isfinite(value).all()
        for value in (pg, qg, vm, va, pf, qf, pt, qt)
    ):
        raise ValueError("target is incomplete or contains non-finite values")
    return {
        "objective": float(objective_match.group(1)),
        "pg_pu": pg,
        "qg_pu": qg,
        "vm_pu": vm,
        "va_rad": va,
        "pf_pu": pf,
        "qf_pu": qf,
        "pt_pu": pt,
        "qt_pu": qt,
    }
