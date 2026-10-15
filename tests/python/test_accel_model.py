"""Tests of the Python reference (accel_model): data hygiene, closed forms, and the validation checks."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import accel_model as a  # noqa: E402

STATUSES = {"spec", "derived", "reported", "illustrative", "undisclosed"}


def all_fields():
    for p in a.PARTS.values():
        yield p["avail"]
        yield from p["fields"].values()
    for f in a.FAMILIES:
        for k in ("placement", "schedule", "grain"):
            yield f[k]
    yield from a.PUBLISHED.values()
    yield from a.ILLUSTRATIVE.values()


def test_every_field_has_a_known_source_and_status():
    for f in all_fields():
        assert f["src"] in a.SOURCES, f
        assert f["st"] in STATUSES, f
        assert (f["v"] is None) == (f["st"] == "undisclosed"), f
        assert f["ref"]


def test_every_source_is_dated_and_used():
    used = {f["src"] for f in all_fields()} | {c["src"] for c in a.CLAIMS.values()} | {m["src"] for m in a.MODELS.values()}
    used |= {x for f in [*a.FAMILIES, *a.PUBLISHED.values()] for x in f.get("also", [])}
    for sid, s in a.SOURCES.items():
        assert s["accessed"] == a.ACCESSED == "2026-10-15"
        assert s["url"].startswith("https://")
        assert sid in used, f"source {sid} is never cited"


def test_availability_values():
    for p in a.PARTS.values():
        assert p["avail"]["v"] in ("product", "announced", "retired")
    assert a.part("wse-3t")["avail"] == "announced"
    assert a.part("groq3")["avail"] == "announced"


def test_parameter_counts_match_the_model_names():
    # Llama 3 Table 3 shapes give the published sizes
    assert a.shape("llama3-8b")["params"] == 8_027_897_856
    assert abs(a.shape("llama3-70b")["params"] / 1e9 - 70.55) < 0.01
    assert abs(a.shape("llama2-70b")["params"] / 1e9 - 68.98) < 0.01
    s = a.shape("llama3-70b")
    assert s["head_dim"] == 128
    assert s["kv_elems_per_token"] == 2 * 80 * 8 * 128


def test_peak_basis():
    assert a.peak("h100", "fp8") == {"flops": 1979e12, "basis": "dense", "key": "peak_fp8"}
    assert a.peak("groqchip", "bf16")["key"] == "peak_fp16"
    assert a.peak("wse-3", "bf16")["basis"] == "sparse"
    assert a.peak("wse-3", "fp8")["flops"] is None


def test_placement_is_the_fewest_chips_that_hold_it():
    for pid in a.PART_ORDER:
        for mid in a.MODEL_ORDER:
            pl = a.placement(pid, mid, "bf16", 4, 2048)
            assert pl["chips"] * pl["cap"] >= pl["need"]
            assert (pl["chips"] - 1) * pl["cap"] < pl["need"]
            assert 0 < pl["fill"] <= 1


def test_decode_terms():
    d = a.decode("h100", "llama3-70b", "fp8", 1, 4096, 8)
    assert d["bound"] == "memory" and d["stages"] == 1 and d["group"] == 8
    assert d["t_mem"] == pytest.approx(d["bytes"] / (8 * 3.35e12), rel=1e-15)
    assert not d["hop_known"] and d["t_hops"] == 0
    # big batches become compute-bound on a GPU
    assert a.decode("h100", "llama3-70b", "fp8", 2048, 512, 8)["bound"] == "compute"
    # the pipeline: latency = weights / (one group's bandwidth) + hops
    g = a.decode("groqchip", "llama2-70b", "int8", 1, 1024, 576, 8)
    assert g["stages"] == 72
    assert g["hops"] == 71 + 160
    assert g["t_hops"] == pytest.approx(231 * 722e-9)
    assert g["bound"] == "latency"
    assert g["tok_s"] == pytest.approx(72 * g["tok_s_user"])


def test_ridge_and_ceiling():
    assert a.ridge("h100", "fp8") == pytest.approx(1979e12 / 3.35e12)
    assert a.throughput_ceiling("h100", "llama2-70b", "fp8", 8) == pytest.approx(8 * 1979e12 / (2 * a.shape("llama2-70b")["matmul_params"]))


def test_batch_sweep_crosses_the_ridge_once():
    sw = a.batch_sweep("h100", "llama3-70b", "fp8", 1024, 8)
    kinds = [s["bound"] for s in sw]
    first = kinds.index("compute")
    assert all(k == "memory" for k in kinds[:first]) and all(k == "compute" for k in kinds[first:])


def test_mxu_closed_form_and_steps_agree():
    for rows, cols in ((128, 128), (256, 256)):
        for m in (1, 8, 128, 512):
            r = a.mxu(rows, cols, m, 1024, 768)
            st = a.mxu_steps(rows, cols, m, 1024, 768)
            assert st[-1]["cycles"] == r["cycles"] and st[-1]["macs"] == r["macs"]
            assert st[-1]["util"] == pytest.approx(r["util"])
            assert 0 < r["util"] <= 1
    # batch 1 uses about 1/rows of the array; a batch of rows fills it
    assert a.layer_mxu(256, 256, "llama3-70b", 1)["util"] < 1 / 250
    assert a.layer_mxu(256, 256, "llama3-70b", 256)["util"] > 0.99


def test_wafer_yield():
    y = a.wafer_yield("wse-3", 0.1, 814, 0.015)
    assert y["die_yield"] == pytest.approx(math.exp(-0.814))
    assert y["wafer_perfect"] < 1e-19
    assert y["spares"] == 13500 and y["spare_margin"] > 100


def test_defects_are_deterministic_and_on_the_grid():
    ds = a.defects(7, 46)
    assert ds == a.defects(7, 46)
    assert all(0 <= d["col"] < 12 and 0 <= d["row"] < 7 and 0 <= d["u"] < 10 and 0 <= d["v"] < 10 for d in ds)
    st = a.defect_steps(7, 46, 4)
    assert st[0]["shown"] == 0 and st[-1]["shown"] == 46
    assert all(x["dies_hit"] <= y["dies_hit"] for x, y in zip(st, st[1:]))


def test_chain_steps_sum_to_the_decode_bound():
    for chips, group in ((576, 8), (310, 1), (64, 8)):
        d = a.decode("groqchip", "llama2-70b", "int8", 1, 1024, chips, group)
        st = a.chain_steps("groqchip", "llama2-70b", "int8", 1024, chips, group)
        assert len(st) == d["stages"]
        assert st[-1]["done"] == pytest.approx(d["t_step"], rel=1e-12)


def test_validation_checks():
    v = {r["id"]: r for r in a.validation()}
    assert len(v) == 7
    assert v["cerebras-70b-wafers"]["ratio"] == 1.0
    assert v["cerebras-20b-wafer"]["ratio"] == 1.0
    assert v["groq-70b-chips"]["model"] == 301
    assert 0.18 < v["mlperf-h100"]["ratio"] < 0.2
    for k in ("groq-70b-tps", "cerebras-70b-tps", "cerebras-8b-tps"):
        assert v[k]["ratio"] < 0.1  # the bound leaves out per-layer latency
    b = a.band()
    assert b["capacity"]["n"] == 3 and b["latency"]["n"] == 3


def test_implied_overhead_is_tens_of_microseconds_per_layer():
    for k in ("groq-70b-tps", "cerebras-70b-tps", "cerebras-8b-tps"):
        o = a.implied_overhead(k)
        assert 10e-6 < o["per_layer"] < 50e-6
