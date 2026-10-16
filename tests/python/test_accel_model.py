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
    for p in [*a.PARTS.values(), *a.OTHER_PARTS.values(), *a.IMC.values()]:
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
    used |= {x for f in [*a.FAMILIES, *a.PUBLISHED.values(), *a.OTHER_PARTS.values()] for x in f.get("also", [])}
    for sid, s in a.SOURCES.items():
        assert a.ACCESSED == "2026-10-15" and a.ACCESSED_42 == "2026-10-16"
        assert s["accessed"] in (a.ACCESSED, a.ACCESSED_42), sid
        assert s["url"].startswith("https://")
        assert sid in used, f"source {sid} is never cited"


def test_availability_values():
    for p in a.PARTS.values():
        assert p["avail"]["v"] in ("product", "announced", "retired")
    assert a.part("wse-3t")["avail"] == "announced"
    assert a.part("groq3")["avail"] == "announced"
    assert a.part("gc2")["avail"] == "retired"
    assert {p["avail"]["v"] for p in a.OTHER_PARTS.values()} == {"announced", "retired", "product"}
    assert a.IMC["hermes"]["avail"]["v"] == "research"
    # quoted parts lack a figure the model needs, so none of them is in PARTS
    assert not set(a.OTHER_PARTS) & set(a.PARTS)


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
    assert len(v) == 10
    assert v["cerebras-70b-wafers"]["ratio"] == 1.0
    assert v["cerebras-20b-wafer"]["ratio"] == 1.0
    assert v["groq-70b-chips"]["model"] == 301
    # with prefill counted (mean prompt 222.33 tokens, brief 42) both GPU rows reach about a third
    assert 0.3 < v["mlperf-h100"]["ratio"] < 0.4
    assert 0.3 < v["mlperf-gb200"]["ratio"] < 0.4
    for k in ("groq-70b-tps", "cerebras-70b-tps", "cerebras-8b-tps"):
        assert v[k]["ratio"] < 0.1  # the bound leaves out per-layer latency
    # the SN40L paper's 8B figure, without speculative decoding, is about 0.58 of the bandwidth bound
    assert 0.55 < v["sn40l-8b-tps"]["ratio"] < 0.6
    # with speculative decoding the 70B figure beats a one-token-per-step bound
    assert v["sn40l-70b-tps"]["ratio"] > 1 and v["sn40l-70b-tps"]["kind"] == "speculative"
    b = a.band()
    assert b["capacity"]["n"] == 3 and b["latency"]["n"] == 4 and b["throughput"]["n"] == 2
    assert "speculative" not in b


def test_implied_overhead_is_tens_of_microseconds_per_layer():
    for k in ("groq-70b-tps", "cerebras-70b-tps", "cerebras-8b-tps"):
        o = a.implied_overhead(k)
        assert 10e-6 < o["per_layer"] < 50e-6


# --- chapters 6-10 -------------------------------------------------------------


def test_layer_ops_carry_every_weight_of_a_layer():
    for m in a.MODEL_ORDER:
        tot = 0
        for o in a.layer_ops(m):
            tot += o["w"]
        assert tot == a.shape(m)["per_layer"]


def test_fusion_keeps_weights_and_kv_and_drops_the_intermediates():
    for case in ((1, 1, 8192, False), (2048, 1, 2048, True)):
        u = a.layer_traffic("sn40l", "llama3-8b", "bf16", *case, False)
        f = a.layer_traffic("sn40l", "llama3-8b", "bf16", *case, True)
        uw = sum(o["w_bytes"] + o["kv_bytes"] for o in u["ops"])
        fw = sum(o["w_bytes"] + o["kv_bytes"] for o in f["ops"])
        assert uw == fw  # the same weights and KV cache either way
        assert f["bytes"] < u["bytes"] and f["t"] <= u["t"]
        assert u["flops"] == f["flops"]
    # at batch 1 fusion saves almost nothing; in a prefill most of the traffic
    u = a.layer_traffic("sn40l", "llama3-8b", "bf16", 1, 1, 8192, False, False)
    f = a.layer_traffic("sn40l", "llama3-8b", "bf16", 1, 1, 8192, False, True)
    assert 1 - f["bytes"] / u["bytes"] < 0.01
    u = a.layer_traffic("sn40l", "llama3-8b", "bf16", 2048, 1, 2048, True, False)
    f = a.layer_traffic("sn40l", "llama3-8b", "bf16", 2048, 1, 2048, True, True)
    assert 1 - f["bytes"] / u["bytes"] > 0.5


def test_dataflow_steps_end_at_the_layer_totals():
    st = a.dataflow_steps("sn40l", "llama3-70b", "bf16", 2048, 1, 2048, True)
    u = a.layer_traffic("sn40l", "llama3-70b", "bf16", 2048, 1, 2048, True, False)
    f = a.layer_traffic("sn40l", "llama3-70b", "bf16", 2048, 1, 2048, True, True)
    assert st[-1]["u_bytes"] == pytest.approx(u["bytes"], rel=1e-12)
    assert st[-1]["u_t"] == pytest.approx(u["t"], rel=1e-12)
    assert st[-1]["f_t"] == pytest.approx(f["t"], rel=1e-12)


def test_tiers():
    x = a.tiers("sn40l", "llama3-70b", "bf16", 16)
    assert x["hbm_models"] == 7 and x["ddr_models"] == 170
    # loading one 70B model from DDR over 16 sockets' DDR bandwidth
    assert x["switch_s"] == pytest.approx(a.shape("llama3-70b")["params"] * 2 / (16 * 200e9))


def test_mesh_layer():
    one = a.mesh_layer("blackhole", "llama3-70b", "fp8", 1, 1024, 1)
    two = a.mesh_layer("blackhole", "llama3-70b", "fp8", 1, 1024, 2)
    assert one["t_eth"] == 0  # no all-reduce on one card
    assert two["t_eth"] > 0
    assert two["t_dram"] == pytest.approx(one["t_dram"] / 2, rel=1e-9)
    assert not one["layer_fits"]  # a 70B layer is bigger than the cores' SRAM: it streams from GDDR
    st = a.mesh_steps("blackhole", "llama3-70b", "fp8", 1, 1024, 2)
    assert st[-1]["t"] == pytest.approx(two["t"], rel=1e-12)
    assert [s["kind"] for s in st].count("ethernet") == 2


def test_edge_bound_scales_with_bits():
    st = a.edge_steps("m4", "llama3-8b", 1024)
    assert [s["bits"] for s in st] == list(a.BITS)
    for x, y in zip(st, st[1:]):
        assert y["t"] < x["t"]
    v = a.edge_bound("v70", "llama3-8b", 8, 1024)
    assert v["fits"] is None  # the V70's DDR capacity is not disclosed
    assert v["j_per_token"] == pytest.approx(75 * v["t"])
    assert a.edge_bound("m4", "llama3-8b", 8, 1024)["j_per_token"] is None


def test_imc_layer():
    x = a.imc_layer("llama3-8b", 1)
    assert x["layer_tiles"] == 16 * 24 + 16 * 16 + 16 * 112 + 56 * 16
    assert x["capacity"] >= x["weights"]
    assert (x["chips"] - 1) * 4194304 < x["weights"]
    assert x["t_layer"] == pytest.approx(4 * 133e-9)
    assert a.imc_layer("llama3-8b", 4)["t_layer"] == pytest.approx(4 * 520e-9)
    st = a.imc_steps("llama3-8b", 1)
    assert [s["name"] for s in st] == ["program", "QKV", "attention", "attention out", "MLP up and gate", "MLP down"]
    assert st[-1]["t"] == pytest.approx(x["t_layer"])
    assert a.adc_bits(8, 8, 256) == 24


def test_choose_covers_every_family_once():
    rows = a.choose("llama3-70b", 1, 1024, 2048)
    assert [r["family"] for r in rows] == [f["id"] for f in a.FAMILIES]
    for r in rows:
        assert r["t_decode"] > 0 and r["t_prefill"] > 0 and r["chips"] >= 1
        if r["util"] is not None:
            assert 0 < r["util"] <= 1
    by = {r["family"]: r for r in rows}
    assert by["gpu"]["t_decode"] == a.decode("h100", "llama3-70b", "fp8", 1, 1024)["t_step"]


def test_mlperf_ceiling_counts_prefill():
    P = {k: f["v"] for k, f in a.PUBLISHED.items()}
    m = a.mlperf_ceiling("h100", "llama2-70b", "fp8", 8, P["mlperf-input-mean"], P["mlperf-tokens-per-sample"])
    # the old decode-only ceiling is the same function with no prompt and no attention
    assert m["decodeOnly"] == a.throughput_ceiling("h100", "llama2-70b", "fp8", 8)
    assert m["ceiling"] < m["decodeOnly"]
    assert 0.4 < m["prefillShare"] < 0.45  # 222 prompt tokens against 294 output tokens
    sh = a.shape("llama2-70b")
    f = a.sample_flops("llama2-70b", 0, 1)
    assert f["prefill"] == 0 and f["total"] == 2 * sh["matmul_params"] + sh["attn_flops_per_pos"] * 0.5
    assert P["mlperf-input-mean"] == pytest.approx(222.3298, abs=1e-4)


def test_floor_fit_is_leave_one_out():
    f = a.floor_fit()
    rows = {r["id"]: r for r in f["rows"]}
    assert set(rows) == set(a.FLOOR_CHECKS)
    for r in f["rows"]:
        others = [x["per_layer"] for x in f["rows"] if x["id"] != r["id"]]
        assert r["tauLoo"] == pytest.approx(sum(others) / 3, rel=1e-12)
        assert r["tPred"] == pytest.approx(r["t_model"] + r["layers"] * r["tauLoo"], rel=1e-12)
    assert f["tau"] == pytest.approx(sum(r["per_layer"] for r in f["rows"]) / 4, rel=1e-12)
    # the term turns misses of up to 770x into misses of under 2x
    assert f["worstBound"] > 700 and f["worstPred"] < 2
    # and on the wafer, the floor is nearly all of a token's time
    assert rows["cerebras-70b-tps"]["floorShare"] > 0.99


def test_critical_path_adds_up():
    for c in a.FLOOR_CHECKS:
        r = next(x for x in a.floor_fit()["rows"] if x["id"] == c)
        st = a.critical_path_steps(c, "own")
        assert len(st) == r["layers"] + 1
        assert st[-1]["t"] == pytest.approx(r["t_published"], rel=1e-9)
        assert a.critical_path_steps(c, "none")[-1]["t"] == pytest.approx(r["t_model"], rel=1e-9)


def test_new_parts():
    m = a.part("mi355x")
    assert (m["peak_bf16"], m["peak_fp8"], m["off_bytes"], m["off_bw"]) == (2.5e15, 5e15, 288e9, 8e12)
    # 288 GB holds Llama 3.1 70B at FP8 on one GPU, as the B200's 186 GB does
    assert a.placement("mi355x", "llama3-70b", "fp8", 1, 4096)["chips"] == 1
    e = a.edge_bound("v80", "llama3-8b", 4, 1024)
    assert e["fits"] and e["t_comp"] is None
    assert e["tok_s"] == pytest.approx(820e9 / e["bytes"], rel=1e-12)
