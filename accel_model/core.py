"""
The accelerator model: closed forms of the sourced data (data.py).

A teaching model, not a simulator of any product. For each part it computes
where a decoder model's weights must live and how many chips that takes, and
the bound on one decode step and one prefill: the larger of the memory term
(bytes that must be read / bandwidth) and the compute term (flops / peak
rate), plus the chip-to-chip latency the layout forces, where the vendor
discloses it. Families differ explicitly, not through fudge factors:

  - "tensor" layout (GPUs, systolic chips): every chip holds a slice of every
    layer and works on every token, so the memory and compute terms divide by
    the chip count; per-layer collectives are latency the sources do not give
    (left out, and flagged);
  - "pipeline" layout (wafer-scale, compiler-scheduled chips): the model is cut
    at layer boundaries into stages of `group` chips, a token passes through
    the stages in turn, so one token's latency is the sum of the stages'
    times plus a hop between stages, and a static schedule has no cache
    misses: every byte is read from SRAM exactly once per step.

The TypeScript port (src/lib/accel/model.ts) repeats every function with the
same operations in the same order; the parity fixtures must match exactly,
except the functions that call exp(), which are compared to 1e-12.

Conventions: bytes and bits SI (1 GB = 1e9 B); bandwidths per direction; peak
rates dense unless a result says `basis: "sparse"`.
"""
from __future__ import annotations

import math
from typing import Any

from .data import FAMILIES, FORMATS, ILLUSTRATIVE, IMC, MODELS, OTHER_PARTS, PARTS, PUBLISHED

# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def part(pid: str) -> dict[str, Any]:
    """A part's values, flattened. Undisclosed figures are None."""
    p = PARTS[pid]
    out: dict[str, Any] = {
        "id": p["id"],
        "name": p["name"],
        "family": p["family"],
        "layout": p["layout"],
        "avail": p["avail"]["v"],
    }
    for k, f in p["fields"].items():
        out[k] = f["v"]
    return out


def fval(pid: str, key: str) -> Any:
    """One field's value, or None when the part has no such field."""
    f = PARTS[pid]["fields"].get(key)
    return None if f is None else f["v"]


def shape(mid: str) -> dict[str, Any]:
    """
    Parameter counts and per-token costs of a Llama-style decoder (untied
    embeddings, grouped-query attention, a gated MLP of three matrices).
    """
    m = MODELS[mid]
    L, d, ffn, h, kv, V = m["layers"], m["d"], m["ffn"], m["heads"], m["kv_heads"], m["vocab"]
    hd = d // h
    attn = d * d + 2 * d * (kv * hd) + d * d
    mlp = 3 * d * ffn
    per_layer = attn + mlp
    embed = V * d
    params = 2 * embed + L * per_layer
    matmul = params - embed  # the input embedding is a lookup, not a matmul
    return {
        "id": mid,
        "name": m["name"],
        "layers": L,
        "d": d,
        "ffn": ffn,
        "heads": h,
        "kv_heads": kv,
        "head_dim": hd,
        "vocab": V,
        "per_layer": per_layer,
        "params": params,
        "matmul_params": matmul,
        "kv_elems_per_token": 2 * L * kv * hd,
        # attention reads every earlier position: QK^T and AV, 2 flops per MAC
        "attn_flops_per_pos": 4 * L * d,
    }


def peak(pid: str, fmt: str) -> dict[str, Any]:
    """
    The peak rate a part's figures give for a format, and its basis. 16-bit
    work uses BF16, else FP16, else the sparse FP16 figure (flagged sparse:
    the dense rate is not disclosed, so a compute term from it is a floor).
    """
    if fmt == "bf16":
        for key, basis in (("peak_bf16", "dense"), ("peak_fp16", "dense"), ("peak_fp16_sparse", "sparse")):
            v = fval(pid, key)
            if v is not None:
                return {"flops": v, "basis": basis, "key": key}
        return {"flops": None, "basis": "none", "key": ""}
    key = "peak_fp8" if fmt == "fp8" else "peak_int8"
    v = fval(pid, key)
    return {"flops": v, "basis": "dense" if v is not None else "none", "key": key if v is not None else ""}


def kv_bytes_per_elem(fmt: str) -> float:
    """The KV cache is kept in the weights' format, at least one byte."""
    return max(1.0, FORMATS[fmt])


# ---------------------------------------------------------------------------
# Where the weights live, and how many chips that takes
# ---------------------------------------------------------------------------


def placement(pid: str, mid: str, fmt: str, batch: int, ctx: int) -> dict[str, Any]:
    """
    Weights plus the KV cache of `batch` sequences of `ctx` tokens, in the
    part's main memory: HBM (or other DRAM) when it has some, else on-chip
    SRAM. `chips` is the fewest chips whose memory holds it all.
    """
    s = shape(mid)
    b = FORMATS[fmt]
    weights = s["params"] * b
    weights_read = s["matmul_params"] * b
    kv_tok = s["kv_elems_per_token"] * kv_bytes_per_elem(fmt)
    kv = batch * ctx * kv_tok
    need = weights + kv
    off = fval(pid, "off_bytes")
    if off is not None and off > 0:
        where = "off-chip"
        cap = off
        bw = fval(pid, "off_bw")
    else:
        where = "sram"
        cap = fval(pid, "sram_bytes")
        bw = fval(pid, "sram_bw")
    chips = math.ceil(need / cap)
    return {
        "weights": weights,
        "weights_read": weights_read,
        "kv_per_token": kv_tok,
        "kv": kv,
        "need": need,
        "where": where,
        "cap": cap,
        "bw": bw,
        "chips": chips,
        "fill": need / (chips * cap),
    }


# ---------------------------------------------------------------------------
# One decode step and one prefill
# ---------------------------------------------------------------------------


def layout_of(pid: str, chips: int, group: int) -> dict[str, int]:
    """Stages and chips per stage: tensor layout = one stage of all chips."""
    if PARTS[pid]["layout"] == "tensor":
        return {"stages": 1, "group": chips}
    return {"stages": math.ceil(chips / group), "group": group}


def decode(pid: str, mid: str, fmt: str, batch: int, ctx: int, chips: int = 0, group: int = 1) -> dict[str, Any]:
    """
    The bound on one decode step: every weight and the batch's KV cache read
    once, 2 flops per weight per sequence plus attention over `ctx` positions,
    and the latency of the hops the layout forces. `chips` = 0 means the fewest
    that hold the model.
    """
    s = shape(mid)
    pl = placement(pid, mid, fmt, batch, ctx)
    n = max(chips, pl["chips"])
    lay = layout_of(pid, n, group)
    g = lay["group"]
    st = lay["stages"]
    bytes_step = pl["weights_read"] + pl["kv"]
    t_mem = bytes_step / (g * pl["bw"])
    flops = 2 * s["matmul_params"] * batch + s["attn_flops_per_pos"] * ctx * batch
    pk = peak(pid, fmt)
    t_comp = None if pk["flops"] is None else flops / (g * pk["flops"])
    hop = fval(pid, "hop_s")
    hops = (st - 1) + (2 * s["layers"] if g > 1 else 0)
    t_hops = 0.0 if hop is None else hops * hop
    t_work = t_mem if t_comp is None or t_mem >= t_comp else t_comp
    t_step = t_work + t_hops
    if t_hops > t_work:
        bound = "latency"
    elif t_comp is not None and t_comp > t_mem:
        bound = "compute"
    else:
        bound = "memory"
    tok_s_user = 1 / t_step
    # a pipeline keeps one batch in flight per stage
    tok_s = batch * st / t_step
    pw = fval(pid, "power_w")
    j_tok = None if pw is None else n * pw / tok_s
    return {
        "chips": n,
        "stages": st,
        "group": g,
        "where": pl["where"],
        "bytes": bytes_step,
        "flops": flops,
        "t_mem": t_mem,
        "t_comp": t_comp,
        "hops": hops,
        "t_hops": t_hops,
        "hop_known": hop is not None,
        "t_step": t_step,
        "bound": bound,
        "basis": pk["basis"],
        "tok_s_user": tok_s_user,
        "tok_s": tok_s,
        "j_per_token": j_tok,
        "intensity": flops / bytes_step,
    }


def prefill(pid: str, mid: str, fmt: str, prompt: int, chips: int = 0, group: int = 1) -> dict[str, Any]:
    """
    The bound on prefilling one prompt: the weights read once, 2 flops per
    weight per prompt token plus causal attention (on average half the prompt
    per token).
    """
    s = shape(mid)
    pl = placement(pid, mid, fmt, 1, prompt)
    n = max(chips, pl["chips"])
    lay = layout_of(pid, n, group)
    g = lay["group"]
    t_mem = (pl["weights_read"] + pl["kv"]) / (g * pl["bw"])
    flops = 2 * s["matmul_params"] * prompt + s["attn_flops_per_pos"] * (prompt * prompt / 2)
    pk = peak(pid, fmt)
    t_comp = None if pk["flops"] is None else flops / (g * pk["flops"])
    t = t_mem if t_comp is None or t_mem >= t_comp else t_comp
    return {
        "chips": n,
        "t_mem": t_mem,
        "t_comp": t_comp,
        "t": t,
        "bound": "compute" if t_comp is not None and t_comp > t_mem else "memory",
        "basis": pk["basis"],
        "flops": flops,
        "tok_s": prompt / t,
    }


def ridge(pid: str, fmt: str) -> float | None:
    """Flops per byte where the compute and memory terms meet."""
    pl_bw = fval(pid, "off_bw") or fval(pid, "sram_bw")
    pk = peak(pid, fmt)["flops"]
    return None if pk is None or not pl_bw else pk / pl_bw


def throughput_ceiling(pid: str, mid: str, fmt: str, chips: int) -> float:
    """Output tokens per second if every chip ran at peak on the weight matmuls (no attention, no prefill)."""
    s = shape(mid)
    return chips * peak(pid, fmt)["flops"] / (2 * s["matmul_params"])


def peak_any(pid: str, fmt: str) -> float | None:
    """`peak` for bf16/fp8/int8; for "fp4" the part's dense 4-bit rate (NVFP4 or MXFP4), if stated."""
    if fmt == "fp4":
        return fval(pid, "peak_fp4")
    return peak(pid, fmt)["flops"]


def sample_flops(mid: str, n_in: float, n_out: float) -> dict[str, float]:
    """
    The flops one benchmark sample costs: prefill of n_in prompt tokens
    (weights plus causal attention over half the prompt on average) and
    n_out decode steps (weights plus attention over the prompt and, on
    average, half the output so far).
    """
    s = shape(mid)
    w = 2 * s["matmul_params"]
    a = s["attn_flops_per_pos"]
    prefill = w * n_in + a * (n_in * n_in / 2)
    decode = w * n_out + a * n_out * (n_in + n_out / 2)
    return {"prefill": prefill, "decode": decode, "total": prefill + decode}


def mlperf_ceiling(pid: str, mid: str, fmt: str, chips: int, n_in: float, n_out: float) -> dict[str, Any]:
    """
    MLPerf Offline's output tokens per second if every chip ran at peak on
    everything a sample costs (prefill included): chips x peak / flops per
    sample x output tokens per sample. Also the decode-only ceiling and how
    much prefill and attention lower it.
    """
    pk = peak_any(pid, fmt)
    f = sample_flops(mid, n_in, n_out)
    full = chips * pk / f["total"] * n_out
    decode_only = chips * pk / (2 * shape(mid)["matmul_params"])
    return {"ceiling": full, "decodeOnly": decode_only, "prefillShare": f["prefill"] / f["total"],
            "flops": f, "peak": pk, "chips": chips}


# ---------------------------------------------------------------------------
# Chapter 1: the map
# ---------------------------------------------------------------------------

# The representative part and format each family's map step quotes.
MAP_PARTS: dict[str, tuple[str, str]] = {
    "gpu": ("h100", "fp8"),
    "systolic": ("tpu-7x", "fp8"),
    "wafer": ("wse-3", "bf16"),
    "lpu": ("groqchip", "int8"),
    "dataflow": ("sn40l", "bf16"),
    "mesh": ("blackhole", "fp8"),
    "fpga-npu": ("m4", "int8"),
}


def family_map() -> list[dict[str, Any]]:
    """The families with their positions on the axes."""
    out = []
    for i, f in enumerate(FAMILIES):
        out.append(
            {
                "i": i,
                "id": f["id"],
                "name": f["name"],
                "chapter": f["chapter"],
                "example": f["example"],
                "placement": f["placement"]["v"],
                "schedule": f["schedule"]["v"],
                "grain": f["grain"]["v"],
            }
        )
    return out


def map_steps(mid: str, batch: int, ctx: int) -> list[dict[str, Any]]:
    """One step per family: where the running model's weights live, and (for the parts in this part of the site) how many chips that takes."""
    out = []
    for f in family_map():
        step = dict(f)
        if f["id"] in MAP_PARTS:
            pid, fmt = MAP_PARTS[f["id"]]
            pl = placement(pid, mid, fmt, batch, ctx)
            step.update({"part": pid, "fmt": fmt, "chips": pl["chips"], "need": pl["need"], "where": pl["where"], "cap": pl["cap"]})
        else:
            step.update({"part": "", "fmt": "", "chips": 0, "need": 0, "where": "", "cap": 0})
        out.append(step)
    return out


# ---------------------------------------------------------------------------
# Chapter 2: GPUs, the roofline over batch size
# ---------------------------------------------------------------------------

BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)


def batch_sweep(pid: str, mid: str, fmt: str, ctx: int, chips: int) -> list[dict[str, Any]]:
    """Decode bounds at each batch size: the frames of chapter 2's animation."""
    cap = max_batch(pid, mid, fmt, ctx, chips)
    out = []
    for i, b in enumerate(BATCHES):
        d = decode(pid, mid, fmt, b, ctx, chips)
        d["i"] = i
        d["batch"] = b
        d["fits"] = b <= cap
        out.append(d)
    return out


def max_batch(pid: str, mid: str, fmt: str, ctx: int, chips: int) -> int:
    """The largest batch whose KV cache still fits beside the weights on `chips` chips."""
    pl = placement(pid, mid, fmt, 1, ctx)
    free = chips * pl["cap"] - pl["weights"]
    return 0 if free <= 0 else math.floor(free / (ctx * pl["kv_per_token"]))


# ---------------------------------------------------------------------------
# Chapter 3: systolic arrays, tiles and utilisation
# ---------------------------------------------------------------------------


def layer_gemms(mid: str) -> list[dict[str, Any]]:
    """The weight matrices of one decoder layer, k (in) x n (out)."""
    s = shape(mid)
    d, kvd = s["d"], s["kv_heads"] * s["head_dim"]
    return [
        {"name": "QKV", "k": d, "n": d + 2 * kvd},
        {"name": "attention out", "k": d, "n": d},
        {"name": "MLP up and gate", "k": d, "n": 2 * s["ffn"]},
        {"name": "MLP down", "k": s["ffn"], "n": d},
    ]


def mxu(rows: int, cols: int, m: int, k: int, n: int) -> dict[str, Any]:
    """
    A weight-stationary array of rows x cols PEs running an m x k by k x n
    matmul, tile after tile with the next tile's weights shifted in behind the
    current one (shadow weights): each tile holds the array for max(m, rows)
    cycles (m input rows streamed, or rows cycles to load the next weights),
    plus one fill and drain of rows + cols - 2 cycles.
    """
    tiles = math.ceil(k / rows) * math.ceil(n / cols)
    per_tile = max(m, rows)
    cycles = tiles * per_tile + rows + cols - 2
    macs = m * k * n
    return {"tiles": tiles, "per_tile": per_tile, "cycles": cycles, "macs": macs, "util": macs / (rows * cols * cycles)}


def layer_mxu(rows: int, cols: int, mid: str, m: int) -> dict[str, Any]:
    """One decoder layer's matmuls on one array: cycles and utilisation."""
    cyc = 0
    macs = 0
    for g in layer_gemms(mid):
        r = mxu(rows, cols, m, g["k"], g["n"])
        cyc += r["cycles"]
        macs += r["macs"]
    return {"cycles": cyc, "macs": macs, "util": macs / (rows * cols * cyc)}


def mxu_steps(rows: int, cols: int, m: int, k: int, n: int) -> list[dict[str, Any]]:
    """Chapter 3's frames: the tiles of one matmul in order, with the cycles and MACs so far."""
    tr = math.ceil(k / rows)
    tc = math.ceil(n / cols)
    per = max(m, rows)
    out = [{"i": 0, "tile": -1, "tr": -1, "tc": -1, "cycles": rows + cols - 2, "macs": 0, "util": 0.0}]
    macs = 0
    for t in range(tr * tc):
        a = t // tc
        c = t % tc
        kk = min(rows, k - a * rows)
        nn = min(cols, n - c * cols)
        macs += m * kk * nn
        cycles = (t + 1) * per + rows + cols - 2
        out.append({"i": t + 1, "tile": t, "tr": a, "tc": c, "cycles": cycles, "macs": macs, "util": macs / (rows * cols * cycles)})
    return out


# ---------------------------------------------------------------------------
# Chapter 4: wafer-scale
# ---------------------------------------------------------------------------

DIE_COLS = 12
DIE_ROWS = 7  # 84 dies drawn as 12 x 7 (the vendor gives the count, not the layout)


def poisson_zero(lam: float) -> float:
    """Probability of no defect when `lam` are expected (Poisson)."""
    return math.exp(-lam)


def wafer_yield(pid: str, d0_per_cm2: float, die_mm2: float, spare_frac: float) -> dict[str, Any]:
    """
    Expected defects on a whole wafer at defect density d0, the yield of a
    reticle-sized die and of a defect-free wafer, and the cores a wafer loses
    if each defect takes out one core, against its spares.
    """
    area = fval(pid, "area_mm2")
    cores = fval(pid, "units")
    lam_wafer = d0_per_cm2 * area / 100
    lam_die = d0_per_cm2 * die_mm2 / 100
    spares = math.floor(cores * spare_frac)
    return {
        "lam_wafer": lam_wafer,
        "lam_die": lam_die,
        "die_yield": poisson_zero(lam_die),
        "wafer_perfect": poisson_zero(lam_wafer),
        "spares": spares,
        "lost_share": lam_wafer / cores,
        "spare_margin": spares / lam_wafer,
        "die_cores": cores / fval(pid, "dies"),
    }


def xorshift32(x: int) -> int:
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= x >> 17
    x ^= (x << 5) & 0xFFFFFFFF
    return x & 0xFFFFFFFF


def defects(seed: int, count: int) -> list[dict[str, int]]:
    """`count` defect sites: a die (column, row) and a core cell (u, v) in a 10 x 10 grid of the die, from a seeded generator."""
    x = seed
    out = []
    for i in range(count):
        x = xorshift32(x)
        die = x % (DIE_COLS * DIE_ROWS)
        x = xorshift32(x)
        cell = x % 100
        out.append({"i": i, "col": die % DIE_COLS, "row": die // DIE_COLS, "u": cell % 10, "v": cell // 10})
    return out


def defect_steps(seed: int, count: int, per_step: int) -> list[dict[str, Any]]:
    """Chapter 4's frames: defects arriving `per_step` at a time; each kills one core, which a spare replaces."""
    ds = defects(seed, count)
    out = []
    k = 0
    while True:
        shown = min(k * per_step, count)
        dies_hit = len({(d["col"], d["row"]) for d in ds[:shown]})
        out.append({"i": k, "shown": shown, "dies_hit": dies_hit})
        if shown == count:
            break
        k += 1
    return out


def wafers_needed(pid: str, mid: str, fmt: str, batch: int, ctx: int) -> dict[str, Any]:
    """Wafers to hold a model, and the time to stream its weights in once over the system's I/O instead."""
    pl = placement(pid, mid, fmt, batch, ctx)
    io = fval(pid, "io_bps") / 8
    t_stream = pl["weights_read"] / io
    return {"wafers": pl["chips"], "need": pl["need"], "fill": pl["fill"], "io": io, "t_stream": t_stream, "stream_tok_s": 1 / t_stream}


# ---------------------------------------------------------------------------
# Chapter 5: a chain of compiler-scheduled chips
# ---------------------------------------------------------------------------


def chain_steps(pid: str, mid: str, fmt: str, ctx: int, chips: int, group: int) -> list[dict[str, Any]]:
    """
    Chapter 5's frames: one token through the stages of a pipeline, on the
    static schedule. Each stage reads its share of the weights at its chips'
    SRAM bandwidth (and computes, if that takes longer), its group's per-layer
    all-reduces take one hop each, then the activation hops to the next stage.
    """
    d = decode(pid, mid, fmt, 1, ctx, chips, group)
    st = d["stages"]
    s = shape(mid)
    work = d["t_mem"] if d["t_comp"] is None or d["t_mem"] >= d["t_comp"] else d["t_comp"]
    stage_work = work / st
    hop = fval(pid, "hop_s") or 0.0
    # the stage's share of the per-layer all-reduces (layers spread evenly)
    stage_reduce = (2 * s["layers"] * hop if d["group"] > 1 else 0.0) / st
    out = []
    t = 0.0
    for k in range(st):
        start = t
        t = t + stage_work + stage_reduce
        done = t
        if k < st - 1:
            t = t + hop
        out.append({"i": k, "stage": k, "start": start, "work_end": start + stage_work, "done": done, "next": t})
    return out


# The chain animation's cases: part, model, format, chips (0 = fewest), chips per stage.
CHAIN_CASES: list[dict[str, Any]] = [
    {"key": "demo", "part": "groqchip", "model": "llama2-70b", "fmt": "int8", "chips": 576, "group": 8},
    {"key": "fewest-nodes", "part": "groqchip", "model": "llama2-70b", "fmt": "int8", "chips": 304, "group": 8},
    {"key": "fewest-chain", "part": "groqchip", "model": "llama2-70b", "fmt": "int8", "chips": 0, "group": 1},
    {"key": "groq3", "part": "groq3", "model": "llama3-70b", "fmt": "fp8", "chips": 0, "group": 1},
]
CHAIN_CTX = 1024


def lpu_capacity(pid: str, mid: str, fmt: str, ctx: int) -> dict[str, Any]:
    """Chips (and racks, for the LPX) to hold a model in SRAM."""
    pl = placement(pid, mid, fmt, 1, ctx)
    per_rack = fval(pid, "rack_chips")
    return {"chips": pl["chips"], "need": pl["need"], "racks": None if per_rack is None else math.ceil(pl["chips"] / per_rack)}


# ---------------------------------------------------------------------------
# Validation against published figures
# ---------------------------------------------------------------------------

GROQ_CTX = 1024  # the demonstration's 512-token prompts and 1,024-token outputs: mid-generation context
CEREBRAS_CTX = 1024  # not stated; a mid-length context (the KV cache is small beside 16-bit weights)
SN40L_CTX = 8192  # the SN40L paper's Table IV: 'Sequence length of 8K is used'


def validation() -> list[dict[str, Any]]:
    """
    Each check: what was published, what the model gives, their ratio
    (published / model), and the kind of check. Capacity checks compare chip
    counts; speed checks compare a published speed with the model's bound, so
    their ratio is the share of the bound the system achieved.
    """
    P = {k: f["v"] for k, f in PUBLISHED.items()}
    rows = []

    def add(cid: str, kind: str, label: str, pub: float, model: float, src: str) -> None:
        rows.append({"id": cid, "kind": kind, "label": label, "published": pub, "model": model, "ratio": pub / model, "src": src})

    cap70 = placement("wse-3", "llama3-70b", "bf16", 1, CEREBRAS_CTX)["chips"]
    add("cerebras-70b-wafers", "capacity", "Llama 3.1 70B, 16-bit: CS-3 systems needed", P["cerebras-70b-systems"], cap70, "cerebras-inference")
    # 20B parameters at 16 bits on one wafer: weights only
    add("cerebras-20b-wafer", "capacity", "a 20B model, 16-bit: CS-3 systems needed (weights only)", P["cerebras-20b-one"], math.ceil(20e9 * 2 / fval("wse-3", "sram_bytes")), "cerebras-inference")
    capg = placement("groqchip", "llama2-70b", "int8", 1, GROQ_CTX)["chips"]
    add("groq-70b-chips", "capacity", "Llama 2 70B, INT8: GroqChips used vs the fewest that hold it", P["groq-70b-chips"], capg, "nextplatform-groq")
    n_in, n_out = P["mlperf-input-mean"], P["mlperf-tokens-per-sample"]
    ceil_h100 = mlperf_ceiling("h100", "llama2-70b", "fp8", 8, n_in, n_out)["ceiling"]
    add("mlperf-h100", "throughput", "8x H100, Llama 2 70B, FP8, MLPerf Offline tokens/s vs the compute ceiling (prefill included)", P["mlperf-h100"], ceil_h100, "mlperf-h100")
    ceil_gb200 = mlperf_ceiling("b200", "llama2-70b", "fp4", P["mlperf-gb200-chips"], n_in, n_out)["ceiling"]
    add("mlperf-gb200", "throughput", "4x GB200 (Blackwell), Llama 2 70B, FP4, MLPerf Offline tokens/s vs the compute ceiling (prefill included)", P["mlperf-gb200"], ceil_gb200, "mlperf-gb200")
    g = decode("groqchip", "llama2-70b", "int8", 1, GROQ_CTX, P["groq-70b-chips"], fval("groqchip", "node_chips"))
    add("groq-70b-tps", "latency", "576 GroqChips, Llama 2 70B: tokens/s per user vs the bound", P["groq-70b-tps"], g["tok_s_user"], "nextplatform-groq")
    c70 = decode("wse-3", "llama3-70b", "bf16", 1, CEREBRAS_CTX)
    add("cerebras-70b-tps", "latency", "4x CS-3, Llama 3.1 70B: tokens/s per user vs the bound", P["cerebras-70b-tps"], c70["tok_s_user"], "cerebras-inference")
    c8 = decode("wse-3", "llama3-8b", "bf16", 1, CEREBRAS_CTX)
    add("cerebras-8b-tps", "latency", "1x CS-3, Llama 3.1 8B: tokens/s per user vs the bound", P["cerebras-8b-tps"], c8["tok_s_user"], "cerebras-inference")
    n = P["sn40l-sockets"]
    s8 = decode("sn40l", "llama3-8b", "bf16", 1, SN40L_CTX, n)
    add("sn40l-8b-tps", "latency", "16 SN40L sockets, Llama 3.1 8B, BF16: tokens/s per user vs the bound", P["sn40l-8b-tps"], s8["tok_s_user"], "sn40l")
    s70 = decode("sn40l", "llama3-70b", "bf16", 1, SN40L_CTX, n)
    add("sn40l-70b-tps", "speculative", "16 SN40L sockets, Llama 3.1 70B, BF16, speculative decoding: tokens/s per user vs the one-token-per-step bound", P["sn40l-70b-tps"], s70["tok_s_user"], "sn40l")
    return rows


OVERHEAD_MODEL = {"groq-70b-tps": "llama2-70b", "cerebras-70b-tps": "llama3-70b", "cerebras-8b-tps": "llama3-8b", "sn40l-8b-tps": "llama3-8b"}


def implied_overhead(check_id: str) -> dict[str, Any]:
    """For a speed check: the time per token the system took beyond the model's bound, per layer."""
    v = next(r for r in validation() if r["id"] == check_id)
    mid = OVERHEAD_MODEL[check_id]
    t_pub = 1 / v["published"]
    t_mod = 1 / v["model"]
    layers = MODELS[mid]["layers"]
    return {"t_published": t_pub, "t_model": t_mod, "extra": t_pub - t_mod, "per_layer": (t_pub - t_mod) / layers}


# ---------------------------------------------------------------------------
# The per-layer latency floor of the SRAM machines (chapters 4, 5 and 10)
# ---------------------------------------------------------------------------

FLOOR_CHECKS = ("groq-70b-tps", "cerebras-70b-tps", "cerebras-8b-tps", "sn40l-8b-tps")
FLOOR_LABEL = {
    "groq-70b-tps": "Groq, 576 chips, Llama 2 70B",
    "cerebras-70b-tps": "Cerebras, 4 CS-3, Llama 3.1 70B",
    "cerebras-8b-tps": "Cerebras, 1 CS-3, Llama 3.1 8B",
    "sn40l-8b-tps": "SambaNova, 16 SN40L, Llama 3.1 8B",
}


def floor_fit() -> dict[str, Any]:
    """
    A fixed time per layer, tau, added to the bound: t = t_bound + L tau.
    The fitted tau is the mean of the four systems' implied per-layer
    overheads. Each system is also predicted with tau fitted on the other
    three only (leave-one-out), so the test of the term is honest: the
    ratio published / predicted is reported for every system.
    """
    rows = []
    for cid in FLOOR_CHECKS:
        o = implied_overhead(cid)
        layers = MODELS[OVERHEAD_MODEL[cid]]["layers"]
        rows.append({"id": cid, "label": FLOOR_LABEL[cid], "layers": layers, **o})
    n = len(rows)
    # explicit loops: Python's sum() of floats is compensated, the port's is not
    total = 0.0
    for r in rows:
        total += r["per_layer"]
    tau = total / n
    for r in rows:
        o = 0.0
        k = 0
        for x in rows:
            if x["id"] != r["id"]:
                o += x["per_layer"]
                k += 1
        tau_loo = o / k
        t_pred = r["t_model"] + r["layers"] * tau_loo
        r["tauLoo"] = tau_loo
        r["tPred"] = t_pred
        r["ratioPred"] = r["t_published"] / t_pred  # time: published / predicted
        r["ratioBound"] = r["t_published"] / r["t_model"]
        r["floorShare"] = r["layers"] * tau_loo / t_pred
    lo = min(r["per_layer"] for r in rows)
    hi = max(r["per_layer"] for r in rows)
    return {"tau": tau, "min": lo, "max": hi, "rows": rows,
            "worstPred": max(max(r["ratioPred"], 1 / r["ratioPred"]) for r in rows),
            "worstBound": max(r["ratioBound"] for r in rows)}


def critical_path_steps(check_id: str, mode: str) -> list[dict[str, Any]]:
    """
    Chapter 10's animation: one token's critical path, layer by layer, for
    one system. Each layer costs its share of the bound (t_model / L: the
    weight reads or compute the bound counts) plus a floor: the leave-one-out
    tau ("fitted"), the system's own implied overhead ("own") or none.
    Step 0 is the token's start; step i ends layer i; the last step adds the
    published time for comparison.
    """
    fit = floor_fit()
    r = next(x for x in fit["rows"] if x["id"] == check_id)
    L = r["layers"]
    work = r["t_model"] / L
    floor = {"fitted": r["tauLoo"], "own": r["per_layer"], "none": 0.0}[mode]
    out = [{"i": 0, "layer": 0, "t": 0.0, "work": 0.0, "floor": 0.0, "layers": L}]
    t = 0.0
    w_acc = 0.0
    f_acc = 0.0
    for i in range(1, L + 1):
        t = t + work + floor
        w_acc = w_acc + work
        f_acc = f_acc + floor
        out.append({"i": i, "layer": i, "t": t, "work": w_acc, "floor": f_acc, "layers": L})
    return out


def band() -> dict[str, Any]:
    """The validation band, per kind of check."""
    out: dict[str, Any] = {}
    for kind in ("capacity", "throughput", "latency"):
        rs = [r["ratio"] for r in validation() if r["kind"] == kind]
        out[kind] = {"min": min(rs), "max": max(rs), "n": len(rs)}
    return out


def illustrative(key: str) -> float:
    return ILLUSTRATIVE[key]["v"]


# ---------------------------------------------------------------------------
# Chapter 6: reconfigurable dataflow (one decoder layer, kernel by kernel or fused)
# ---------------------------------------------------------------------------

ACT_BYTES = 2.0  # activations in 16-bit (the SN40L paper: BF16 weights, mixed BF16/FP32 activations)


def layer_ops(mid: str) -> list[dict[str, Any]]:
    """
    The operations of one decoder layer, in order: per token, the weights each
    multiplies (`w`, parameters) and the width of its output (`out`,
    elements). The attention scores' width depends on the context, so it is
    worked out in `layer_traffic`.
    """
    s = shape(mid)
    d, ffn = s["d"], s["ffn"]
    kvd = s["kv_heads"] * s["head_dim"]
    return [
        {"name": "RMSNorm", "kind": "vector", "w": 0, "out": d},
        {"name": "QKV", "kind": "matmul", "w": d * (d + 2 * kvd), "out": d + 2 * kvd},
        {"name": "scores QK^T", "kind": "scores", "w": 0, "out": 0},
        {"name": "softmax, x V", "kind": "values", "w": 0, "out": d},
        {"name": "attention out", "kind": "matmul", "w": d * d, "out": d},
        {"name": "residual add", "kind": "vector", "w": 0, "out": d},
        {"name": "RMSNorm", "kind": "vector", "w": 0, "out": d},
        {"name": "MLP up and gate", "kind": "matmul", "w": d * 2 * ffn, "out": 2 * ffn},
        {"name": "SiLU x gate", "kind": "vector", "w": 0, "out": ffn},
        {"name": "MLP down", "kind": "matmul", "w": ffn * d, "out": d},
        {"name": "residual add", "kind": "vector", "w": 0, "out": d},
    ]


def layer_traffic(pid: str, mid: str, fmt: str, tokens: int, seqs: int, ctx: int, causal: bool, fused: bool) -> dict[str, Any]:
    """
    One decoder layer on one chip: the bytes each operation moves to and from
    off-chip memory, its flops and its time. `tokens` rows go through the
    layer, from `seqs` sequences that each attend over `ctx` positions (half
    of them on average when `causal`, as in a prefill).

    Kernel by kernel (fused = False), each operation reads its input from
    memory and writes its output back, including the attention scores; the
    layer's time is the sum of each kernel's max(memory, compute). Fused into
    one dataflow pipeline (fused = True), the intermediate results stay in
    on-chip buffers: memory sees only the weights, the KV cache and the
    layer's input and output, and since the stages overlap, the layer's time
    is max(all its bytes / bandwidth, all its flops / peak).
    """
    s = shape(mid)
    b = FORMATS[fmt]
    kvb = kv_bytes_per_elem(fmt)
    d, h = s["d"], s["heads"]
    kvd = s["kv_heads"] * s["head_dim"]
    avg = ctx / 2 if causal else ctx
    bw = fval(pid, "off_bw")
    pk = peak(pid, fmt)["flops"]
    kv_read = seqs * ctx * kvd * kvb
    ops = layer_ops(mid)
    out = []
    prev = d
    tot_bytes = 0.0
    tot_flops = 0.0
    t_sum = 0.0
    for i, o in enumerate(ops):
        wbytes = o["w"] * b
        width = h * avg if o["kind"] == "scores" else o["out"]
        kv = 0.0
        flops = 0.0
        if o["kind"] == "matmul":
            flops = 2 * o["w"] * tokens
        if o["kind"] == "scores":
            kv = kv_read
            flops = 2 * tokens * d * avg
        if o["kind"] == "values":
            kv = kv_read
            flops = 2 * tokens * d * avg
        if o["name"] == "QKV":
            kv = kv + tokens * 2 * kvd * kvb
        if fused:
            act = 0.0
            if i == 0:
                act = act + tokens * d * ACT_BYTES
            if i == len(ops) - 1:
                act = act + tokens * d * ACT_BYTES
        else:
            # the scores op reads Q and K; the values op reads the scores
            act = (prev + width) * tokens * ACT_BYTES
        nbytes = wbytes + act + kv
        t_mem = nbytes / bw
        t_comp = flops / pk
        t = t_mem if t_mem >= t_comp else t_comp
        out.append({"i": i, "name": o["name"], "kind": o["kind"], "w_bytes": wbytes, "act_bytes": act, "kv_bytes": kv, "bytes": nbytes, "flops": flops, "t": t})
        tot_bytes = tot_bytes + nbytes
        tot_flops = tot_flops + flops
        t_sum = t_sum + t
        prev = width
    if fused:
        tm = tot_bytes / bw
        tc = tot_flops / pk
        t_layer = tm if tm >= tc else tc
    else:
        t_layer = t_sum
    return {"ops": out, "bytes": tot_bytes, "flops": tot_flops, "t": t_layer}


def dataflow_steps(pid: str, mid: str, fmt: str, tokens: int, seqs: int, ctx: int, causal: bool) -> list[dict[str, Any]]:
    """
    Chapter 6's frames: the layer's operations one at a time, with the bytes
    and time so far kernel by kernel and fused (fused time so far = the larger
    of its bytes and flops so far over bandwidth and peak).
    """
    u = layer_traffic(pid, mid, fmt, tokens, seqs, ctx, causal, False)
    f = layer_traffic(pid, mid, fmt, tokens, seqs, ctx, causal, True)
    bw = fval(pid, "off_bw")
    pk = peak(pid, fmt)["flops"]
    out = []
    ub = 0.0
    ut = 0.0
    fb = 0.0
    ff = 0.0
    for i in range(len(u["ops"])):
        uo = u["ops"][i]
        fo = f["ops"][i]
        ub = ub + uo["bytes"]
        ut = ut + uo["t"]
        fb = fb + fo["bytes"]
        ff = ff + fo["flops"]
        tm = fb / bw
        tc = ff / pk
        out.append({"i": i, "name": uo["name"], "kind": uo["kind"], "u_bytes": ub, "u_t": ut, "f_bytes": fb, "f_t": tm if tm >= tc else tc, "u_op": uo["bytes"], "f_op": fo["bytes"]})
    return out


def tiers(pid: str, mid: str, fmt: str, sockets: int) -> dict[str, Any]:
    """
    A tiered-memory machine of `sockets` chips: how many layers' weights its
    SRAM holds at once, how many copies of the model its HBM and its DDR hold,
    and how long one model takes to load from DDR into HBM.
    """
    s = shape(mid)
    b = FORMATS[fmt]
    w = s["params"] * b
    layer = s["per_layer"] * b
    sram = sockets * fval(pid, "sram_bytes")
    hbm = sockets * fval(pid, "off_bytes")
    ddr = sockets * fval(pid, "ddr_bytes")
    return {
        "weights": w,
        "sram": sram,
        "hbm": hbm,
        "ddr": ddr,
        "sram_layers": math.floor(sram / layer),
        "hbm_models": math.floor(hbm / w),
        "ddr_models": math.floor(ddr / w),
        "switch_s": w / (sockets * fval(pid, "ddr_bw")),
    }


# ---------------------------------------------------------------------------
# Chapter 7: a mesh of small cores, cards tied by Ethernet
# ---------------------------------------------------------------------------


def mesh_layer(pid: str, mid: str, fmt: str, batch: int, ctx: int, cards: int) -> dict[str, Any]:
    """
    One decoder layer split over `cards` mesh chips by tensor parallelism.
    Each card holds 1/cards of every weight matrix in its DRAM and spreads it
    over its cores; per decode step it streams its share in from DRAM
    (`t` = max(bytes / DRAM bandwidth, flops / peak)); after the attention
    output and the MLP down projection the cards all-reduce the activations
    over their Ethernet ports (a ring: 2 (cards - 1) / cards of the data, at
    the per-direction rate). Ethernet latency is not disclosed and not added.
    """
    s = shape(mid)
    b = FORMATS[fmt]
    kvb = kv_bytes_per_elem(fmt)
    kvd = s["kv_heads"] * s["head_dim"]
    cores = fval(pid, "units")
    core_sram = fval(pid, "sram_bytes") / cores
    bw = fval(pid, "off_bw")
    pk = peak(pid, fmt)["flops"]
    link = fval(pid, "c2c_bidir") / 2
    g = layer_gemms(mid)
    ar_bytes = 0.0 if cards == 1 else 2 * (cards - 1) / cards * batch * s["d"] * ACT_BYTES

    def gemm(x: dict[str, Any]) -> dict[str, Any]:
        wc = x["k"] * x["n"] * b / cards
        fl = 2 * x["k"] * x["n"] * batch / cards
        tm = wc / bw
        tc = fl / pk
        return {"name": x["name"], "kind": "matmul", "bytes": wc, "per_core": wc / cores, "fits_core": wc / cores <= core_sram, "flops": fl, "t": tm if tm >= tc else tc}

    kv = 2 * batch * ctx * kvd * kvb / cards
    afl = 2 * 2 * s["d"] * ctx * batch / cards
    am = kv / bw
    ac = afl / pk
    attn = {"name": "attention", "kind": "attention", "bytes": kv, "per_core": kv / cores, "fits_core": kv / cores <= core_sram, "flops": afl, "t": am if am >= ac else ac}
    ar = {"name": "all-reduce", "kind": "ethernet", "bytes": ar_bytes, "per_core": 0.0, "fits_core": True, "flops": 0.0, "t": ar_bytes / link}
    ops = [gemm(g[0]), attn, gemm(g[1]), dict(ar), gemm(g[2]), gemm(g[3]), dict(ar)]
    t = 0.0
    eth = 0.0
    dram = 0.0
    per_core = 0.0
    for i, o in enumerate(ops):
        o["i"] = i
        t = t + o["t"]
        if o["kind"] == "ethernet":
            eth = eth + o["t"]
        else:
            dram = dram + o["t"]
        if o["kind"] == "matmul":
            per_core = per_core + o["per_core"]
    return {
        "ops": ops,
        "t": t,
        "t_dram": dram,
        "t_eth": eth,
        "core_sram": core_sram,
        "cores": cores,
        "cards": cards,
        "layer_per_core": per_core,
        "layer_fits": per_core <= core_sram,
        "tok_s_user": 1 / (t * s["layers"]),
    }


def mesh_steps(pid: str, mid: str, fmt: str, batch: int, ctx: int, cards: int) -> list[dict[str, Any]]:
    """Chapter 7's frames: the layer's steps in order, with the time so far."""
    m = mesh_layer(pid, mid, fmt, batch, ctx, cards)
    out = []
    t = 0.0
    for o in m["ops"]:
        t = t + o["t"]
        out.append({"i": o["i"], "name": o["name"], "kind": o["kind"], "bytes": o["bytes"], "per_core": o["per_core"], "fits_core": o["fits_core"], "t_op": o["t"], "t": t})
    return out


# ---------------------------------------------------------------------------
# Chapter 8: FPGAs and edge NPUs (a bandwidth budget, custom numerics)
# ---------------------------------------------------------------------------

BITS = (16, 8, 6, 5, 4, 3, 2)
EDGE_DEVICES = ("m4", "v70", "v80")


def anyval(pid: str, key: str) -> Any:
    """A field of a modelled part or of a quoted one (OTHER_PARTS)."""
    rec = PARTS[pid] if pid in PARTS else OTHER_PARTS[pid]
    f = rec["fields"].get(key)
    return None if f is None else f["v"]


def edge_bound(pid: str, mid: str, bits: int, ctx: int) -> dict[str, Any]:
    """
    One decode step at batch 1 with weights stored at `bits` bits each (a
    custom width an FPGA can implement; an NPU runs the formats its toolchain
    supports): every weight and the KV cache read once from the device's
    memory, against its peak (the 8-bit rate for 8 bits or fewer, the 16-bit
    rate above). Fit and energy only where the capacity and power are given.
    """
    s = shape(mid)
    wb = bits / 8
    kvb = 1.0 if wb < 1 else wb
    weights = s["params"] * wb
    nbytes = s["matmul_params"] * wb + ctx * s["kv_elems_per_token"] * kvb
    bw = anyval(pid, "off_bw")
    t_mem = nbytes / bw
    flops = 2 * s["matmul_params"] + s["attn_flops_per_pos"] * ctx
    pk = anyval(pid, "peak_int8") if bits <= 8 else anyval(pid, "peak_bf16")
    t_comp = None if pk is None else flops / pk
    t = t_mem if t_comp is None or t_mem >= t_comp else t_comp
    cap = anyval(pid, "off_bytes")
    pw = anyval(pid, "power_w")
    return {
        "bits": bits,
        "weights": weights,
        "bytes": nbytes,
        "t_mem": t_mem,
        "t_comp": t_comp,
        "t": t,
        "tok_s": 1 / t,
        "fits": None if cap is None else weights + ctx * s["kv_elems_per_token"] * kvb <= cap,
        "j_per_token": None if pw is None else pw * t,
    }


def edge_steps(pid: str, mid: str, ctx: int) -> list[dict[str, Any]]:
    """Chapter 8's frames: the same decode step at narrower and narrower weights."""
    out = []
    for i, bits in enumerate(BITS):
        e = edge_bound(pid, mid, bits, ctx)
        e["i"] = i
        out.append(e)
    return out


# ---------------------------------------------------------------------------
# Chapter 9: analogue in-memory compute (a decoder layer on PCM crossbars)
# ---------------------------------------------------------------------------


def ival(key: str) -> Any:
    return IMC["hermes"]["fields"][key]["v"]


def imc_layer(mid: str, phase: int) -> dict[str, Any]:
    """
    A decoder model mapped onto the HERMES chip's 256 x 256 crossbar cores:
    every weight matrix cut into crossbar tiles (weights stay put), the cores
    and chips that takes, and one token's time and energy if the four
    matrix-vector stages of every layer (QKV, attention out, MLP up and gate,
    MLP down) each take one MVM latency and every tile does one MVM per token.
    Attention over the KV cache is not an in-memory operation (the cache
    changes every token) and the chip-to-chip transfers have no disclosed
    figures, so both are left out: the time is the crossbars' share only.
    `phase` 1 = the low-precision read mode, 4 = the high-precision one.
    """
    s = shape(mid)
    rows = ival("rows")
    cols = ival("cols")
    cores = ival("cores")
    lat = ival("lat_1p") if phase == 1 else ival("lat_4p")
    e_core = (ival("e_1p") if phase == 1 else ival("e_4p")) / cores
    gemms = []
    layer_tiles = 0
    psums = 0
    for g in layer_gemms(mid):
        tk = math.ceil(g["k"] / rows)
        tn = math.ceil(g["n"] / cols)
        tiles = tk * tn
        layer_tiles += tiles
        psums += (tk - 1) * g["n"]
        gemms.append({"name": g["name"], "k": g["k"], "n": g["n"], "tiles": tiles, "util": g["k"] * g["n"] / (tiles * rows * cols)})
    head = math.ceil(s["d"] / rows) * math.ceil(s["vocab"] / cols)
    tiles = s["layers"] * layer_tiles + head
    chips = math.ceil(tiles / cores)
    t_layer = 4 * lat
    t_token = s["layers"] * t_layer + lat
    return {
        "gemms": gemms,
        "layer_tiles": layer_tiles,
        "layer_chips": layer_tiles / cores,
        "tiles": tiles,
        "chips": chips,
        "psums": psums,
        "lat": lat,
        "t_layer": t_layer,
        "t_token": t_token,
        "e_token": tiles * e_core,
        "weights": s["matmul_params"],
        "capacity": chips * ival("weights"),
    }


def imc_steps(mid: str, phase: int) -> list[dict[str, Any]]:
    """
    Chapter 9's frames: one layer for one token. Frame 0 is the weights
    programmed into the crossbars (once, before inference); then QKV, the
    attention (digital, off the crossbars), and the three other matrix
    stages, with the crossbar time and energy so far.
    """
    m = imc_layer(mid, phase)
    e_core = (ival("e_1p") if phase == 1 else ival("e_4p")) / ival("cores")
    order = [("program", -1), ("QKV", 0), ("attention", -1), ("attention out", 1), ("MLP up and gate", 2), ("MLP down", 3)]
    out = []
    t = 0.0
    e = 0.0
    for i, (name, gi) in enumerate(order):
        tiles = 0
        if gi >= 0:
            tiles = m["gemms"][gi]["tiles"]
            t = t + m["lat"]
            e = e + tiles * e_core
        out.append({"i": i, "name": name, "gemm": gi, "tiles": tiles, "t": t, "e": e})
    return out


def adc_bits(in_bits: int, w_bits: int, rows: int) -> int:
    """Bits an exact column sum needs: input bits + weight bits + log2(rows), against the ADC's output."""
    return in_bits + w_bits + math.ceil(math.log2(rows))


# ---------------------------------------------------------------------------
# Chapter 10: every family on the same decoder model
# ---------------------------------------------------------------------------

# The representative part and format each family runs in chapter 10.
CHOICE: dict[str, tuple[str, str]] = {
    "gpu": ("h100", "fp8"),
    "systolic": ("tpu-7x", "fp8"),
    "wafer": ("wse-3", "bf16"),
    "lpu": ("groqchip", "int8"),
    "dataflow": ("sn40l", "bf16"),
    "mesh": ("blackhole", "fp8"),
    "fpga-npu": ("m4", "int8"),
    "imc": ("hermes", "int8"),
}


def choose(mid: str, batch: int, ctx: int, prompt: int) -> list[dict[str, Any]]:
    """
    One row per family: the fewest chips that hold the model, one decode
    step's bound (and per layer), a prefill's bound, energy per token where
    power is given, and the share of the chips' peak a decode step uses.
    """
    s = shape(mid)
    out = []
    for i, f in enumerate(FAMILIES):
        pid, fmt = CHOICE[f["id"]]
        if pid == "hermes":
            m = imc_layer(mid, 1)
            ops = 2 * s["matmul_params"]
            t_pf = (prompt + 4 * s["layers"]) * m["lat"]
            out.append({
                "i": i, "family": f["id"], "name": f["name"], "part": pid, "fmt": fmt,
                "chips": m["chips"], "where": "in-array",
                "t_decode": m["t_token"], "t_layer": m["t_layer"], "tok_s_user": 1 / m["t_token"],
                "t_prefill": t_pf, "j_per_token": m["e_token"],
                "util": ops / (m["t_token"] * m["chips"] * ival("tops_1p")),
                "bound": "crossbar-only", "basis": "dense",
            })
            continue
        d = decode(pid, mid, fmt, batch, ctx)
        pf = prefill(pid, mid, fmt, prompt)
        pk = peak(pid, fmt)
        out.append({
            "i": i, "family": f["id"], "name": f["name"], "part": pid, "fmt": fmt,
            "chips": d["chips"], "where": d["where"],
            "t_decode": d["t_step"], "t_layer": d["t_step"] / s["layers"], "tok_s_user": d["tok_s_user"],
            "t_prefill": pf["t"], "j_per_token": d["j_per_token"],
            "util": None if pk["flops"] is None else d["flops"] / (d["t_step"] * d["chips"] * pk["flops"]),
            "bound": d["bound"], "basis": pk["basis"],
        })
    return out
