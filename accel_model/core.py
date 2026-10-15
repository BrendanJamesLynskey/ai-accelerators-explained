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

from .data import FAMILIES, FORMATS, ILLUSTRATIVE, MODELS, PARTS, PUBLISHED

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


# ---------------------------------------------------------------------------
# Chapter 1: the map
# ---------------------------------------------------------------------------

# The representative part and format each family's map step quotes.
MAP_PARTS: dict[str, tuple[str, str]] = {
    "gpu": ("h100", "fp8"),
    "systolic": ("tpu-7x", "fp8"),
    "wafer": ("wse-3", "bf16"),
    "lpu": ("groqchip", "int8"),
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
    ceil_h100 = throughput_ceiling("h100", "llama2-70b", "fp8", 8)
    add("mlperf-h100", "throughput", "8x H100, Llama 2 70B, FP8, MLPerf Offline tokens/s vs the compute ceiling", P["mlperf-h100"], ceil_h100, "mlperf-h100")
    g = decode("groqchip", "llama2-70b", "int8", 1, GROQ_CTX, P["groq-70b-chips"], fval("groqchip", "node_chips"))
    add("groq-70b-tps", "latency", "576 GroqChips, Llama 2 70B: tokens/s per user vs the bound", P["groq-70b-tps"], g["tok_s_user"], "nextplatform-groq")
    c70 = decode("wse-3", "llama3-70b", "bf16", 1, CEREBRAS_CTX)
    add("cerebras-70b-tps", "latency", "4x CS-3, Llama 3.1 70B: tokens/s per user vs the bound", P["cerebras-70b-tps"], c70["tok_s_user"], "cerebras-inference")
    c8 = decode("wse-3", "llama3-8b", "bf16", 1, CEREBRAS_CTX)
    add("cerebras-8b-tps", "latency", "1x CS-3, Llama 3.1 8B: tokens/s per user vs the bound", P["cerebras-8b-tps"], c8["tok_s_user"], "cerebras-inference")
    return rows


def implied_overhead(check_id: str) -> dict[str, Any]:
    """For a speed check: the time per token the system took beyond the model's bound, per layer."""
    v = next(r for r in validation() if r["id"] == check_id)
    mid = {"groq-70b-tps": "llama2-70b", "cerebras-70b-tps": "llama3-70b", "cerebras-8b-tps": "llama3-8b"}[check_id]
    t_pub = 1 / v["published"]
    t_mod = 1 / v["model"]
    layers = MODELS[mid]["layers"]
    return {"t_published": t_pub, "t_model": t_mod, "extra": t_pub - t_mod, "per_layer": (t_pub - t_mod) / layers}


def band() -> dict[str, Any]:
    """The validation band, per kind of check."""
    out: dict[str, Any] = {}
    for kind in ("capacity", "throughput", "latency"):
        rs = [r["ratio"] for r in validation() if r["kind"] == kind]
        out[kind] = {"min": min(rs), "max": max(rs), "n": len(rs)}
    return out


def illustrative(key: str) -> float:
    return ILLUSTRATIVE[key]["v"]
