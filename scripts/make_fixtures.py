"""
Writes the site's model data, the parity fixtures and results.md from the
Python reference (accel_model/):

  src/data/accel.json                 the sourced specifications, families,
                                      claims and model inputs the site shows
  tests/fixtures/accel_fixtures.json  reference results the TypeScript port
                                      must reproduce (tests/unit/model.test.ts)
  results.md                          every figure with its source and access
                                      date, the numbers the chapters quote,
                                      and the validation band

    python3 scripts/make_fixtures.py          # write all three
    python3 scripts/make_fixtures.py --check  # fail if any is out of date (CI)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import accel_model as a  # noqa: E402

FMTS = tuple(a.FORMATS)
# the grids the fixtures cover (the widgets' controls stay inside them)
BATCH = (1, 16, 256)
CTX = (1024, 4096)
SWEEP_PARTS = ("h100", "b200")
SWEEP_CASES = [(m, f) for m in ("llama3-8b", "llama3-70b") for f in ("bf16", "fp8")]
SWEEP_CTX = (1024, 8192)
SWEEP_CHIPS = (1, 8)
ARRAYS = ((128, 128), (256, 256))
MXU_M = (1, 16, 128, 256, 512)
MXU_K = 1024
MXU_N = 1024
D0 = (0.05, 0.1, 0.2)
SPARE = (0.005, 0.015)
SEEDS = (7, 11)
DEFECT_COUNTS = (23, 46, 92)
PER_STEP = (1, 2, 4)
# (tokens, sequences, context, causal): a decode step at batch 1 and 64, and a 2,048-token prefill
DATAFLOW_CASES = ((1, 1, 8192, False), (64, 64, 1024, False), (2048, 1, 2048, True))
MESH_CARDS = (1, 2, 3, 4, 8)
OVERHEAD_IDS = ("groq-70b-tps", "cerebras-70b-tps", "cerebras-8b-tps", "sn40l-8b-tps")


def site_data() -> dict:
    return {
        "accessed": a.ACCESSED,
        "sources": a.SOURCES,
        "families": a.FAMILIES,
        "placementLabels": list(a.PLACEMENT_LABELS),
        "scheduleLabels": list(a.SCHEDULE_LABELS),
        "grainLabels": list(a.GRAIN_LABELS),
        "parts": [a.PARTS[p] for p in a.PART_ORDER],
        "models": [a.MODELS[m] for m in a.MODEL_ORDER],
        "formats": a.FORMATS,
        "published": a.PUBLISHED,
        "claims": a.CLAIMS,
        "illustrative": a.ILLUSTRATIVE,
        "mapParts": {k: list(v) for k, v in a.MAP_PARTS.items()},
        "batches": list(a.BATCHES),
        "dieCols": a.DIE_COLS,
        "dieRows": a.DIE_ROWS,
        "chainCases": a.CHAIN_CASES,
        "chainCtx": a.CHAIN_CTX,
        "groqCtx": a.GROQ_CTX,
        "cerebrasCtx": a.CEREBRAS_CTX,
        "sn40lCtx": a.SN40L_CTX,
        "otherParts": [a.OTHER_PARTS[p] for p in a.OTHER_ORDER],
        "imc": a.IMC,
        "choice": {k: list(v) for k, v in a.CHOICE.items()},
        "bits": list(a.BITS),
        "edgeDevices": list(a.EDGE_DEVICES),
        "actBytes": a.ACT_BYTES,
    }


def fixtures() -> dict:
    out: dict = {}
    out["parts"] = {p: a.part(p) for p in a.PART_ORDER}
    out["shapes"] = {m: a.shape(m) for m in a.MODEL_ORDER}
    out["peaks"] = [{"part": p, "fmt": f, "out": a.peak(p, f)} for p in a.PART_ORDER for f in FMTS]
    out["placement"] = [
        {"part": p, "model": m, "fmt": f, "batch": b, "ctx": c, "out": a.placement(p, m, f, b, c)}
        for p in a.PART_ORDER
        for m in a.MODEL_ORDER
        for f in FMTS
        for b in BATCH
        for c in CTX
    ]
    dec = []
    for p in a.PART_ORDER:
        groups = (1,) if a.PARTS[p]["layout"] == "tensor" else (1, 8)
        for m in a.MODEL_ORDER:
            for f in FMTS:
                for b in BATCH:
                    for c in CTX:
                        for n in (0, 8, 512):
                            for g in groups:
                                dec.append({"part": p, "model": m, "fmt": f, "batch": b, "ctx": c, "chips": n, "group": g, "out": a.decode(p, m, f, b, c, n, g)})
    out["decode"] = dec
    out["prefill"] = [
        {"part": p, "model": m, "fmt": f, "prompt": q, "chips": n, "out": a.prefill(p, m, f, q, n)}
        for p in a.PART_ORDER
        for m in a.MODEL_ORDER
        for f in FMTS
        for q in (128, 2048)
        for n in (0, 8)
    ]
    out["ridge"] = [{"part": p, "fmt": f, "out": a.ridge(p, f)} for p in a.PART_ORDER for f in FMTS]
    out["ceiling"] = [
        {"part": p, "model": m, "fmt": f, "chips": n, "out": a.throughput_ceiling(p, m, f, n)}
        for p in ("h100", "b200", "tpu-7x", "trn2")
        for m in a.MODEL_ORDER
        for f in ("bf16", "fp8")
        for n in (1, 8)
    ]
    out["familyMap"] = a.family_map()
    out["mapSteps"] = [{"model": m, "batch": 1, "ctx": c, "out": a.map_steps(m, 1, c)} for m in a.MODEL_ORDER for c in CTX]
    out["batchSweep"] = [
        {"part": p, "model": m, "fmt": f, "ctx": c, "chips": n, "out": a.batch_sweep(p, m, f, c, n)}
        for p in SWEEP_PARTS
        for m, f in SWEEP_CASES
        for c in SWEEP_CTX
        for n in SWEEP_CHIPS
    ]
    out["maxBatch"] = [
        {"part": p, "model": m, "fmt": f, "ctx": c, "chips": n, "out": a.max_batch(p, m, f, c, n)}
        for p in SWEEP_PARTS
        for m, f in SWEEP_CASES
        for c in SWEEP_CTX
        for n in SWEEP_CHIPS
    ]
    out["layerGemms"] = {m: a.layer_gemms(m) for m in a.MODEL_ORDER}
    out["mxu"] = [
        {"rows": r, "cols": c, "m": m, "k": k, "n": n, "out": a.mxu(r, c, m, k, n)}
        for r, c in ARRAYS
        for m in MXU_M
        for k, n in ((MXU_K, MXU_N), (4096, 6144), (28672, 8192), (1000, 300))
    ]
    out["layerMxu"] = [{"rows": r, "cols": c, "model": md, "m": m, "out": a.layer_mxu(r, c, md, m)} for r, c in ARRAYS for md in a.MODEL_ORDER for m in MXU_M]
    out["mxuSteps"] = [{"rows": r, "cols": c, "m": m, "out": a.mxu_steps(r, c, m, MXU_K, MXU_N)} for r, c in ARRAYS for m in MXU_M]
    out["waferYield"] = [
        {"part": p, "d0": d, "die": a.illustrative("gpu_die_mm2"), "spare": s, "out": a.wafer_yield(p, d, a.illustrative("gpu_die_mm2"), s)}
        for p in ("wse-3", "wse-3t")
        for d in D0
        for s in SPARE
    ]
    out["defects"] = [{"seed": s, "count": n, "out": a.defects(s, n)} for s in SEEDS for n in DEFECT_COUNTS]
    out["defectSteps"] = [{"seed": s, "count": n, "per": k, "out": a.defect_steps(s, n, k)} for s in SEEDS for n in DEFECT_COUNTS for k in PER_STEP]
    out["wafersNeeded"] = [
        {"part": p, "model": m, "fmt": f, "ctx": c, "out": a.wafers_needed(p, m, f, 1, c)}
        for p in ("wse-3", "wse-3t")
        for m in a.MODEL_ORDER
        for f in ("bf16", "fp8")
        for c in CTX
    ]
    out["chainSteps"] = [{"case": cc, "out": a.chain_steps(cc["part"], cc["model"], cc["fmt"], a.CHAIN_CTX, cc["chips"], cc["group"])} for cc in a.CHAIN_CASES]
    out["lpuCapacity"] = [
        {"part": p, "model": m, "fmt": f, "ctx": c, "out": a.lpu_capacity(p, m, f, c)}
        for p in ("groqchip", "groq3")
        for m in a.MODEL_ORDER
        for f in FMTS
        for c in CTX
    ]
    out["layerOps"] = {m: a.layer_ops(m) for m in a.MODEL_ORDER}
    out["layerTraffic"] = [
        {"part": p, "model": m, "fmt": "bf16", "tokens": t, "seqs": q, "ctx": c, "causal": cz, "fused": fu, "out": a.layer_traffic(p, m, "bf16", t, q, c, cz, fu)}
        for p in ("sn40l",)
        for m in a.MODEL_ORDER
        for (t, q, c, cz) in DATAFLOW_CASES
        for fu in (False, True)
    ]
    out["dataflowSteps"] = [
        {"part": "sn40l", "model": m, "fmt": "bf16", "tokens": t, "seqs": q, "ctx": c, "causal": cz, "out": a.dataflow_steps("sn40l", m, "bf16", t, q, c, cz)}
        for m in a.MODEL_ORDER
        for (t, q, c, cz) in DATAFLOW_CASES
    ]
    out["tiers"] = [{"part": "sn40l", "model": m, "fmt": f, "sockets": n, "out": a.tiers("sn40l", m, f, n)} for m in a.MODEL_ORDER for f in ("bf16",) for n in (1, 8, 16)]
    out["meshLayer"] = [
        {"part": "blackhole", "model": m, "fmt": "fp8", "batch": b, "ctx": c, "cards": n, "out": a.mesh_layer("blackhole", m, "fp8", b, c, n)}
        for m in a.MODEL_ORDER
        for b in (1, 32)
        for c in CTX
        for n in MESH_CARDS
    ]
    out["meshSteps"] = [
        {"part": "blackhole", "model": m, "fmt": "fp8", "batch": b, "ctx": 1024, "cards": n, "out": a.mesh_steps("blackhole", m, "fp8", b, 1024, n)}
        for m in a.MODEL_ORDER
        for b in (1, 32)
        for n in MESH_CARDS
    ]
    out["edgeBound"] = [
        {"part": p, "model": m, "bits": bt, "ctx": c, "out": a.edge_bound(p, m, bt, c)}
        for p in a.EDGE_DEVICES
        for m in a.MODEL_ORDER
        for bt in a.BITS
        for c in CTX
    ]
    out["edgeSteps"] = [{"part": p, "model": m, "ctx": c, "out": a.edge_steps(p, m, c)} for p in a.EDGE_DEVICES for m in a.MODEL_ORDER for c in CTX]
    out["imcLayer"] = [{"model": m, "phase": ph, "out": a.imc_layer(m, ph)} for m in a.MODEL_ORDER for ph in (1, 4)]
    out["imcSteps"] = [{"model": m, "phase": ph, "out": a.imc_steps(m, ph)} for m in a.MODEL_ORDER for ph in (1, 4)]
    out["adcBits"] = [{"in": i, "w": w, "rows": r, "out": a.adc_bits(i, w, r)} for i in (4, 8) for w in (4, 8) for r in (64, 256, 512)]
    out["choose"] = [
        {"model": m, "batch": b, "ctx": c, "prompt": q, "out": a.choose(m, b, c, q)}
        for m in ("llama3-8b", "llama3-70b")
        for b in (1, 64)
        for c in CTX
        for q in (2048,)
    ]
    out["validation"] = a.validation()
    out["band"] = a.band()
    out["overhead"] = {k: a.implied_overhead(k) for k in OVERHEAD_IDS}
    return out


# ---------------------------------------------------------------------------
# results.md
# ---------------------------------------------------------------------------


def _num(v) -> str:
    if v is None:
        return "not disclosed"
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if v == int(v) and abs(v) < 1e6:
        return f"{int(v):,}"
    return f"{v:.6g}"


def _src(sid: str) -> str:
    return f"[{sid}]({a.SOURCES[sid]['url']})"


def _rows(fields: dict) -> list[str]:
    out = []
    for k, f in fields.items():
        out.append(f"| `{k}` | {_num(f['v'])} | {f['st']} | {_src(f['src'])} | {f['ref'].replace('|', '/')} |")
    return out


def _t(x) -> str:
    return "not disclosed" if x is None else f"{x:.4g}"


def results_md() -> str:
    L: list[str] = []
    w = L.append
    w("# Results and sources")
    w("")
    w(
        "Generated by `python3 scripts/make_fixtures.py` from `accel_model/`; do not edit by hand. "
        f"Every source below was read on **{a.ACCESSED}**. Statuses: `spec` (the vendor or the paper states it), "
        "`derived` (computed from sourced figures; the formula is in `where`), `reported` (a third party reports it), "
        "`illustrative` (a model parameter, labelled on the site), `undisclosed` (not disclosed in the sources used). "
        "Availability: `product` (shipping or generally available), `announced` (announced; shipping not yet stated, or "
        "only starting), `retired` (superseded; kept because published results used it). Bandwidths are per direction "
        "unless a field says `bidir`; peak rates are dense unless a field says `sparse`."
    )
    w("")
    w("## Sources")
    w("")
    w("| id | source | kind | accessed |")
    w("| --- | --- | --- | --- |")
    for sid, s in a.SOURCES.items():
        w(f"| `{sid}` | [{s['title']}]({s['url']}) | {s['kind']} | {s['accessed']} |")
    w("")
    w("## The families (chapter 1's map)")
    w("")
    w("| family | chapter | example | weights live in | who schedules | granularity |")
    w("| --- | --- | --- | --- | --- | --- |")
    for f in a.FAMILIES:
        def cell(k: str, labels) -> str:
            x = f[k]
            return f"{labels[x['v']]} ({x['ref']}; {_src(x['src'])})"
        w(
            f"| {f['name']} | {f['chapter']} | {f['example']} | {cell('placement', a.PLACEMENT_LABELS)} | "
            f"{cell('schedule', a.SCHEDULE_LABELS)} | {cell('grain', a.GRAIN_LABELS)} |"
        )
    w("")
    w("## Parts")
    for pid in a.PART_ORDER:
        p = a.PARTS[pid]
        w("")
        w(f"### {p['name']} (`{pid}`), family `{p['family']}`, layout `{p['layout']}`, availability: {p['avail']['v']}")
        w("")
        w(f"Availability: {p['avail']['ref']} ({_src(p['avail']['src'])}).")
        w("")
        w("| field | value | status | source | where |")
        w("| --- | --- | --- | --- | --- |")
        L.extend(_rows(p["fields"]))
    w("")
    w("## Parts quoted, not modelled")
    w("")
    w("Each lacks a figure the model needs (a capacity or a bandwidth), so the site quotes them and runs nothing on them.")
    for pid in a.OTHER_ORDER:
        p = a.OTHER_PARTS[pid]
        w("")
        w(f"### {p['name']} (`{pid}`), family `{p['family']}`, availability: {p['avail']['v']}")
        w("")
        w(f"Availability: {p['avail']['ref']} ({_src(p['avail']['src'])}).")
        w("")
        w("| field | value | status | source | where |")
        w("| --- | --- | --- | --- | --- |")
        L.extend(_rows(p["fields"]))
    w("")
    w("## In-memory compute chip (chapter 9)")
    for pid, p in a.IMC.items():
        w("")
        w(f"### {p['name']} (`{pid}`), availability: {p['avail']['v']}")
        w("")
        w(f"Availability: {p['avail']['ref']} ({_src(p['avail']['src'])}).")
        w("")
        w("| field | value | status | source | where |")
        w("| --- | --- | --- | --- | --- |")
        L.extend(_rows(p["fields"]))
    w("")
    w("## The running workload")
    w("")
    w("| model | layers | d | FFN | heads | KV heads | vocabulary | parameters | matmul parameters | KV elements per token | source |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for m in a.MODEL_ORDER:
        s = a.shape(m)
        w(
            f"| {s['name']} | {s['layers']} | {s['d']:,} | {s['ffn']:,} | {s['heads']} | {s['kv_heads']} | {s['vocab']:,} | "
            f"{s['params']:,} | {s['matmul_params']:,} | {s['kv_elems_per_token']:,} | {_src(a.MODELS[m]['src'])} |"
        )
    w("")
    w("Llama 2's paper gives no layer table; its 70B uses the Llama 3 70B layer shape with Llama 2's 32k vocabulary (68.98 B parameters).")
    w("")
    w("## Published figures (validation targets, never inputs)")
    w("")
    w("| id | value | status | source | where |")
    w("| --- | --- | --- | --- | --- |")
    L.extend(_rows(a.PUBLISHED))
    w("")
    w("## Quoted claims (shown, attributed, never model inputs)")
    w("")
    for cid, c in a.CLAIMS.items():
        w(f"- `{cid}`: {c['text']} ({_src(c['src'])})")
    w("")
    w("## Illustrative parameters")
    w("")
    w("| id | value | status | where |")
    w("| --- | --- | --- | --- |")
    for k, f in a.ILLUSTRATIVE.items():
        w(f"| `{k}` | {_num(f['v'])} | {f['st']} | {f['ref']} |")
    w("")
    w("## Validation band")
    w("")
    w(
        "Ratio = published / model. **Capacity** checks compare chip counts (the model gives the fewest chips whose memory "
        "holds the weights and KV cache). **Throughput** and **latency** checks compare a published speed with the model's "
        "bound, so the ratio is the share of the bound the system reached."
    )
    w("")
    w("| check | kind | published | model | ratio | source |")
    w("| --- | --- | --- | --- | --- | --- |")
    for v in a.validation():
        w(f"| {v['label']} | {v['kind']} | {_num(v['published'])} | {v['model']:.6g} | {v['ratio']:.4g} | {_src(v['src'])} |")
    w("")
    b = a.band()
    for k in ("capacity", "throughput", "latency"):
        w(f"- {k}: {b[k]['n']} check(s), ratio {b[k]['min']:.4g} to {b[k]['max']:.4g}")
    w("")
    w("What the latency bound leaves out, per layer (published time per token minus the model's bound, over the layers):")
    w("")
    w("| check | published (s/token) | bound (s/token) | extra per layer (s) |")
    w("| --- | --- | --- | --- |")
    for k in OVERHEAD_IDS:
        o = a.implied_overhead(k)
        w(f"| `{k}` | {o['t_published']:.4g} | {o['t_model']:.4g} | {o['per_layer']:.4g} |")
    w("")
    w("## Numbers the chapters use")
    w("")
    w("### Chapter 1: where Llama 3.1 70B's weights live (batch 1, 1,024-token context)")
    w("")
    w("| family | part | format | weights + KV (B) | in | per chip (B) | chips |")
    w("| --- | --- | --- | --- | --- | --- | --- |")
    for s in a.map_steps("llama3-70b", 1, 1024):
        if s["part"]:
            w(f"| {s['name']} | {a.PARTS[s['part']]['name']} | {s['fmt']} | {_num(s['need'])} | {s['where']} | {_num(s['cap'])} | {s['chips']} |")
    w("")
    w("### Decode bounds, batch 1, 4,096-token context, fewest chips")
    w("")
    w("| part | model | format | chips | memory term (s) | compute term (s) | hops (s) | step (s) | bound | tokens/s per user | J/token | peak basis |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for p in a.PART_ORDER:
        for m in ("llama3-8b", "llama3-70b"):
            for f in FMTS:
                if a.peak(p, f)["flops"] is None:
                    continue
                d = a.decode(p, m, f, 1, 4096)
                w(
                    f"| {a.PARTS[p]['name']} | {m} | {f} | {d['chips']} | {d['t_mem']:.4g} | {_t(d['t_comp'])} | {d['t_hops']:.4g} | "
                    f"{d['t_step']:.4g} | {d['bound']} | {d['tok_s_user']:.4g} | {_t(d['j_per_token'])} | {d['basis']} |"
                )
    w("")
    w("### Chapter 2: the batch sweep (Llama 3.1 70B, FP8, 1,024-token context)")
    w("")
    for p, n in (("h100", 8), ("b200", 8)):
        w(f"{a.PARTS[p]['name']} x {n}; ridge {a.ridge(p, 'fp8'):.4g} flop/B; largest batch that fits {a.max_batch(p, 'llama3-70b', 'fp8', 1024, n):,}:")
        w("")
        w("| batch | fits | intensity (flop/B) | memory term (s) | compute term (s) | bound | tokens/s | tokens/s per user | J/token |")
        w("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for s in a.batch_sweep(p, "llama3-70b", "fp8", 1024, n):
            w(
                f"| {s['batch']} | {_num(s['fits'])} | {s['intensity']:.4g} | {s['t_mem']:.4g} | {_t(s['t_comp'])} | {s['bound']} | "
                f"{s['tok_s']:.4g} | {s['tok_s_user']:.4g} | {_t(s['j_per_token'])} |"
            )
        w("")
    w("Prefill of a 2,048-token prompt (fewest chips):")
    w("")
    w("| part | model | format | chips | time (s) | bound | prompt tokens/s |")
    w("| --- | --- | --- | --- | --- | --- | --- |")
    for p in ("h100", "b200", "tpu-7x", "trn2", "wse-3", "groqchip"):
        for m, f in (("llama3-8b", "bf16"), ("llama3-70b", "fp8"), ("llama3-70b", "bf16")):
            pf = a.prefill(p, m, f, 2048)
            if pf["t_comp"] is None:
                continue
            w(f"| {a.PARTS[p]['name']} | {m} | {f} | {pf['chips']} | {pf['t']:.4g} | {pf['bound']} ({pf['basis']}) | {pf['tok_s']:.4g} |")
    w("")
    w("### Chapter 3: one decoder layer on one systolic array")
    w("")
    w("| array | model | batch | cycles | utilisation |")
    w("| --- | --- | --- | --- | --- |")
    for r, c in ARRAYS:
        for m in MXU_M:
            x = a.layer_mxu(r, c, "llama3-70b", m)
            w(f"| {r} x {c} | Llama 3.1 70B | {m} | {x['cycles']:,} | {x['util']:.4g} |")
    w("")
    w("### Chapter 4: wafer-scale")
    w("")
    y = a.wafer_yield("wse-3", a.illustrative("d0_per_cm2"), a.illustrative("gpu_die_mm2"), a.illustrative("spare_frac"))
    for k, v in y.items():
        w(f"- `{k}`: {_num(v)}")
    w("")
    w("| part | model | format | wafers | fill | stream all weights once over I/O (s) |")
    w("| --- | --- | --- | --- | --- | --- |")
    for p in ("wse-3", "wse-3t"):
        for m in a.MODEL_ORDER:
            x = a.wafers_needed(p, m, "bf16", 1, 1024)
            w(f"| {a.PARTS[p]['name']} | {m} | bf16 | {x['wafers']} | {x['fill']:.4g} | {x['t_stream']:.4g} |")
    w("")
    w("### Chapter 5: chains of compiler-scheduled chips (1,024-token context, batch 1)")
    w("")
    w("| case | part | model | format | chips | stages | per stage | memory term (s) | hops (s) | step (s) | tokens/s per user |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for cc in a.CHAIN_CASES:
        d = a.decode(cc["part"], cc["model"], cc["fmt"], 1, a.CHAIN_CTX, cc["chips"], cc["group"])
        w(
            f"| {cc['key']} | {a.PARTS[cc['part']]['name']} | {cc['model']} | {cc['fmt']} | {d['chips']} | {d['stages']} | {d['group']} | "
            f"{d['t_mem']:.4g} | {d['t_hops']:.4g}{'' if d['hop_known'] else ' (latency not disclosed)'} | {d['t_step']:.4g} | {d['tok_s_user']:.4g} |"
        )
    w("")
    w("| part | model | format | chips to hold it | racks |")
    w("| --- | --- | --- | --- | --- |")
    for p in ("groqchip", "groq3"):
        for m in ("llama3-8b", "llama3-70b", "llama2-70b"):
            for f in ("int8", "bf16"):
                c = a.lpu_capacity(p, m, f, 1024)
                w(f"| {a.PARTS[p]['name']} | {m} | {f} | {c['chips']} | {_num(c['racks'])} |")
    w("")
    w("### Chapter 6: one decoder layer on one SN40L socket, kernel by kernel vs fused (BF16)")
    w("")
    w("| model | case | kernel by kernel: bytes | time (s) | fused: bytes | time (s) | bytes saved |")
    w("| --- | --- | --- | --- | --- | --- | --- |")
    for m in ("llama3-8b", "llama3-70b"):
        for (t, q, c, cz) in DATAFLOW_CASES:
            u = a.layer_traffic("sn40l", m, "bf16", t, q, c, cz, False)
            f = a.layer_traffic("sn40l", m, "bf16", t, q, c, cz, True)
            case = f"prefill of {t:,} tokens" if cz else f"decode, batch {t}, context {c:,}"
            w(f"| {m} | {case} | {_num(u['bytes'])} | {u['t']:.4g} | {_num(f['bytes'])} | {f['t']:.4g} | {1 - f['bytes'] / u['bytes']:.4g} |")
    w("")
    w("| model | sockets | SRAM (B) | layers in SRAM | HBM (B) | models in HBM | DDR (B) | models in DDR | DDR to HBM, one model (s) |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for m in ("llama3-8b", "llama3-70b"):
        for n in (1, 16):
            x = a.tiers("sn40l", m, "bf16", n)
            w(f"| {m} | {n} | {_num(x['sram'])} | {x['sram_layers']} | {_num(x['hbm'])} | {x['hbm_models']} | {_num(x['ddr'])} | {x['ddr_models']} | {x['switch_s']:.4g} |")
    w("")
    w("Graphcore GC2 (SRAM only), chips to hold each model in 16-bit:")
    w("")
    for m in a.MODEL_ORDER:
        pl = a.placement("gc2", m, "bf16", 1, 1024)
        w(f"- {m}: {pl['chips']} chips ({_num(pl['need'])} B)")
    w("")
    w("### Chapter 7: one decoder layer on Blackhole cards (FP8, 1,024-token context)")
    w("")
    w("| model | batch | cards | layer (s) | DRAM part (s) | Ethernet part (s) | layer weights per core (B) | fits in a core's SRAM | tokens/s per user (layers only) |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for m in ("llama3-8b", "llama3-70b"):
        for b in (1, 32):
            for n in MESH_CARDS:
                x = a.mesh_layer("blackhole", m, "fp8", b, 1024, n)
                w(f"| {m} | {b} | {n} | {x['t']:.4g} | {x['t_dram']:.4g} | {x['t_eth']:.4g} | {_num(x['layer_per_core'])} | {_num(x['layer_fits'])} | {x['tok_s_user']:.4g} |")
    w("")
    w("### Chapter 8: one decode step at batch 1 by weight width (1,024-token context)")
    w("")
    w("| device | model | bits | weights (B) | bytes per step | step (s) | tokens/s | fits | J/token |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for p in a.EDGE_DEVICES:
        for m in ("llama3-8b",):
            for e in a.edge_steps(p, m, 1024):
                fits = "capacity not disclosed" if e["fits"] is None else _num(e["fits"])
                w(f"| {p} | {m} | {e['bits']} | {_num(e['weights'])} | {_num(e['bytes'])} | {e['t']:.4g} | {e['tok_s']:.4g} | {fits} | {_t(e['j_per_token'])} |")
    w("")
    w("### Chapter 9: decoder models on the HERMES crossbars")
    w("")
    w("| model | read mode | tiles per layer | chips per layer | tiles | chips | crossbar time per layer (s) | crossbar time per token (s) | crossbar energy per token (J) |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for m in a.MODEL_ORDER:
        for ph in (1, 4):
            x = a.imc_layer(m, ph)
            w(f"| {m} | {ph}-phase | {x['layer_tiles']:,} | {x['layer_chips']:.4g} | {x['tiles']:,} | {x['chips']:,} | {x['t_layer']:.4g} | {x['t_token']:.4g} | {x['e_token']:.4g} |")
    w("")
    w(f"Bits an exact 256-row column sum of 8-bit inputs and 8-bit weights needs: {a.adc_bits(8, 8, 256)} (the chip's ADC path is 8-bit in and out).")
    w("")
    w("### Chapter 10: every family on the same model (batch 1, 1,024-token context, 2,048-token prompt)")
    for m in ("llama3-8b", "llama3-70b"):
        w("")
        w(f"{m}:")
        w("")
        w("| family | part | format | chips | weights in | decode step (s) | per layer (s) | tokens/s per user | prefill (s) | J/token | decode utilisation | bound |")
        w("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in a.choose(m, 1, 1024, 2048):
            name = a.PARTS[r["part"]]["name"] if r["part"] in a.PARTS else a.IMC[r["part"]]["name"]
            w(
                f"| {r['name']} | {name} | {r['fmt']} | {r['chips']:,} | {r['where']} | {r['t_decode']:.4g} | {r['t_layer']:.4g} | "
                f"{r['tok_s_user']:.4g} | {r['t_prefill']:.4g} | {_t(r['j_per_token'])} | {_t(r['util'])} | {r['bound']} ({r['basis']}) |"
            )
    w("")
    return "\n".join(L)


def dump(obj) -> str:
    return json.dumps(obj, indent=None, separators=(",", ":"), allow_nan=False) + "\n"


def main() -> int:
    check = "--check" in sys.argv
    files = {
        ROOT / "src" / "data" / "accel.json": json.dumps(site_data(), indent=1, allow_nan=False) + "\n",
        ROOT / "tests" / "fixtures" / "accel_fixtures.json": dump(fixtures()),
        ROOT / "results.md": results_md(),
    }
    stale = []
    for path, text in files.items():
        if check:
            if not path.exists() or path.read_text() != text:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    if check:
        if stale:
            print("out of date (run python3 scripts/make_fixtures.py):", ", ".join(stale))
            return 1
        print(f"up to date: {len(files)} files")
    else:
        print("wrote", ", ".join(str(p.relative_to(ROOT)) for p in files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
