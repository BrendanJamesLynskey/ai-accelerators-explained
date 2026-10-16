/**
 * The TypeScript model against the Python reference's fixtures
 * (scripts/make_fixtures.py): every value exactly, no tolerance, except the
 * wafer-yield exponentials (V8 and libm may differ in the last bit), which
 * are compared to 1e-12.
 */
import { describe, expect, it } from "vitest";

import fx from "../fixtures/accel_fixtures.json";

import {
  PART_ORDER,
  MODEL_ORDER,
  adcBits,
  band,
  choose,
  dataflowSteps,
  edgeBound,
  edgeSteps,
  imcLayer,
  imcSteps,
  layerOps,
  layerTraffic,
  meshLayer,
  meshSteps,
  tiers,
  batchSweep,
  chainSteps,
  decode,
  defectSteps,
  defects,
  familyMap,
  impliedOverhead,
  layerGemms,
  layerMxu,
  lpuCapacity,
  mapSteps,
  maxBatch,
  mxu,
  mxuSteps,
  part,
  peak,
  placement,
  prefill,
  ridge,
  shape,
  throughputCeiling,
  validation,
  PUBLISHED,
  mlperfCeiling,
  sampleFlops,
  floorFit,
  criticalPathSteps,
  type FloorCheck,
  type FloorMode,
  waferYield,
  wafersNeeded,
  type ChainCase,
  type Fmt,
  type ModelId,
  type PartId,
} from "@/lib/accel/model";

type Row<I> = I & { out: unknown };
const rows = <I>(x: unknown) => x as Row<I>[];

describe("records", () => {
  it("flattens every part and shapes every model identically", () => {
    for (const p of PART_ORDER)
      expect(part(p)).toEqual((fx.parts as Record<string, unknown>)[p]);
    for (const m of MODEL_ORDER)
      expect(shape(m)).toEqual((fx.shapes as Record<string, unknown>)[m]);
  });
  it("reads the same peak rate and basis", () => {
    for (const r of rows<{ part: PartId; fmt: Fmt }>(fx.peaks))
      expect(peak(r.part, r.fmt)).toEqual(r.out);
  });
});

describe("placement, decode and prefill", () => {
  it("places every model on every part", () => {
    const rs = rows<{
      part: PartId;
      model: ModelId;
      fmt: Fmt;
      batch: number;
      ctx: number;
    }>(fx.placement);
    expect(rs.length).toBeGreaterThan(400);
    for (const r of rs)
      expect(placement(r.part, r.model, r.fmt, r.batch, r.ctx)).toEqual(r.out);
  });
  it("bounds every decode step", () => {
    const rs = rows<{
      part: PartId;
      model: ModelId;
      fmt: Fmt;
      batch: number;
      ctx: number;
      chips: number;
      group: number;
    }>(fx.decode);
    expect(rs.length).toBeGreaterThan(1000);
    for (const r of rs)
      expect(
        decode(r.part, r.model, r.fmt, r.batch, r.ctx, r.chips, r.group),
      ).toEqual(r.out);
  });
  it("bounds every prefill", () => {
    for (const r of rows<{
      part: PartId;
      model: ModelId;
      fmt: Fmt;
      prompt: number;
      chips: number;
    }>(fx.prefill))
      expect(prefill(r.part, r.model, r.fmt, r.prompt, r.chips)).toEqual(r.out);
  });
  it("gives the same ridge points and throughput ceilings", () => {
    for (const r of rows<{ part: PartId; fmt: Fmt }>(fx.ridge))
      expect(ridge(r.part, r.fmt)).toEqual(r.out);
    for (const r of rows<{
      part: PartId;
      model: ModelId;
      fmt: Fmt;
      chips: number;
    }>(fx.ceiling))
      expect(throughputCeiling(r.part, r.model, r.fmt, r.chips)).toEqual(r.out);
  });
});

describe("chapter 1: the map", () => {
  it("places the families and steps through them", () => {
    expect(familyMap()).toEqual(fx.familyMap);
    for (const r of rows<{ model: ModelId; batch: number; ctx: number }>(
      fx.mapSteps,
    ))
      expect(mapSteps(r.model, r.batch, r.ctx)).toEqual(r.out);
  });
});

describe("chapter 2: the batch sweep", () => {
  it("sweeps every case the widget offers", () => {
    type R = {
      part: PartId;
      model: ModelId;
      fmt: Fmt;
      ctx: number;
      chips: number;
    };
    for (const r of rows<R>(fx.batchSweep))
      expect(batchSweep(r.part, r.model, r.fmt, r.ctx, r.chips)).toEqual(r.out);
    for (const r of rows<R>(fx.maxBatch))
      expect(maxBatch(r.part, r.model, r.fmt, r.ctx, r.chips)).toEqual(r.out);
  });
});

describe("chapter 3: systolic arrays", () => {
  it("tiles the same matmuls in the same cycles", () => {
    for (const m of MODEL_ORDER)
      expect(layerGemms(m)).toEqual(
        (fx.layerGemms as Record<string, unknown>)[m],
      );
    for (const r of rows<{
      rows: number;
      cols: number;
      m: number;
      k: number;
      n: number;
    }>(fx.mxu))
      expect(mxu(r.rows, r.cols, r.m, r.k, r.n)).toEqual(r.out);
    for (const r of rows<{
      rows: number;
      cols: number;
      model: ModelId;
      m: number;
    }>(fx.layerMxu))
      expect(layerMxu(r.rows, r.cols, r.model, r.m)).toEqual(r.out);
    for (const r of rows<{ rows: number; cols: number; m: number }>(
      fx.mxuSteps,
    ))
      expect(mxuSteps(r.rows, r.cols, r.m, 1024, 1024)).toEqual(r.out);
  });
});

describe("chapter 4: wafer-scale", () => {
  it("gives the same yields (exp to 1e-12) and the same defects", () => {
    for (const r of rows<{
      part: PartId;
      d0: number;
      die: number;
      spare: number;
    }>(fx.waferYield)) {
      const ts = waferYield(r.part, r.d0, r.die, r.spare) as unknown as Record<
        string,
        number
      >;
      const py = r.out as Record<string, number>;
      expect(Object.keys(ts).sort()).toEqual(Object.keys(py).sort());
      for (const k of Object.keys(py)) {
        if (k === "die_yield" || k === "wafer_perfect")
          expect(Math.abs(ts[k]! - py[k]!)).toBeLessThanOrEqual(
            1e-12 * Math.abs(py[k]!),
          );
        else expect(ts[k], k).toBe(py[k]);
      }
    }
    for (const r of rows<{ seed: number; count: number }>(fx.defects))
      expect(defects(r.seed, r.count)).toEqual(r.out);
    for (const r of rows<{ seed: number; count: number; per: number }>(
      fx.defectSteps,
    ))
      expect(defectSteps(r.seed, r.count, r.per)).toEqual(r.out);
    for (const r of rows<{
      part: PartId;
      model: ModelId;
      fmt: Fmt;
      ctx: number;
    }>(fx.wafersNeeded))
      expect(wafersNeeded(r.part, r.model, r.fmt, 1, r.ctx)).toEqual(r.out);
  });
});

describe("chapter 5: chains", () => {
  it("runs the same chains and capacities", () => {
    for (const r of rows<{ case: ChainCase }>(fx.chainSteps)) {
      const c = r.case;
      expect(
        chainSteps(c.part, c.model, c.fmt, 1024, c.chips, c.group),
      ).toEqual(r.out);
    }
    for (const r of rows<{
      part: PartId;
      model: ModelId;
      fmt: Fmt;
      ctx: number;
    }>(fx.lpuCapacity))
      expect(lpuCapacity(r.part, r.model, r.fmt, r.ctx)).toEqual(r.out);
  });
});

describe("validation", () => {
  it("reproduces every check, the band and the implied overheads", () => {
    expect(validation()).toEqual(fx.validation);
    expect(band()).toEqual(fx.band);
    for (const [k, v] of Object.entries(fx.overhead))
      expect(impliedOverhead(k)).toEqual(v);
    expect(() => impliedOverhead("mlperf-h100")).toThrow();
  });
});

describe("chapter 6: dataflow", () => {
  type C = {
    part: PartId;
    model: ModelId;
    fmt: Fmt;
    tokens: number;
    seqs: number;
    ctx: number;
    causal: boolean;
  };
  it("lists the same operations and moves the same bytes, fused or not", () => {
    for (const m of MODEL_ORDER)
      expect(layerOps(m)).toEqual((fx.layerOps as Record<string, unknown>)[m]);
    const rs = rows<C & { fused: boolean }>(fx.layerTraffic);
    expect(rs.length).toBeGreaterThan(10);
    for (const r of rs)
      expect(
        layerTraffic(
          r.part,
          r.model,
          r.fmt,
          r.tokens,
          r.seqs,
          r.ctx,
          r.causal,
          r.fused,
        ),
      ).toEqual(r.out);
    for (const r of rows<C>(fx.dataflowSteps))
      expect(
        dataflowSteps(
          r.part,
          r.model,
          r.fmt,
          r.tokens,
          r.seqs,
          r.ctx,
          r.causal,
        ),
      ).toEqual(r.out);
    for (const r of rows<{
      part: PartId;
      model: ModelId;
      fmt: Fmt;
      sockets: number;
    }>(fx.tiers))
      expect(tiers(r.part, r.model, r.fmt, r.sockets)).toEqual(r.out);
  });
});

describe("chapter 7: a mesh of cores", () => {
  it("splits the same layer over the same cards", () => {
    type R = {
      part: PartId;
      model: ModelId;
      fmt: Fmt;
      batch: number;
      ctx: number;
      cards: number;
    };
    for (const r of rows<R>(fx.meshLayer))
      expect(
        meshLayer(r.part, r.model, r.fmt, r.batch, r.ctx, r.cards),
      ).toEqual(r.out);
    for (const r of rows<R>(fx.meshSteps))
      expect(
        meshSteps(r.part, r.model, r.fmt, r.batch, r.ctx, r.cards),
      ).toEqual(r.out);
  });
});

describe("chapter 8: edge devices", () => {
  it("bounds the same decode step at every weight width", () => {
    for (const r of rows<{
      part: string;
      model: ModelId;
      bits: number;
      ctx: number;
    }>(fx.edgeBound))
      expect(edgeBound(r.part, r.model, r.bits, r.ctx)).toEqual(r.out);
    for (const r of rows<{ part: string; model: ModelId; ctx: number }>(
      fx.edgeSteps,
    ))
      expect(edgeSteps(r.part, r.model, r.ctx)).toEqual(r.out);
  });
});

describe("chapter 9: in-memory compute", () => {
  it("maps the same layers onto the same crossbars", () => {
    for (const r of rows<{ model: ModelId; phase: number }>(fx.imcLayer))
      expect(imcLayer(r.model, r.phase)).toEqual(r.out);
    for (const r of rows<{ model: ModelId; phase: number }>(fx.imcSteps))
      expect(imcSteps(r.model, r.phase)).toEqual(r.out);
    for (const r of rows<{ in: number; w: number; rows: number }>(fx.adcBits))
      expect(adcBits(r.in, r.w, r.rows)).toEqual(r.out);
  });
});

describe("chapter 10: choosing", () => {
  it("runs every family on the same model", () => {
    for (const r of rows<{
      model: ModelId;
      batch: number;
      ctx: number;
      prompt: number;
    }>(fx.choose))
      expect(choose(r.model, r.batch, r.ctx, r.prompt)).toEqual(r.out);
  });
});

describe("brief 42: MLPerf with prefill, and the latency floor", () => {
  it("gives the same MLPerf ceilings and sample flops", () => {
    const P = (k: string) => PUBLISHED[k]!.v;
    const nIn = P("mlperf-input-mean");
    const nOut = P("mlperf-tokens-per-sample");
    expect(mlperfCeiling("h100", "llama2-70b", "fp8", 8, nIn, nOut)).toEqual(
      fx.mlperf.h100,
    );
    expect(
      mlperfCeiling(
        "b200",
        "llama2-70b",
        "fp4",
        P("mlperf-gb200-chips"),
        nIn,
        nOut,
      ),
    ).toEqual(fx.mlperf.gb200);
    expect(mlperfCeiling("mi355x", "llama2-70b", "fp4", 8, nIn, nOut)).toEqual(
      fx.mlperf.mi355x,
    );
    for (const r of rows<{
      model: ModelId;
      in: number;
      out: number;
      flops: unknown;
    }>(fx.sampleFlops))
      expect(sampleFlops(r.model, r.in, r.out)).toEqual(r.flops);
  });

  it("fits the same floor and walks the same critical paths", () => {
    expect(floorFit()).toEqual(fx.floor);
    for (const r of rows<{ id: FloorCheck; mode: FloorMode; out: unknown }>(
      fx.path,
    ))
      expect(criticalPathSteps(r.id, r.mode)).toEqual(r.out);
  });

  it("including prefill raises both GPU ratios to about a third", () => {
    const v = validation();
    const h = v.find((x) => x.id === "mlperf-h100")!;
    const g = v.find((x) => x.id === "mlperf-gb200")!;
    expect(h.ratio).toBeGreaterThan(0.3);
    expect(h.ratio).toBeLessThan(0.4);
    expect(g.ratio).toBeGreaterThan(0.3);
    expect(g.ratio).toBeLessThan(0.4);
  });

  it("the floor term predicts every SRAM system within 2x, the bound alone misses by up to 770x", () => {
    const f = floorFit();
    expect(f.worstPred).toBeLessThan(2);
    expect(f.worstBound).toBeGreaterThan(700);
    expect(f.min).toBeGreaterThan(10e-6);
    expect(f.max).toBeLessThan(40e-6);
  });
});
