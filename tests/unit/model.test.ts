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
  band,
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
