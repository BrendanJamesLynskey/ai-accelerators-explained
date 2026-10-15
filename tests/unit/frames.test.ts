/**
 * Frame tests (visual standard §4): for key frames of every animation, the
 * state the site draws equals the Python reference's state for that frame
 * (from the fixtures), and the caption built from the reference's state is
 * the caption the site shows. tests/e2e/frames.spec.ts then checks the page.
 */
import { describe, expect, it } from "vitest";

import fx from "../fixtures/accel_fixtures.json";

import {
  chainCaption,
  mapCaption,
  mxuCaption,
  sweepCaption,
  waferCaption,
} from "@/lib/accel/captions";
import {
  batchSweep,
  chainSteps,
  decode,
  defectSteps,
  mapSteps,
  mxuSteps,
  part,
  ridge,
  type ChainStep,
  type DefectStep,
  type MapStep,
  type MxuStep,
  type SweepStep,
} from "@/lib/accel/model";

const pname = (id: string) =>
  (part(id as never).name as string).split(" (")[0]!;

describe("chapter 1: map frames", () => {
  const py = fx.mapSteps.find(
    (c) => c.model === "llama3-70b" && c.ctx === 1024,
  )!.out as MapStep[];
  const ts = mapSteps("llama3-70b", 1, 1024);
  const cap = (s: MapStep) =>
    mapCaption(s, 8, "Llama 3.1 70B", s.part ? pname(s.part) : "");
  for (const k of [0, 2, 3, 7])
    it(`family ${k + 1}`, () => {
      expect(ts[k]).toEqual(py[k]);
      expect(cap(ts[k]!)).toBe(cap(py[k]!));
    });
  it("says what the frames show", () => {
    expect(cap(py[3]!)).toBe(
      "Family 4 of 8, compiler-scheduled (chapter 5; e.g. Groq LPU): the weights live in on-chip SRAM; data movement is scheduled by the compiler, ahead of time; a few large engines. Llama 3.1 70B in INT8, weights plus one 1,024-token KV cache, is 70.7 GB: 308 GroqChips at 230 MB of on-chip SRAM each.",
    );
    expect(cap(py[7]!)).toContain("Its numbers come in chapter 9.");
  });
});

describe("chapter 2: roofline frames", () => {
  const py = fx.batchSweep.find(
    (c) =>
      c.part === "h100" &&
      c.model === "llama3-70b" &&
      c.fmt === "fp8" &&
      c.ctx === 1024 &&
      c.chips === 8,
  )!.out as SweepStep[];
  const ts = batchSweep("h100", "llama3-70b", "fp8", 1024, 8);
  const r = ridge("h100", "fp8")!;
  for (const k of [0, 6, 10, 11])
    it(`batch ${py[k]!.batch}`, () => {
      expect(ts[k]).toEqual(py[k]);
      expect(sweepCaption(ts[k]!, 12, "H100 SXM", r)).toBe(
        sweepCaption(py[k]!, 12, "H100 SXM", r),
      );
    });
  it("says which side of the ridge the step is on", () => {
    expect(sweepCaption(py[0]!, 12, "H100 SXM", r)).toContain(
      "below the ridge of 591, so the step is memory-bound",
    );
    expect(sweepCaption(py[10]!, 12, "H100 SXM", r)).toContain("compute-bound");
    const wide = fx.batchSweep.find(
      (c) =>
        c.part === "h100" &&
        c.model === "llama3-70b" &&
        c.ctx === 8192 &&
        c.chips === 8,
    )!.out as SweepStep[];
    expect(sweepCaption(wide[11]!, 12, "H100 SXM", r)).toContain(
      "do not fit in memory",
    );
  });
});

describe("chapter 3: tile frames", () => {
  const py = fx.mxuSteps.find((c) => c.rows === 256 && c.m === 1)!
    .out as MxuStep[];
  const ts = mxuSteps(256, 256, 1, 1024, 1024);
  for (const k of [0, 1, 8, 16])
    it(`tile ${k}`, () => {
      expect(ts[k]).toEqual(py[k]);
      expect(mxuCaption(ts[k]!, 16, 256, 256, 1)).toBe(
        mxuCaption(py[k]!, 16, 256, 256, 1),
      );
    });
  it("says what the frames show", () => {
    expect(mxuCaption(py[0]!, 16, 256, 256, 1)).toContain(
      "costs 510 cycles once",
    );
    expect(mxuCaption(py[16]!, 16, 256, 256, 1)).toContain(
      "PEs, the last tile. 4,606 cycles so far; 0.3%",
    );
  });
});

describe("chapter 4: wafer frames", () => {
  const py = fx.defectSteps.find(
    (c) => c.seed === 7 && c.count === 46 && c.per === 2,
  )!.out as DefectStep[];
  const ts = defectSteps(7, 46, 2);
  for (const k of [0, 5, 23])
    it(`step ${k}`, () => {
      expect(ts[k]).toEqual(py[k]);
      expect(waferCaption(ts[k]!, 46, 84, 900000, 13500)).toBe(
        waferCaption(py[k]!, 46, 84, 900000, 13500),
      );
    });
  it("says what the frames show", () => {
    expect(waferCaption(py[0]!, 46, 84, 900000, 13500)).toContain(
      "13,500 cores are held as spares",
    );
    expect(waferCaption(py[23]!, 46, 84, 900000, 13500)).toContain(
      "46 of 46 defects placed",
    );
  });
});

describe("chapter 5: chain frames", () => {
  const py = fx.chainSteps.find((c) => c.case.key === "demo")!
    .out as ChainStep[];
  const ts = chainSteps("groqchip", "llama2-70b", "int8", 1024, 576, 8);
  const d = decode("groqchip", "llama2-70b", "int8", 1, 1024, 576, 8);
  for (const k of [0, 35, 71])
    it(`stage ${k + 1}`, () => {
      expect(ts[k]).toEqual(py[k]);
      expect(chainCaption(ts[k]!, d, "GroqChip", 722e-9)).toBe(
        chainCaption(py[k]!, d, "GroqChip", 722e-9),
      );
    });
  it("says what the frames show", () => {
    expect(chainCaption(py[0]!, d, "GroqChip", 722e-9)).toContain(
      "then hops 722 ns to stage 2",
    );
    expect(chainCaption(py[71]!, d, "GroqChip", 722e-9)).toContain(
      "the token is out after 274 µs",
    );
    const g3 = fx.chainSteps.find((c) => c.case.key === "groq3")!
      .out as ChainStep[];
    const dg = decode("groq3", "llama3-70b", "fp8", 1, 1024);
    expect(chainCaption(g3[0]!, dg, "NVIDIA Groq 3 LPU", null)).toContain(
      "hop latency not disclosed",
    );
  });
});
