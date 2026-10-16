/**
 * Frame tests (visual standard §4): for key frames of every animation, the
 * state the site draws equals the Python reference's state for that frame
 * (from the fixtures), and the caption built from the reference's state is
 * the caption the site shows. tests/e2e/frames.spec.ts then checks the page.
 */
import { describe, expect, it } from "vitest";

import fx from "../fixtures/accel_fixtures.json";
import { pathCaption } from "@/lib/accel/captions";
import type { FloorFit, FloorMode, PathStep } from "@/lib/accel/model";

import {
  chainCaption,
  chooseCaption,
  dataflowCaption,
  edgeCaption,
  imcCaption,
  meshCaption,
  partShort,
  mapCaption,
  mxuCaption,
  sweepCaption,
  waferCaption,
} from "@/lib/accel/captions";
import {
  batchSweep,
  chainSteps,
  choose,
  dataflowSteps,
  edgeSteps,
  imcLayer,
  imcSteps,
  meshLayer,
  meshSteps,
  type Choice,
  type DataflowStep,
  type EdgeStep,
  type ImcStep,
  type MeshStep,
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

describe("chapter 6: dataflow frames", () => {
  const py = fx.dataflowSteps.find(
    (c) => c.model === "llama3-8b" && c.tokens === 2048,
  )!.out as DataflowStep[];
  const ts = dataflowSteps("sn40l", "llama3-8b", "bf16", 2048, 1, 2048, true);
  for (const k of [0, 2, 10])
    it(`operation ${k + 1}`, () => {
      expect(ts[k]).toEqual(py[k]);
      expect(dataflowCaption(ts[k]!, 11, "SN40L RDU")).toBe(
        dataflowCaption(py[k]!, 11, "SN40L RDU"),
      );
    });
  it("says what the frames show", () => {
    expect(dataflowCaption(py[2]!, 11, "SN40L RDU")).toContain(
      "Operation 3 of 11, scores QK^T, on one SN40L RDU.",
    );
    expect(py[10]!.f_bytes).toBeLessThan(py[10]!.u_bytes / 2);
  });
});

describe("chapter 7: mesh frames", () => {
  const py = fx.meshSteps.find(
    (c) => c.model === "llama3-70b" && c.batch === 1 && c.cards === 3,
  )!.out as MeshStep[];
  const ts = meshSteps("blackhole", "llama3-70b", "fp8", 1, 1024, 3);
  const m = meshLayer("blackhole", "llama3-70b", "fp8", 1, 1024, 3);
  for (const k of [0, 3, 6])
    it(`step ${k + 1}`, () => {
      expect(ts[k]).toEqual(py[k]);
      expect(meshCaption(ts[k]!, 7, 3, m.cores, m.core_sram)).toBe(
        meshCaption(py[k]!, 7, 3, m.cores, m.core_sram),
      );
    });
  it("says what the frames show", () => {
    expect(meshCaption(py[3]!, 7, 3, m.cores, m.core_sram)).toContain(
      "the 3 cards sum their partial results over Ethernet",
    );
    expect(meshCaption(py[0]!, 7, 3, m.cores, m.core_sram)).toContain(
      "into its 120 cores",
    );
  });
});

describe("chapter 8: edge frames", () => {
  const py = fx.edgeSteps.find(
    (c) => c.part === "m4" && c.model === "llama3-8b" && c.ctx === 1024,
  )!.out as EdgeStep[];
  const ts = edgeSteps("m4", "llama3-8b", 1024);
  for (const k of [0, 1, 6])
    it(`width ${py[k]!.bits}`, () => {
      expect(ts[k]).toEqual(py[k]);
      expect(edgeCaption(ts[k]!, 7, "Apple M4")).toBe(
        edgeCaption(py[k]!, 7, "Apple M4"),
      );
    });
  it("says what the frames show", () => {
    expect(edgeCaption(py[1]!, 7, "Apple M4")).toContain(
      "8-bit weights, 8.03 GB",
    );
    const v = fx.edgeSteps.find(
      (c) => c.part === "v70" && c.model === "llama3-8b" && c.ctx === 1024,
    )!.out as EdgeStep[];
    expect(edgeCaption(v[0]!, 7, "AMD Alveo V70")).toContain(
      "capacity is not disclosed",
    );
  });
});

describe("chapter 9: crossbar frames", () => {
  const py = fx.imcSteps.find((c) => c.model === "llama3-8b" && c.phase === 1)!
    .out as ImcStep[];
  const ts = imcSteps("llama3-8b", 1);
  const m = imcLayer("llama3-8b", 1);
  for (const k of [0, 2, 5])
    it(`stage ${k}`, () => {
      expect(ts[k]).toEqual(py[k]);
      expect(imcCaption(ts[k]!, m, "Llama 3.1 8B")).toBe(
        imcCaption(py[k]!, m, "Llama 3.1 8B"),
      );
    });
  it("says what the frames show", () => {
    expect(imcCaption(py[0]!, m, "Llama 3.1 8B")).toContain(
      "1,789 chips for the whole model",
    );
    expect(imcCaption(py[2]!, m, "Llama 3.1 8B")).toContain(
      "cannot sit in a crossbar",
    );
  });
});

describe("chapter 10: choosing frames", () => {
  const py = fx.choose.find(
    (c) =>
      c.model === "llama3-70b" &&
      c.batch === 1 &&
      c.ctx === 1024 &&
      c.prompt === 2048,
  )!.out as Choice[];
  const ts = choose("llama3-70b", 1, 1024, 2048);
  for (const k of [0, 3, 7])
    it(`family ${k + 1}`, () => {
      expect(ts[k]).toEqual(py[k]);
      expect(
        chooseCaption(ts[k]!, 8, "Llama 3.1 70B", partShort(ts[k]!.part)),
      ).toBe(chooseCaption(py[k]!, 8, "Llama 3.1 70B", partShort(py[k]!.part)));
    });
  it("says what the frames show", () => {
    expect(
      chooseCaption(py[3]!, 8, "Llama 3.1 70B", partShort(py[3]!.part)),
    ).toContain("needs 308 GroqChips");
    expect(
      chooseCaption(py[7]!, 8, "Llama 3.1 70B", partShort(py[7]!.part)),
    ).toContain("crossbar time only");
  });
});

describe("captions: the other branches", () => {
  it("covers one card, attention, a misfit and undisclosed energy", () => {
    const one = meshSteps("blackhole", "llama3-70b", "fp8", 1, 1024, 1);
    const m1 = meshLayer("blackhole", "llama3-70b", "fp8", 1, 1024, 1);
    expect(meshCaption(one[3]!, 7, 1, m1.cores, m1.core_sram)).toContain(
      "one card, so nothing to all-reduce",
    );
    expect(meshCaption(one[1]!, 7, 1, m1.cores, m1.core_sram)).toContain(
      "reads its share of the KV cache",
    );
    const big = edgeSteps("m4", "llama3-70b", 1024);
    expect(edgeCaption(big[0]!, 7, "Apple M4")).toContain(
      "(more than its memory holds)",
    );
    const v = edgeSteps("v70", "llama3-8b", 1024);
    expect(edgeCaption(v[0]!, 7, "AMD Alveo V70")).toContain("J per token");
    const rows = choose("llama3-70b", 1, 1024, 2048);
    expect(
      chooseCaption(rows[1]!, 8, "Llama 3.1 70B", partShort(rows[1]!.part)),
    ).toContain("energy not disclosed");
    expect(partShort("h100")).toBe("H100 SXM");
  });
});

describe("chapter 10: the per-layer floor frames", () => {
  it("captions every mode from the reference's steps", () => {
    const fit = fx.floor as unknown as FloorFit;
    for (const c of fx.path as unknown as {
      id: string;
      mode: FloorMode;
      out: PathStep[];
    }[]) {
      const r = fit.rows.find((x) => x.id === c.id)!;
      const first = pathCaption(c.out[0]!, r, c.mode);
      expect(first).toContain(r.label);
      expect(first).toContain(
        c.mode === "fitted"
          ? "fitted on the other three"
          : c.mode === "own"
            ? "own floor"
            : "no floor",
      );
      const mid = pathCaption(c.out[1]!, r, c.mode);
      expect(mid).toMatch(/^Layer 1 of \d+:/);
      const last = pathCaption(c.out[c.out.length - 1]!, r, c.mode);
      expect(last).toContain("the system published");
    }
  });
});
