/**
 * Frame tests on the page (visual standard §4): set key frames of every
 * animation and require the caption on screen to be the caption built from
 * the Python reference's state for that frame (tests/fixtures), and the
 * picture to show that state.
 */
import { expect, test, type Locator } from "@playwright/test";

import fx from "../fixtures/accel_fixtures.json";

import type { FloorFit, PathStep } from "@/lib/accel/model";
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
  pathCaption,
  sweepCaption,
  waferCaption,
} from "@/lib/accel/captions";
import {
  decode,
  imcLayer,
  meshLayer,
  part,
  type Choice,
  type DataflowStep,
  type EdgeStep,
  type ImcStep,
  type MeshStep,
  ridge,
  type ChainStep,
  type DefectStep,
  type MapStep,
  type MxuStep,
  type SweepStep,
  type WaferYield,
} from "@/lib/accel/model";

// No autoplay (reduced motion): the test sets each frame itself.
test.use({ contextOptions: { reducedMotion: "reduce" } });

/**
 * Change a parameter and wait until the animation has restarted for it
 * (its data-key changes in the reset itself), so a following scrub cannot
 * be undone by the reset.
 */
async function change(fig: Locator, act: () => Promise<void>): Promise<void> {
  const before = (await fig.getAttribute("data-key")) ?? "";
  await act();
  await expect(fig).not.toHaveAttribute("data-key", before);
}

async function show(fig: Locator, s: number): Promise<void> {
  if ((await fig.getAttribute("data-playing")) === "true")
    await fig.getByTestId("play").click();
  await fig.getByTestId("scrub").fill(String(s));
  await expect(fig).toHaveAttribute("data-step", String(s));
  await expect(fig).toHaveAttribute("data-playing", "false");
}

const pname = (id: string) =>
  (part(id as never).name as string).split(" (")[0]!;

test("map: every family for Llama 3.1 70B, then 8B", async ({ page }) => {
  await page.goto("/learn/01-the-design-space");
  const fig = page.getByTestId("map-widget");
  const steps = (m: string) =>
    fx.mapSteps.find((c) => c.model === m && c.ctx === 1024)!.out as MapStep[];
  const cap = (s: MapStep, name: string) =>
    mapCaption(s, 8, name, s.part ? pname(s.part) : "");
  const big = steps("llama3-70b");
  for (const s of [0, 3, 4, 7]) {
    await show(fig, s);
    await expect(fig.getByTestId("caption")).toHaveText(
      cap(big[s]!, "Llama 3.1 70B"),
    );
    await expect(fig.locator(`[data-family='${big[s]!.id}']`)).toHaveAttribute(
      "data-active",
      "true",
    );
  }
  await show(fig, 3);
  await expect(fig.locator("[data-chips]")).toHaveAttribute(
    "data-chips",
    String(big[3]!.chips),
  );
  await change(fig, () =>
    fig.getByRole("radio", { name: "Llama 3.1 8B" }).click(),
  );
  const small = steps("llama3-8b");
  await show(fig, 2);
  await expect(fig.getByTestId("caption")).toHaveText(
    cap(small[2]!, "Llama 3.1 8B"),
  );
});

test("roofline: 70B in FP8 on 8 H100s, batch by batch, then B200", async ({
  page,
}) => {
  await page.goto("/learn/02-gpus-the-baseline");
  const fig = page.getByTestId("roofline-widget");
  const sweep = (p: string) =>
    fx.batchSweep.find(
      (c) =>
        c.part === p &&
        c.model === "llama3-70b" &&
        c.fmt === "fp8" &&
        c.ctx === 1024 &&
        c.chips === 8,
    )!.out as SweepStep[];
  const h = sweep("h100");
  for (const s of [0, 6, 10]) {
    await show(fig, s);
    await expect(fig.getByTestId("caption")).toHaveText(
      sweepCaption(h[s]!, 12, "H100 SXM", ridge("h100", "fp8")!),
    );
  }
  // memory-bound at batch 64, compute-bound at 1,024
  expect(h[6]!.bound).toBe("memory");
  expect(h[10]!.bound).toBe("compute");
  await change(fig, () => fig.getByRole("radio", { name: "B200" }).click());
  const b = sweep("b200");
  await show(fig, 11);
  await expect(fig.getByTestId("caption")).toHaveText(
    sweepCaption(b[11]!, 12, "B200", ridge("b200", "fp8")!),
  );
  await expect(fig.locator("[data-batch]")).toHaveCount(12);
});

test("mxu: a 256 x 256 array at batch 1, then 128 x 128 at batch 128", async ({
  page,
}) => {
  await page.goto("/learn/03-systolic-designs");
  const fig = page.getByTestId("mxu-widget");
  const steps = (r: number, m: number) =>
    fx.mxuSteps.find((c) => c.rows === r && c.m === m)!.out as MxuStep[];
  const a = steps(256, 1);
  for (const s of [0, 5, 16]) {
    await show(fig, s);
    await expect(fig.getByTestId("caption")).toHaveText(
      mxuCaption(a[s]!, 16, 256, 256, 1),
    );
  }
  await show(fig, 5);
  await expect(fig.locator("[data-tile='4']")).toHaveAttribute(
    "data-state",
    "active",
  );
  await expect(fig.locator("[data-tile='5']")).toHaveAttribute(
    "data-state",
    "loading",
  );
  await expect(fig.locator("[data-tile='3']")).toHaveAttribute(
    "data-state",
    "done",
  );
  await change(fig, () =>
    fig.getByRole("radio", { name: "128 x 128 (Trainium2)" }).click(),
  );
  await change(fig, () =>
    fig.getByRole("radio", { name: "128", exact: true }).click(),
  );
  const b = steps(128, 128);
  await show(fig, 40);
  await expect(fig.getByTestId("caption")).toHaveText(
    mxuCaption(b[40]!, 64, 128, 128, 128),
  );
  await expect(fig.locator("[data-tile]")).toHaveCount(64);
});

test("wafer: defects at 0.1 per cm², then 0.2", async ({ page }) => {
  await page.goto("/learn/04-wafer-scale");
  const fig = page.getByTestId("wafer-widget");
  const y = (d0: number) =>
    fx.waferYield.find(
      (c) => c.part === "wse-3" && c.d0 === d0 && c.spare === 0.015,
    )!.out as WaferYield;
  const steps = (n: number, per: number) =>
    fx.defectSteps.find((c) => c.seed === 7 && c.count === n && c.per === per)!
      .out as DefectStep[];
  const a = steps(46, 2);
  for (const s of [0, 5, 23]) {
    await show(fig, s);
    await expect(fig.getByTestId("caption")).toHaveText(
      waferCaption(a[s]!, 46, 84, 900000, y(0.1).spares),
    );
    await expect(fig.locator("[data-defect]")).toHaveCount(a[s]!.shown);
    await expect(fig.locator("[data-die][data-hit='true']")).toHaveCount(
      a[s]!.dies_hit,
    );
  }
  await change(fig, () => fig.getByRole("radio", { name: "0.2 /cm²" }).click());
  const b = steps(92, 4);
  await show(fig, 10);
  await expect(fig.getByTestId("caption")).toHaveText(
    waferCaption(b[10]!, 92, 84, 900000, y(0.2).spares),
  );
});

test("chain: one token through 576 GroqChips, then one chip per stage", async ({
  page,
}) => {
  await page.goto("/learn/05-compiler-scheduled");
  const fig = page.getByTestId("chain-widget");
  const run = (key: string) => fx.chainSteps.find((c) => c.case.key === key)!;
  const demo = run("demo");
  const d = decode("groqchip", "llama2-70b", "int8", 1, 1024, 576, 8);
  const steps = demo.out as ChainStep[];
  expect(steps.length).toBe(72);
  for (const s of [0, 35, 71]) {
    await show(fig, s);
    await expect(fig.getByTestId("caption")).toHaveText(
      chainCaption(steps[s]!, d, "GroqChip", 722e-9),
    );
    await expect(fig.locator("[data-stage][data-state='done']")).toHaveCount(s);
  }
  await change(fig, () =>
    fig.getByRole("radio", { name: "fewest chips, one per stage" }).click(),
  );
  const one = run("fewest-chain").out as ChainStep[];
  const d1 = decode("groqchip", "llama2-70b", "int8", 1, 1024, 0, 1);
  await show(fig, 100);
  await expect(fig.getByTestId("caption")).toHaveText(
    chainCaption(one[100]!, d1, "GroqChip", 722e-9),
  );
  await expect(fig.locator("[data-stage]")).toHaveCount(one.length);
});

test("dataflow: a prefill layer kernel by kernel and fused, then decode", async ({
  page,
}) => {
  await page.goto("/learn/06-reconfigurable-dataflow");
  const fig = page.getByTestId("dataflow-widget");
  const steps = (m: string, t: number) =>
    fx.dataflowSteps.find((c) => c.model === m && c.tokens === t)!
      .out as DataflowStep[];
  const pf = steps("llama3-8b", 2048);
  for (const s of [0, 2, 10]) {
    await show(fig, s);
    await expect(fig.getByTestId("caption")).toHaveText(
      dataflowCaption(pf[s]!, 11, "SN40L RDU"),
    );
    await expect(
      fig.locator(`[data-lane='u'][data-op='${s}']`),
    ).toHaveAttribute("data-state", "active");
  }
  await change(fig, () =>
    fig.getByRole("radio", { name: "Decode, batch 1" }).click(),
  );
  const d1 = steps("llama3-8b", 1);
  await show(fig, 4);
  await expect(fig.getByTestId("caption")).toHaveText(
    dataflowCaption(d1[4]!, 11, "SN40L RDU"),
  );
});

test("mesh: a 70B layer on 3 cards, then on 1", async ({ page }) => {
  await page.goto("/learn/07-meshes-of-cores");
  const fig = page.getByTestId("mesh-widget");
  const steps = (n: number) =>
    fx.meshSteps.find(
      (c) => c.model === "llama3-70b" && c.batch === 1 && c.cards === n,
    )!.out as MeshStep[];
  const m3 = meshLayer("blackhole", "llama3-70b", "fp8", 1, 1024, 3);
  const three = steps(3);
  for (const s of [0, 3, 6]) {
    await show(fig, s);
    await expect(fig.getByTestId("caption")).toHaveText(
      meshCaption(three[s]!, 7, 3, m3.cores, m3.core_sram),
    );
  }
  await expect(fig.locator("[data-cores]")).toHaveAttribute(
    "data-cores",
    "120",
  );
  await change(fig, () =>
    fig
      .getByRole("radiogroup", { name: "Cards" })
      .getByRole("radio", { name: "1", exact: true })
      .click(),
  );
  const one = steps(1);
  await show(fig, 3);
  await expect(fig.getByTestId("caption")).toHaveText(
    meshCaption(one[3]!, 7, 1, m3.cores, m3.core_sram),
  );
});

test("edge: Llama 3.1 8B at every width on the M4, then the V70", async ({
  page,
}) => {
  await page.goto("/learn/08-fpgas-and-npus");
  const fig = page.getByTestId("edge-widget");
  const steps = (p: string) =>
    fx.edgeSteps.find(
      (c) => c.part === p && c.model === "llama3-8b" && c.ctx === 1024,
    )!.out as EdgeStep[];
  const m4 = steps("m4");
  for (const s of [0, 1, 6]) {
    await show(fig, s);
    await expect(fig.getByTestId("caption")).toHaveText(
      edgeCaption(m4[s]!, 7, "Apple M4"),
    );
    await expect(fig.locator(`[data-bits='${m4[s]!.bits}']`)).toHaveAttribute(
      "data-state",
      "active",
    );
  }
  await change(fig, () =>
    fig.getByRole("radio", { name: "AMD Alveo V70" }).click(),
  );
  const v = steps("v70");
  await show(fig, 4);
  await expect(fig.getByTestId("caption")).toHaveText(
    edgeCaption(v[4]!, 7, "AMD Alveo V70"),
  );
  await change(fig, () =>
    fig.getByRole("radio", { name: "AMD Alveo V80" }).click(),
  );
  const h = steps("v80");
  for (const s of [0, 4, 6]) {
    await show(fig, s);
    await expect(fig.getByTestId("caption")).toHaveText(
      edgeCaption(h[s]!, 7, "AMD Alveo V80"),
    );
  }
});

test("crossbar: one 8B layer, then the 70B model in the precise mode", async ({
  page,
}) => {
  await page.goto("/learn/09-in-memory-compute");
  const fig = page.getByTestId("crossbar-widget");
  const steps = (m: string, ph: number) =>
    fx.imcSteps.find((c) => c.model === m && c.phase === ph)!.out as ImcStep[];
  const a = steps("llama3-8b", 1);
  const ma = imcLayer("llama3-8b", 1);
  for (const s of [0, 1, 2, 5]) {
    await show(fig, s);
    await expect(fig.getByTestId("caption")).toHaveText(
      imcCaption(a[s]!, ma, "Llama 3.1 8B"),
    );
  }
  await show(fig, 2);
  await expect(fig.locator("[data-digital]")).toHaveAttribute(
    "data-digital",
    "active",
  );
  await change(fig, () =>
    fig.getByRole("radio", { name: "4-phase (precise)" }).click(),
  );
  await change(fig, () =>
    fig.getByRole("radio", { name: "Llama 3.1 70B" }).click(),
  );
  const b = steps("llama3-70b", 4);
  await show(fig, 4);
  await expect(fig.getByTestId("caption")).toHaveText(
    imcCaption(b[4]!, imcLayer("llama3-70b", 4), "Llama 3.1 70B"),
  );
});

test("choose: every family on Llama 3.1 70B, then 8B", async ({ page }) => {
  await page.goto("/learn/10-choosing");
  const fig = page.getByTestId("choose-widget");
  const rows = (m: string) =>
    fx.choose.find(
      (c) =>
        c.model === m && c.batch === 1 && c.ctx === 1024 && c.prompt === 2048,
    )!.out as Choice[];
  const big = rows("llama3-70b");
  for (const s of [0, 3, 7]) {
    await show(fig, s);
    await expect(fig.getByTestId("caption")).toHaveText(
      chooseCaption(big[s]!, 8, "Llama 3.1 70B", partShort(big[s]!.part)),
    );
    await expect(
      fig.locator(`[data-family='${big[s]!.family}']`),
    ).toHaveAttribute("data-state", "active");
  }
  await change(fig, () =>
    fig.getByRole("radio", { name: "Llama 3.1 8B" }).click(),
  );
  const small = rows("llama3-8b");
  await show(fig, 6);
  await expect(fig.getByTestId("caption")).toHaveText(
    chooseCaption(small[6]!, 8, "Llama 3.1 8B", partShort(small[6]!.part)),
  );
});

test("the per-layer floor: one token's critical path", async ({ page }) => {
  await page.goto("/learn/10-choosing");
  const fig = page.getByTestId("path-widget");
  const fit = fx.floor as unknown as FloorFit;
  const path = (id: string, mode: string) =>
    fx.path.find((c) => c.id === id && c.mode === mode)!
      .out as unknown as PathStep[];
  const row = (id: string) => fit.rows.find((r) => r.id === id)!;
  // default: Cerebras 70B, the floor fitted on the other three
  const a = path("cerebras-70b-tps", "fitted");
  for (const s of [0, 1, 40, a.length - 1]) {
    await show(fig, s);
    await expect(fig.getByTestId("caption")).toHaveText(
      pathCaption(a[s]!, row("cerebras-70b-tps"), "fitted"),
    );
  }
  await change(fig, () => fig.getByRole("radio", { name: "SN40L 8B" }).click());
  await change(fig, () =>
    fig.getByRole("radio", { name: "none (the bound)" }).click(),
  );
  const b = path("sn40l-8b-tps", "none");
  for (const s of [0, 16, b.length - 1]) {
    await show(fig, s);
    await expect(fig.getByTestId("caption")).toHaveText(
      pathCaption(b[s]!, row("sn40l-8b-tps"), "none"),
    );
  }
});
