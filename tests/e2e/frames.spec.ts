/**
 * Frame tests on the page (visual standard §4): set key frames of every
 * animation and require the caption on screen to be the caption built from
 * the Python reference's state for that frame (tests/fixtures), and the
 * picture to show that state.
 */
import { expect, test, type Locator } from "@playwright/test";

import fx from "../fixtures/accel_fixtures.json";

import {
  chainCaption,
  mapCaption,
  mxuCaption,
  sweepCaption,
  waferCaption,
} from "@/lib/accel/captions";
import {
  decode,
  part,
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
