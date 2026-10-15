/**
 * Captures the README screenshots. Manual run, output committed.
 *
 *   pnpm build && pnpm start   # in another shell
 *   pnpm screenshots           # headless Chromium writes docs/screenshots/*.png
 *
 * Light theme, fixed viewport, reduced motion (nothing plays by itself), and
 * each animation set to a chosen mid-animation frame by its scrub bar, so
 * the pictures are reproducible.
 */
import { mkdirSync } from "node:fs";
import path from "node:path";

import { chromium } from "@playwright/test";

const OUT = path.join(process.cwd(), "docs", "screenshots");
const BASE = process.env.SCREENSHOT_BASE_URL ?? "http://localhost:3000";

type Shot = { name: string; path: string; widget?: string; step?: number };

const SHOTS: Shot[] = [
  { name: "01-landing", path: "/" },
  {
    name: "02-map",
    path: "/learn/01-the-design-space",
    widget: "map-widget",
    step: 3,
  },
  {
    name: "03-roofline",
    path: "/learn/02-gpus-the-baseline",
    widget: "roofline-widget",
    step: 7,
  },
  {
    name: "04-mxu",
    path: "/learn/03-systolic-designs",
    widget: "mxu-widget",
    step: 6,
  },
  {
    name: "05-wafer",
    path: "/learn/04-wafer-scale",
    widget: "wafer-widget",
    step: 15,
  },
  {
    name: "06-chain",
    path: "/learn/05-compiler-scheduled",
    widget: "chain-widget",
    step: 40,
  },
  { name: "07-specs", path: "/specs" },
];

async function main(): Promise<void> {
  mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    colorScheme: "light",
    reducedMotion: "reduce",
  });
  const page = await context.newPage();
  for (const s of SHOTS) {
    await page.goto(BASE + s.path, { waitUntil: "networkidle" });
    const file = path.join(OUT, `${s.name}.png`);
    if (s.widget) {
      const fig = page.getByTestId(s.widget);
      await fig.waitFor();
      const sim = fig.locator("[data-sim]");
      if (await sim.count())
        await sim.and(page.locator("[data-sim='ready']")).waitFor();
      if (s.step !== undefined)
        await fig.getByTestId("scrub").fill(String(s.step));
      await fig.screenshot({ path: file });
    } else {
      await page.screenshot({ path: file });
    }
    console.log(`wrote ${path.relative(process.cwd(), file)}`);
  }
  await browser.close();
}

void main();
