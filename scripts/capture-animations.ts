/**
 * Records the README's animated GIFs (and WebM videos) of the hero
 * animations. Manual run, output committed; a heavy job, so run it alone:
 *
 *   pnpm build && pnpm start   # in another shell
 *   pnpm animations            # writes docs/media/*.gif and *.webm
 *
 * Every frame is a model state set by the animation's scrub bar (reduced
 * motion, so nothing plays by itself), screenshotted, then joined by
 * ffmpeg: the recordings are reproducible frame for frame. Needs ffmpeg on
 * the PATH.
 */
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

import { chromium } from "@playwright/test";

const OUT = path.join(process.cwd(), "docs", "media");
const BASE = process.env.SCREENSHOT_BASE_URL ?? "http://localhost:3000";

type Clip = {
  name: string;
  path: string;
  widget: string;
  /** frames per second of the output */
  fps: number;
  /** a radio to press first (a parameter), if any */
  radio?: string;
  /** record every n-th frame only (long animations) */
  stride?: number;
};

const CLIPS: Clip[] = [
  {
    name: "map",
    path: "/learn/01-the-design-space",
    widget: "map-widget",
    fps: 1,
  },
  {
    name: "roofline",
    path: "/learn/02-gpus-the-baseline",
    widget: "roofline-widget",
    fps: 2,
  },
  {
    name: "mxu",
    path: "/learn/03-systolic-designs",
    widget: "mxu-widget",
    fps: 3,
  },
  {
    name: "wafer",
    path: "/learn/04-wafer-scale",
    widget: "wafer-widget",
    fps: 4,
  },
  {
    name: "chain",
    path: "/learn/05-compiler-scheduled",
    widget: "chain-widget",
    fps: 8,
    stride: 2,
  },
  {
    name: "dataflow",
    path: "/learn/06-reconfigurable-dataflow",
    widget: "dataflow-widget",
    fps: 2,
  },
  {
    name: "mesh",
    path: "/learn/07-meshes-of-cores",
    widget: "mesh-widget",
    fps: 2,
  },
  {
    name: "edge",
    path: "/learn/08-fpgas-and-npus",
    widget: "edge-widget",
    fps: 2,
  },
  {
    name: "crossbar",
    path: "/learn/09-in-memory-compute",
    widget: "crossbar-widget",
    fps: 1,
  },
  {
    name: "choose",
    path: "/learn/10-choosing",
    widget: "choose-widget",
    fps: 1,
  },
  {
    name: "latency-floor",
    path: "/learn/10-choosing",
    widget: "path-widget",
    fps: 6,
  },
];

/** `ONLY=power,cost pnpm animations` records just those clips. */
const ONLY = (process.env.ONLY ?? "").split(",").filter(Boolean);

async function main(): Promise<void> {
  mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1280, height: 900 },
    colorScheme: "light",
    reducedMotion: "reduce",
  });
  const page = await context.newPage();
  for (const c of CLIPS.filter((x) => !ONLY.length || ONLY.includes(x.name))) {
    const dir = mkdtempSync(path.join(tmpdir(), `air-${c.name}-`));
    await page.goto(BASE + c.path, { waitUntil: "networkidle" });
    const fig = page.getByTestId(c.widget);
    await fig.waitFor();
    if (c.radio) await fig.getByRole("radio", { name: c.radio }).click();
    // chapter 8 draws once its simulation has run in the worker
    const sim = fig.locator("[data-sim]");
    if (await sim.count())
      await sim.and(page.locator("[data-sim='ready']")).waitFor();
    const scrub = fig.getByTestId("scrub");
    const max = Number(await scrub.getAttribute("max"));
    const visual = fig.getByTestId("visual");
    const stride = c.stride ?? 1;
    let i = 0;
    for (let s = 0; s <= max; s += stride) {
      await scrub.fill(String(s));
      await visual.screenshot({
        path: path.join(dir, `f${String(i++).padStart(4, "0")}.png`),
      });
    }
    const input = path.join(dir, "f%04d.png");
    const gif = path.join(OUT, `${c.name}.gif`);
    const webm = path.join(OUT, `${c.name}.webm`);
    // even dimensions for the video encoder; a shared palette for the GIF
    const scale = "scale=560:-2:flags=lanczos";
    execFileSync("ffmpeg", [
      "-y",
      "-loglevel",
      "error",
      "-framerate",
      String(c.fps),
      "-i",
      input,
      "-vf",
      `${scale},split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=none`,
      "-loop",
      "0",
      gif,
    ]);
    execFileSync("ffmpeg", [
      "-y",
      "-loglevel",
      "error",
      "-framerate",
      String(c.fps),
      "-i",
      input,
      "-vf",
      scale,
      "-c:v",
      "libvpx-vp9",
      "-b:v",
      "0",
      "-crf",
      "40",
      "-pix_fmt",
      "yuv420p",
      webm,
    ]);
    rmSync(dir, { recursive: true, force: true });
    console.log(
      `wrote ${path.relative(process.cwd(), gif)} and .webm (${i} frames)`,
    );
  }
  await browser.close();
}

void main();
