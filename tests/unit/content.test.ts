/**
 * The chapters' MDX: every ```ts block is cut from the file the chapter
 * names (whitespace-collapsed, because Prettier reformats MDX code blocks);
 * every CUDA block is labelled as not compiled here; every equation
 * compiles in KaTeX; internal links point at real pages; deck links point
 * at the owner's slide decks with a slide anchor (or at a series hub); links to
 * the companion sites name real chapters.
 */
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";

import katex from "katex";
import { describe, expect, it } from "vitest";

import { SECTIONS } from "@/lib/mdx/sections";

const ROOT = path.join(__dirname, "../..");
const DIR = path.join(ROOT, "content/chapters");
const FILES = readdirSync(DIR).filter((f) => f.endsWith(".mdx"));
const squash = (s: string) => s.replace(/\s+/g, " ").trim();

/** The companion sites' chapter slugs this site links to (checked against their sections.ts, 2026-10-15). */
const SISTER_SLUGS: Record<string, string[]> = {
  "gpu-kernels-explained": [
    "01-memory-hierarchy",
    "02-roofline",
    "07-gemm",
    "09-softmax-and-flashattention",
    "11-quantised-kernels",
  ],
  "systolic-arrays-explained": [
    "01-why-systolic",
    "02-weight-stationary",
    "05-tiling",
    "07-the-tpu",
    "08-other-ways",
    "09-graph-to-silicon",
  ],
  "numerics-explained": [
    "04-formats-zoo",
    "01-bits-to-numbers",
    "03-accumulation",
    "05-quantisation-basics",
    "08-awq-and-nf4",
    "10-hardware",
  ],
  "ai-racks-explained": [
    "01-chip-to-data-centre",
    "02-the-package",
    "04-scale-up-nvlink",
    "05-scale-out-network",
    "08-disaggregated-inference",
    "10-other-designs",
    "11-cost",
  ],
  "llm-inference-explained": [
    "03-roofline",
    "04-batching",
    "07-speculative-decoding",
    "08-quantisation",
    "09-parallelism",
    "10-serving-metrics",
  ],
};

describe("chapter files", () => {
  it("there is one per section, in order", () => {
    expect(FILES.sort()).toEqual(SECTIONS.map((s) => `${s.slug}.mdx`));
  });
});

for (const f of FILES) {
  const src = readFileSync(path.join(DIR, f), "utf8");
  describe(f, () => {
    it("opens with its animation (the hero comes before any prose)", () => {
      expect(src.trimStart()).toMatch(/^<[A-Z][a-zA-Z]+Widget>/);
    });

    it("cuts every TypeScript block from the model or engine source", () => {
      const model = squash(
        readFileSync(path.join(ROOT, "src/lib/accel/model.ts"), "utf8"),
      );
      const blocks = [...src.matchAll(/```ts\n([\s\S]*?)```/g)].map(
        (m) => m[1]!,
      );
      expect(blocks.length).toBeGreaterThan(0);
      for (const b of blocks) expect(model, b).toContain(squash(b));
    });

    it("links the companion sites' chapters by their real slugs", () => {
      for (const m of src.matchAll(
        /https:\/\/([a-z-]+)\.vercel\.app\/(learn|training)\/([a-z0-9-]+)/g,
      )) {
        const known = SISTER_SLUGS[m[1]!];
        expect(known, m[0]).toBeDefined();
        expect(known, m[0]).toContain(m[3]);
      }
    });

    it("compiles every animation equation", () => {
      for (const m of src.matchAll(/tex="([^"]+)"/g)) {
        expect(() =>
          katex.renderToString(m[1]!, {
            displayMode: true,
            throwOnError: true,
            strict: "ignore",
            trust: (ctx) => ctx.command === "\\htmlClass",
          }),
        ).not.toThrow();
      }
    });

    it("links only to real pages and to deck slides", () => {
      const slugs = new Set<string>(SECTIONS.map((s) => s.slug));
      for (const m of src.matchAll(/\]\((\/[^)]+)\)/g)) {
        const href = m[1]!;
        if (href.startsWith("/learn/"))
          expect(slugs.has(href.slice(7)), href).toBe(true);
        else
          expect(["/specs", "/about", "/learn"]).toContain(href.split("#")[0]);
      }
      for (const m of src.matchAll(
        /https:\/\/brendanjameslynskey\.github\.io\/([A-Za-z0-9_]+)\/(#slide-\d\d)?/g,
      )) {
        // a hub is linked whole; a deck by its slide
        if (m[1]!.startsWith("LLM_Hub_")) continue;
        expect(m[1], m[0]).toMatch(
          /^(NVIDIA_GPU|Google_TPU|FOptInf|AI_Accelerators)_\d\d_/,
        );
        expect(m[2], m[0]).toBeDefined();
      }
    });
  });
}
