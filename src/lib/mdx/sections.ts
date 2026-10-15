/**
 * Chapter catalogue + filesystem loader for /learn content.
 *
 * MDX sources live under `/content/chapters/`, one per family of chips,
 * each built around an animation. Their slugs and order are defined here (single source
 * of truth); the `[slug]` route validates incoming params against this list
 * before reading from disk. Same shape as the companion sites'
 * `src/lib/mdx/sections.ts`.
 */
import { readFile } from "node:fs/promises";
import { join } from "node:path";

export const SECTIONS = [
  {
    slug: "01-the-design-space",
    title: "The design space",
    summary:
      "Where the weights live, who decides when data moves, and how big the engines are: a map of the families of AI chips, and the decoder layer every chapter runs.",
  },
  {
    slug: "02-gpus-the-baseline",
    title: "GPUs: the baseline",
    summary:
      "HBM, tensor cores and the roofline: why one decode step on a GPU is a memory read, and how batching turns it into a compute problem.",
  },
  {
    slug: "03-systolic-designs",
    title: "Systolic designs",
    summary:
      "TPUs and Trainium: large weight-stationary arrays fed from HBM by a compiler, and why decode needs a batch as big as the array to fill it.",
  },
  {
    slug: "04-wafer-scale",
    title: "Wafer-scale",
    summary:
      "A whole wafer as one chip: the weights in on-chip SRAM, models split across wafers at layer boundaries, and how defects are survived rather than avoided.",
  },
  {
    slug: "05-compiler-scheduled",
    title: "Compiler-scheduled chips",
    summary:
      "No caches and no DRAM: a static schedule over SRAM-only chips chained into a pipeline, why one token comes back fast, and what holding a model in SRAM costs.",
  },
  {
    slug: "06-reconfigurable-dataflow",
    title: "Reconfigurable dataflow",
    summary:
      "SambaNova's RDU and Graphcore's IPU: a layer fused into one on-chip pipeline instead of a kernel per operation, three tiers of memory, and bulk-synchronous tiles.",
  },
  {
    slug: "07-meshes-of-cores",
    title: "Meshes of small cores",
    summary:
      "Tenstorrent's Blackhole: many small cores with their own SRAM on a network-on-chip, GDDR6 beside them, and Ethernet built into the chip to join cards.",
  },
  {
    slug: "08-fpgas-and-npus",
    title: "FPGAs and edge NPUs",
    summary:
      "A few watts and a phone's memory: why one user's decode is a bandwidth problem, how narrower weights help, and where FPGAs win on latency and custom numerics.",
  },
  {
    slug: "09-in-memory-compute",
    title: "In-memory and analogue compute",
    summary:
      "Weights that never move: a decoder layer on phase-change crossbars, and what it costs in capacity, ADCs and precision.",
  },
  {
    slug: "10-choosing",
    title: "Choosing",
    summary:
      "The same decoder model on every family, side by side: decode and prefill bounds, energy, chips and utilisation, where each wins and loses, and what they are converging on.",
  },
] as const;

export type SectionSlug = (typeof SECTIONS)[number]["slug"];

const SLUG_SET = new Set<string>(SECTIONS.map((s) => s.slug));

export function isValidSlug(slug: string): slug is SectionSlug {
  return SLUG_SET.has(slug);
}

export function getSectionMeta(slug: SectionSlug): (typeof SECTIONS)[number] {
  return SECTIONS.find((s) => s.slug === slug) ?? SECTIONS[0];
}

/** Read the raw MDX source for a chapter, or `null` if it doesn't exist. */
export async function readSectionMdx(
  slug: SectionSlug,
): Promise<string | null> {
  const path = join(process.cwd(), "content", "chapters", `${slug}.mdx`);
  try {
    return await readFile(path, "utf-8");
  } catch {
    return null;
  }
}
