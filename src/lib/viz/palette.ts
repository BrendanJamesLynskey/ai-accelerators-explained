/**
 * The family's visual language (explained_sites_visual_standard.md §3):
 * one colour-blind-safe palette (Okabe and Ito's eight colours), the same
 * in light and dark mode, with one colour per memory level, used on every
 * site. "Active" is a highlight, "done" is muted, and "stalled" is a
 * warning hue plus a hatch pattern, never colour alone.
 *
 * Okabe, M. and Ito, K. (2008), "Color Universal Design (CUD): how to make
 * figures and presentations that are friendly to colorblind people",
 * https://jfly.uni-koeln.de/color/
 */

export const OKABE_ITO = {
  black: "#000000",
  orange: "#E69F00",
  sky: "#56B4E9",
  green: "#009E73",
  yellow: "#F0E442",
  blue: "#0072B2",
  vermillion: "#D55E00",
  purple: "#CC79A7",
} as const;

/** One colour per memory level, the same on every site in the family. */
export const LEVEL_COLOUR = {
  reg: OKABE_ITO.orange,
  smem: OKABE_ITO.green,
  l2: OKABE_ITO.sky,
  hbm: OKABE_ITO.purple,
} as const;

export const LEVEL_NAME = {
  reg: "Registers",
  smem: "Shared memory",
  l2: "L2 cache",
  hbm: "HBM",
} as const;

/** States of an element in an animation. */
export const STATE_COLOUR = {
  active: OKABE_ITO.blue,
  stalled: OKABE_ITO.vermillion,
  /** compute (ALUs), distinct from every memory level */
  compute: OKABE_ITO.yellow,
} as const;

/** Muted ("done", "idle") greys: Tailwind neutral-400 and neutral-600. */
export const MUTED = { light: "#a3a3a3", dark: "#525252" } as const;

/**
 * This site's addition: where a family keeps its weights, from the same
 * eight colours. Off-chip DRAM keeps the family's HBM colour (purple),
 * on-chip SRAM the shared-memory colour (green), a tiered design the L2
 * colour (sky), and compute inside the memory array the register colour
 * (orange): the nearer the maths, the warmer.
 */
export const PLACEMENT_COLOUR = [
  OKABE_ITO.purple,
  OKABE_ITO.sky,
  OKABE_ITO.green,
  OKABE_ITO.orange,
] as const;

/** Time terms of a step: memory reads, compute, and chip-to-chip hops. */
export const TERM_COLOUR = {
  mem: OKABE_ITO.purple,
  sram: OKABE_ITO.green,
  kv: OKABE_ITO.sky,
  comp: OKABE_ITO.yellow,
  hop: OKABE_ITO.vermillion,
} as const;
