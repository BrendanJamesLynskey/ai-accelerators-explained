/**
 * The live caption of every animation step: a pure function of the model's
 * state, so a test can rebuild the caption of any step from the Python
 * reference's state and compare it with the page (tests/e2e/frames.spec.ts).
 */
import { fmtBytes, fmtFlops, fmtInt, fmtTime, pct, trim } from "@/lib/format";

import {
  GRAIN_LABELS,
  PLACEMENT_LABELS,
  SCHEDULE_LABELS,
  type ChainStep,
  type Decode,
  type DefectStep,
  type MapStep,
  type MxuStep,
  type SweepStep,
} from "./model";

const lower = (s: string) => s.charAt(0).toLowerCase() + s.slice(1);

// --- chapter 1: the map ------------------------------------------------------

export function mapCaption(
  s: MapStep,
  n: number,
  modelName: string,
  partName: string,
): string {
  const head =
    `Family ${s.i + 1} of ${n}, ${lower(s.name)} (chapter ${s.chapter}; e.g. ${s.example}): ` +
    `the weights live in ${PLACEMENT_LABELS[s.placement]}; data movement is scheduled by the ` +
    `${SCHEDULE_LABELS[s.schedule]}; ${GRAIN_LABELS[s.grain]}.`;
  if (!s.part) return `${head} Its numbers come in chapter ${s.chapter}.`;
  const unit = s.where === "sram" ? "on-chip SRAM" : "memory";
  return (
    `${head} ${modelName} in ${s.fmt.toUpperCase()}, weights plus one 1,024-token KV cache, is ${fmtBytes(s.need)}: ` +
    `${fmtInt(s.chips)} ${partName}${s.chips > 1 ? "s" : ""} at ${fmtBytes(s.cap)} of ${unit} each.`
  );
}

// --- chapter 2: the batch sweep -----------------------------------------------

export function sweepCaption(
  s: SweepStep,
  n: number,
  partName: string,
  ridge: number,
): string {
  const where = `Batch ${fmtInt(s.batch)} (${s.i + 1} of ${n}) on ${s.chips} ${partName}${s.chips > 1 ? "s" : ""}`;
  const side =
    s.intensity < ridge
      ? `below the ridge of ${trim(ridge)}, so the step is memory-bound`
      : `above the ridge of ${trim(ridge)}, so the step is compute-bound`;
  const fit = s.fits ? "" : " The batch's KV caches do not fit in memory.";
  return (
    `${where}: each step reads ${fmtBytes(s.bytes)} and does ${fmtFlops(s.flops).replace("/s", "")}, ` +
    `${trim(s.intensity)} flop per byte, ${side}: ${fmtTime(s.t_step)} per step, ` +
    `${trim(s.tok_s)} tokens/s in all, ${trim(s.tok_s_user)} per user.${fit}`
  );
}

// --- chapter 3: tiles on a systolic array -----------------------------------------

export function mxuCaption(
  s: MxuStep,
  tiles: number,
  rows: number,
  cols: number,
  m: number,
): string {
  if (s.tile < 0)
    return (
      `A ${rows} x ${cols} weight-stationary array, ${fmtInt(tiles)} weight tiles to run, ` +
      `a batch of ${m} row${m > 1 ? "s" : ""}. Filling and draining the array costs ${fmtInt(s.cycles)} cycles once.`
    );
  return (
    `Tile ${s.tile + 1} of ${tiles} (row block ${s.tr}, column block ${s.tc}): ` +
    `${m} input row${m > 1 ? "s" : ""} streamed through ${fmtInt(rows * cols)} PEs${s.tile < tiles - 1 ? " while the next tile's weights shift in" : ", the last tile"}. ` +
    `${fmtInt(s.cycles)} cycles so far; ${pct(s.util, 1)} of the array's multiply-accumulates used.`
  );
}

// --- chapter 4: defects on a wafer --------------------------------------------------

export function waferCaption(
  s: DefectStep,
  count: number,
  dies: number,
  cores: number,
  spares: number,
): string {
  if (s.shown === 0)
    return (
      `A wafer of ${dies} dies and ${fmtInt(cores)} cores, with ${fmtInt(count)} defects expected at this defect density. ` +
      `${fmtInt(spares)} cores are held as spares.`
    );
  return (
    `${fmtInt(s.shown)} of ${fmtInt(count)} defects placed, in ${s.dies_hit} of ${dies} dies. ` +
    `Each takes out one core; the fabric routes round it to a spare: ${fmtInt(s.shown)} of ${fmtInt(cores)} cores lost, ` +
    `${fmtInt(spares - s.shown)} spares left.`
  );
}

// --- chapter 5: a token down the chain ------------------------------------------------

export function chainCaption(
  s: ChainStep,
  d: Decode,
  partName: string,
  hop: number | null,
): string {
  const of = `Stage ${s.stage + 1} of ${d.stages} (${d.group} ${partName}${d.group > 1 ? "s" : ""})`;
  const read = `reads its share of the weights from SRAM in ${fmtTime(s.work_end - s.start)}`;
  const reduce =
    s.done > s.work_end
      ? `, finishes its layers' all-reduces at ${fmtTime(s.done)}`
      : "";
  const next =
    s.stage < d.stages - 1
      ? hop === null
        ? `, then passes the activation to stage ${s.stage + 2} (hop latency not disclosed)`
        : `, then hops ${fmtTime(hop)} to stage ${s.stage + 2}`
      : `: the token is out after ${fmtTime(s.done)}`;
  return `${of} ${read}${reduce}${next}.`;
}
