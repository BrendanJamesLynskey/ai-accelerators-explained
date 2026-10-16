/**
 * The live caption of every animation step: a pure function of the model's
 * state, so a test can rebuild the caption of any step from the Python
 * reference's state and compare it with the page (tests/e2e/frames.spec.ts).
 */
import { fmtBytes, fmtFlops, fmtInt, fmtTime, pct, trim } from "@/lib/format";

import {
  GRAIN_LABELS,
  anyName,
  PLACEMENT_LABELS,
  SCHEDULE_LABELS,
  type ChainStep,
  type FloorMode,
  type FloorRow,
  type PathStep,
  type Choice,
  type DataflowStep,
  type Decode,
  type DefectStep,
  type EdgeStep,
  type ImcLayer,
  type ImcStep,
  type MapStep,
  type MeshStep,
  type MxuStep,
  type SweepStep,
} from "./model";

/** Lower-case a name's first letter, but not an acronym's ("GPUs", "FPGAs"). */
const lower = (s: string) =>
  s.length > 1 && s.charAt(1) === s.charAt(1).toUpperCase()
    ? s
    : s.charAt(0).toLowerCase() + s.slice(1);

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

// --- chapter 6: one layer, kernel by kernel or fused ----------------------------------

export function dataflowCaption(
  s: DataflowStep,
  n: number,
  partName: string,
): string {
  return (
    `Operation ${s.i + 1} of ${n}, ${s.name}, on one ${partName}. ` +
    `Kernel by kernel it moves ${fmtBytes(s.u_op)} to and from HBM (${fmtBytes(s.u_bytes)} so far, ${fmtTime(s.u_t)}); ` +
    `fused into the dataflow pipeline it moves ${fmtBytes(s.f_op)} (${fmtBytes(s.f_bytes)} so far, ${fmtTime(s.f_t)}).`
  );
}

// --- chapter 7: a layer on a mesh of cores ------------------------------------------

export function meshCaption(
  s: MeshStep,
  n: number,
  cards: number,
  cores: number,
  coreSram: number,
): string {
  const head = `Step ${s.i + 1} of ${n}, ${s.name}`;
  if (s.kind === "ethernet")
    return cards === 1
      ? `${head}: one card, so nothing to all-reduce. ${fmtTime(s.t)} so far.`
      : `${head}: the ${cards} cards sum their partial results over Ethernet, ${fmtBytes(s.bytes)} each way per card in ${fmtTime(s.t_op)} (latency not disclosed, not added). ${fmtTime(s.t)} so far.`;
  const what =
    s.kind === "attention"
      ? `each card reads its share of the KV cache, ${fmtBytes(s.bytes)}`
      : `each card streams ${fmtBytes(s.bytes)} of weights from GDDR6 into its ${fmtInt(cores)} cores`;
  const core = `${fmtBytes(s.per_core)} per core against ${fmtBytes(coreSram)} of SRAM`;
  return `${head}: ${what}, ${core}, in ${fmtTime(s.t_op)}. ${fmtTime(s.t)} so far.`;
}

// --- chapter 8: narrower weights on an edge budget --------------------------------------

export function edgeCaption(
  s: EdgeStep,
  n: number,
  deviceName: string,
): string {
  const fit =
    s.fits === null
      ? " (its memory capacity is not disclosed)"
      : s.fits
        ? ""
        : " (more than its memory holds)";
  const energy =
    s.j_per_token === null
      ? ""
      : ` At its stated power that is ${trim(s.j_per_token)} J per token.`;
  return (
    `Width ${s.i + 1} of ${n}: ${s.bits}-bit weights, ${fmtBytes(s.weights)}${fit}. ` +
    `${deviceName} reads ${fmtBytes(s.bytes)} per token: ${fmtTime(s.t)}, ${trim(s.tok_s)} tokens/s for one user.${energy}`
  );
}

// --- chapter 9: a layer on crossbars ----------------------------------------------------

export function imcCaption(s: ImcStep, m: ImcLayer, modelName: string): string {
  if (s.name === "program")
    return (
      `Before inference, ${modelName}'s weights are programmed into the crossbars as conductances: ` +
      `${fmtInt(m.layer_tiles)} tiles of 256 x 256 per layer, ${fmtInt(m.chips)} chips for the whole model. They never move again.`
    );
  if (s.name === "attention")
    return (
      `Attention: the KV cache changes every token, so it cannot sit in a crossbar; this step runs in digital logic ` +
      `(its time is not disclosed and not counted). ${fmtTime(s.t)} of crossbar time so far.`
    );
  return (
    `${s.name}: ${fmtInt(s.tiles)} crossbars each do one analogue matrix-vector multiply at once, ${fmtTime(m.lat)}. ` +
    `${fmtTime(s.t)} of crossbar time and ${trim(s.e * 1e6)} µJ so far in this layer.`
  );
}

// --- chapter 10: every family on the same model -------------------------------------------

export function chooseCaption(
  r: Choice,
  n: number,
  modelName: string,
  partName: string,
): string {
  const chips = `${fmtInt(r.chips)} ${partName}${r.chips > 1 ? "s" : ""}`;
  const energy =
    r.j_per_token === null
      ? "energy not disclosed"
      : `${trim(r.j_per_token)} J per token`;
  const bound =
    r.bound === "crossbar-only" ? "crossbar time only" : `${r.bound}-bound`;
  return (
    `Family ${r.i + 1} of ${n}, ${lower(r.name)}: ${modelName} in ${r.fmt.toUpperCase()} needs ${chips}. ` +
    `Bound per token ${fmtTime(r.t_decode)} (${trim(r.tok_s_user)} tokens/s for one user, ${bound}); ` +
    `a 2,048-token prefill ${fmtTime(r.t_prefill)}; ${energy}.`
  );
}

/** A part's short name: "H100 SXM", "SN40L RDU", "M4". */
export function partShort(pid: string): string {
  return anyName(pid)
    .split(" (")[0]!
    .replace(/^(NVIDIA|Google|Cerebras|SambaNova|Tenstorrent|Apple|IBM) /, "");
}

// --- chapter 10: the per-layer latency floor ---------------------------------------

export function pathCaption(s: PathStep, r: FloorRow, mode: FloorMode): string {
  const floorName =
    mode === "fitted"
      ? `the floor fitted on the other three systems (${fmtTime(r.tauLoo)} a layer)`
      : mode === "own"
        ? `this system's own floor (${fmtTime(r.per_layer)} a layer)`
        : "no floor";
  if (s.layer === 0)
    return `${r.label}: one token starts down ${s.layers} layers, with the bound's work per layer (${fmtTime(r.t_model / s.layers)}) and ${floorName}.`;
  const end =
    s.layer === s.layers
      ? ` The token is out after ${fmtTime(s.t)}; the system published ${fmtTime(r.t_published)}.`
      : "";
  return `Layer ${s.layer} of ${s.layers}: ${fmtTime(s.t)} so far, ${fmtTime(s.work)} of it the bound's work and ${fmtTime(s.floor)} the floor.${end}`;
}
