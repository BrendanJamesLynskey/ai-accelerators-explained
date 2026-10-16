/**
 * The accelerator model: the TypeScript port of `accel_model/core.py`.
 *
 * Every function repeats its Python counterpart with the same operations in
 * the same order, so that `tests/unit/model.test.ts` can require the port to
 * reproduce the Python fixtures exactly (IEEE doubles, no tolerance), except
 * `waferYield`, whose exp() is compared to 1e-12. The hardware figures are
 * read from `src/data/accel.json`, which the Python script writes, so both
 * sides start from identical values.
 */
import data from "@/data/accel.json";

export type Status =
  | "spec"
  | "derived"
  | "reported"
  | "illustrative"
  | "undisclosed";
export type Avail = "product" | "announced" | "retired" | "research";
export type Field<T = number | string | null> = {
  v: T;
  st: Status;
  src: string;
  ref: string;
};
export type Source = {
  title: string;
  url: string;
  kind: string;
  accessed: string;
};
export type PartRecord = {
  id: string;
  name: string;
  family: string;
  layout: "tensor" | "pipeline";
  avail: Field;
  fields: Record<string, Field>;
};
export type FamilyRecord = {
  id: string;
  name: string;
  chapter: number;
  example: string;
  also?: string[];
  placement: Field<number>;
  schedule: Field<number>;
  grain: Field<number>;
};
export type ModelRecord = {
  id: string;
  name: string;
  src: string;
  layers: number;
  d: number;
  ffn: number;
  heads: number;
  kv_heads: number;
  vocab: number;
};

export const ACCESSED = data.accessed;
export const SOURCES = data.sources as Record<string, Source>;
export const FAMILIES = data.families as unknown as FamilyRecord[];
export const PLACEMENT_LABELS = data.placementLabels;
export const SCHEDULE_LABELS = data.scheduleLabels;
export const GRAIN_LABELS = data.grainLabels;
export const PART_RECORDS = data.parts as unknown as PartRecord[];
export const MODEL_RECORDS = data.models as unknown as ModelRecord[];
export const FORMATS = data.formats as Record<Fmt, number>;
export const PUBLISHED = data.published as unknown as Record<
  string,
  Field<number> & { also?: string[] }
>;
export const CLAIMS = data.claims as Record<
  string,
  { text: string; src: string }
>;
export const ILLUSTRATIVE = data.illustrative as unknown as Record<
  string,
  Field<number>
>;
export const MAP_PARTS = data.mapParts as unknown as Record<
  string,
  [PartId, Fmt]
>;
export const BATCHES = data.batches;
export const DIE_COLS = data.dieCols;
export const DIE_ROWS = data.dieRows;
export const CHAIN_CASES = data.chainCases as unknown as ChainCase[];
export const CHAIN_CTX = data.chainCtx;
export const GROQ_CTX = data.groqCtx;
export const CEREBRAS_CTX = data.cerebrasCtx;
export const SN40L_CTX = data.sn40lCtx;
/** Parts quoted but not modelled (a figure the model needs is not disclosed). */
export const OTHER_PART_RECORDS = data.otherParts as unknown as (Omit<
  PartRecord,
  "layout"
> & { also?: string[] })[];
/** The in-memory compute chip of chapter 9. */
export const IMC_RECORDS = data.imc as unknown as Record<
  string,
  Omit<PartRecord, "layout">
>;
export const CHOICE = data.choice as unknown as Record<string, [string, Fmt]>;
export const BITS = data.bits;
export const EDGE_DEVICES = data.edgeDevices;
export const ACT_BYTES = data.actBytes;

export type PartId =
  | "h100"
  | "b200"
  | "mi355x"
  | "tpu-v6e"
  | "tpu-7x"
  | "trn2"
  | "wse-3"
  | "wse-3t"
  | "groqchip"
  | "groq3"
  | "sn40l"
  | "gc2"
  | "blackhole"
  | "m4";
export type ModelId = "llama3-8b" | "llama3-70b" | "llama2-70b";
export type Fmt = "bf16" | "fp8" | "int8";
export type ChainCase = {
  key: string;
  part: PartId;
  model: ModelId;
  fmt: Fmt;
  chips: number;
  group: number;
};

export const PART_ORDER = PART_RECORDS.map((p) => p.id as PartId);
export const MODEL_ORDER = MODEL_RECORDS.map((m) => m.id as ModelId);

// ---------------------------------------------------------------------------
// Records
// ---------------------------------------------------------------------------

function rec(pid: string): PartRecord {
  const p = PART_RECORDS.find((r) => r.id === pid);
  if (!p) throw new Error(`unknown part ${pid}`);
  return p;
}

/** A part's values, flattened. Undisclosed figures are null. */
export function part(pid: PartId): Record<string, number | string | null> {
  const p = rec(pid);
  const out: Record<string, number | string | null> = {
    id: p.id,
    name: p.name,
    family: p.family,
    layout: p.layout,
    avail: p.avail.v,
  };
  for (const [k, f] of Object.entries(p.fields)) out[k] = f.v;
  return out;
}

/** One field's value, or null when the part has no such field. */
export function fval(pid: string, key: string): number | null {
  const f = rec(pid).fields[key];
  return f === undefined ? null : (f.v as number | null);
}

export type Shape = {
  id: ModelId;
  name: string;
  layers: number;
  d: number;
  ffn: number;
  heads: number;
  kv_heads: number;
  head_dim: number;
  vocab: number;
  per_layer: number;
  params: number;
  matmul_params: number;
  kv_elems_per_token: number;
  attn_flops_per_pos: number;
};

export function shape(mid: ModelId): Shape {
  const m = MODEL_RECORDS.find((r) => r.id === mid);
  if (!m) throw new Error(`unknown model ${mid}`);
  const L = m.layers,
    d = m.d,
    ffn = m.ffn,
    h = m.heads,
    kv = m.kv_heads,
    V = m.vocab;
  const hd = Math.floor(d / h);
  const attn = d * d + 2 * d * (kv * hd) + d * d;
  const mlp = 3 * d * ffn;
  const perLayer = attn + mlp;
  const embed = V * d;
  const params = 2 * embed + L * perLayer;
  const matmul = params - embed;
  return {
    id: mid,
    name: m.name,
    layers: L,
    d,
    ffn,
    heads: h,
    kv_heads: kv,
    head_dim: hd,
    vocab: V,
    per_layer: perLayer,
    params,
    matmul_params: matmul,
    kv_elems_per_token: 2 * L * kv * hd,
    attn_flops_per_pos: 4 * L * d,
  };
}

export type Basis = "dense" | "sparse" | "none";
export type Peak = { flops: number | null; basis: Basis; key: string };

export function peak(pid: string, fmt: Fmt): Peak {
  if (fmt === "bf16") {
    const order: [string, Basis][] = [
      ["peak_bf16", "dense"],
      ["peak_fp16", "dense"],
      ["peak_fp16_sparse", "sparse"],
    ];
    for (const [key, basis] of order) {
      const v = fval(pid, key);
      if (v !== null) return { flops: v, basis, key };
    }
    return { flops: null, basis: "none", key: "" };
  }
  const key = fmt === "fp8" ? "peak_fp8" : "peak_int8";
  const v = fval(pid, key);
  return {
    flops: v,
    basis: v !== null ? "dense" : "none",
    key: v !== null ? key : "",
  };
}

export function kvBytesPerElem(fmt: Fmt): number {
  return Math.max(1.0, FORMATS[fmt]);
}

// ---------------------------------------------------------------------------
// Where the weights live
// ---------------------------------------------------------------------------

export type Placement = {
  weights: number;
  weights_read: number;
  kv_per_token: number;
  kv: number;
  need: number;
  where: "off-chip" | "sram";
  cap: number;
  bw: number;
  chips: number;
  fill: number;
};

export function placement(
  pid: PartId,
  mid: ModelId,
  fmt: Fmt,
  batch: number,
  ctx: number,
): Placement {
  const s = shape(mid);
  const b = FORMATS[fmt];
  const weights = s.params * b;
  const weightsRead = s.matmul_params * b;
  const kvTok = s.kv_elems_per_token * kvBytesPerElem(fmt);
  const kv = batch * ctx * kvTok;
  const need = weights + kv;
  const off = fval(pid, "off_bytes");
  let where: "off-chip" | "sram";
  let cap: number;
  let bw: number;
  if (off !== null && off > 0) {
    where = "off-chip";
    cap = off;
    bw = fval(pid, "off_bw") as number;
  } else {
    where = "sram";
    cap = fval(pid, "sram_bytes") as number;
    bw = fval(pid, "sram_bw") as number;
  }
  const chips = Math.ceil(need / cap);
  return {
    weights,
    weights_read: weightsRead,
    kv_per_token: kvTok,
    kv,
    need,
    where,
    cap,
    bw,
    chips,
    fill: need / (chips * cap),
  };
}

// ---------------------------------------------------------------------------
// Decode and prefill
// ---------------------------------------------------------------------------

export function layoutOf(
  pid: string,
  chips: number,
  group: number,
): { stages: number; group: number } {
  if (rec(pid).layout === "tensor") return { stages: 1, group: chips };
  return { stages: Math.ceil(chips / group), group };
}

export type Bound = "memory" | "compute" | "latency";
export type Decode = {
  chips: number;
  stages: number;
  group: number;
  where: "off-chip" | "sram";
  bytes: number;
  flops: number;
  t_mem: number;
  t_comp: number | null;
  hops: number;
  t_hops: number;
  hop_known: boolean;
  t_step: number;
  bound: Bound;
  basis: Basis;
  tok_s_user: number;
  tok_s: number;
  j_per_token: number | null;
  intensity: number;
};

export function decode(
  pid: PartId,
  mid: ModelId,
  fmt: Fmt,
  batch: number,
  ctx: number,
  chips = 0,
  group = 1,
): Decode {
  const s = shape(mid);
  const pl = placement(pid, mid, fmt, batch, ctx);
  const n = Math.max(chips, pl.chips);
  const lay = layoutOf(pid, n, group);
  const g = lay.group;
  const st = lay.stages;
  const bytesStep = pl.weights_read + pl.kv;
  const tMem = bytesStep / (g * pl.bw);
  const flops =
    2 * s.matmul_params * batch + s.attn_flops_per_pos * ctx * batch;
  const pk = peak(pid, fmt);
  const tComp = pk.flops === null ? null : flops / (g * pk.flops);
  const hop = fval(pid, "hop_s");
  const hops = st - 1 + (g > 1 ? 2 * s.layers : 0);
  const tHops = hop === null ? 0.0 : hops * hop;
  const tWork = tComp === null || tMem >= tComp ? tMem : tComp;
  const tStep = tWork + tHops;
  let bound: Bound;
  if (tHops > tWork) bound = "latency";
  else if (tComp !== null && tComp > tMem) bound = "compute";
  else bound = "memory";
  const tokS = (batch * st) / tStep;
  const pw = fval(pid, "power_w");
  return {
    chips: n,
    stages: st,
    group: g,
    where: pl.where,
    bytes: bytesStep,
    flops,
    t_mem: tMem,
    t_comp: tComp,
    hops,
    t_hops: tHops,
    hop_known: hop !== null,
    t_step: tStep,
    bound,
    basis: pk.basis,
    tok_s_user: 1 / tStep,
    tok_s: tokS,
    j_per_token: pw === null ? null : (n * pw) / tokS,
    intensity: flops / bytesStep,
  };
}

export type Prefill = {
  chips: number;
  t_mem: number;
  t_comp: number | null;
  t: number;
  bound: "memory" | "compute";
  basis: Basis;
  flops: number;
  tok_s: number;
};

export function prefill(
  pid: PartId,
  mid: ModelId,
  fmt: Fmt,
  prompt: number,
  chips = 0,
  group = 1,
): Prefill {
  const s = shape(mid);
  const pl = placement(pid, mid, fmt, 1, prompt);
  const n = Math.max(chips, pl.chips);
  const lay = layoutOf(pid, n, group);
  const g = lay.group;
  const tMem = (pl.weights_read + pl.kv) / (g * pl.bw);
  const flops =
    2 * s.matmul_params * prompt +
    s.attn_flops_per_pos * ((prompt * prompt) / 2);
  const pk = peak(pid, fmt);
  const tComp = pk.flops === null ? null : flops / (g * pk.flops);
  const t = tComp === null || tMem >= tComp ? tMem : tComp;
  return {
    chips: n,
    t_mem: tMem,
    t_comp: tComp,
    t,
    bound: tComp !== null && tComp > tMem ? "compute" : "memory",
    basis: pk.basis,
    flops,
    tok_s: prompt / t,
  };
}

export function ridge(pid: PartId, fmt: Fmt): number | null {
  const bw = fval(pid, "off_bw") || fval(pid, "sram_bw");
  const pk = peak(pid, fmt).flops;
  return pk === null || !bw ? null : pk / bw;
}

export function throughputCeiling(
  pid: PartId,
  mid: ModelId,
  fmt: Fmt,
  chips: number,
): number {
  const s = shape(mid);
  return (chips * (peak(pid, fmt).flops as number)) / (2 * s.matmul_params);
}

/** `peak` for bf16/fp8/int8; for "fp4" the part's dense 4-bit rate, if stated. */
export function peakAny(pid: PartId, fmt: Fmt | "fp4"): number | null {
  if (fmt === "fp4") return fval(pid, "peak_fp4");
  return peak(pid, fmt).flops;
}

export function sampleFlops(
  mid: ModelId,
  nIn: number,
  nOut: number,
): { prefill: number; decode: number; total: number } {
  const s = shape(mid);
  const w = 2 * s.matmul_params;
  const a = s.attn_flops_per_pos;
  const prefill = w * nIn + a * ((nIn * nIn) / 2);
  const decode = w * nOut + a * nOut * (nIn + nOut / 2);
  return { prefill, decode, total: prefill + decode };
}

export type MlperfCeiling = {
  ceiling: number;
  decodeOnly: number;
  prefillShare: number;
  flops: { prefill: number; decode: number; total: number };
  peak: number;
  chips: number;
};

export function mlperfCeiling(
  pid: PartId,
  mid: ModelId,
  fmt: Fmt | "fp4",
  chips: number,
  nIn: number,
  nOut: number,
): MlperfCeiling {
  const pk = peakAny(pid, fmt) as number;
  const f = sampleFlops(mid, nIn, nOut);
  const full = ((chips * pk) / f.total) * nOut;
  const decodeOnly = (chips * pk) / (2 * shape(mid).matmul_params);
  return {
    ceiling: full,
    decodeOnly,
    prefillShare: f.prefill / f.total,
    flops: f,
    peak: pk,
    chips,
  };
}

// ---------------------------------------------------------------------------
// Chapter 1: the map
// ---------------------------------------------------------------------------

export type MapFamily = {
  i: number;
  id: string;
  name: string;
  chapter: number;
  example: string;
  placement: number;
  schedule: number;
  grain: number;
};

export function familyMap(): MapFamily[] {
  return FAMILIES.map((f, i) => ({
    i,
    id: f.id,
    name: f.name,
    chapter: f.chapter,
    example: f.example,
    placement: f.placement.v,
    schedule: f.schedule.v,
    grain: f.grain.v,
  }));
}

export type MapStep = MapFamily & {
  part: string;
  fmt: string;
  chips: number;
  need: number;
  where: string;
  cap: number;
};

export function mapSteps(mid: ModelId, batch: number, ctx: number): MapStep[] {
  return familyMap().map((f) => {
    const mp = MAP_PARTS[f.id];
    if (mp) {
      const [pid, fmt] = mp;
      const pl = placement(pid, mid, fmt, batch, ctx);
      return {
        ...f,
        part: pid,
        fmt,
        chips: pl.chips,
        need: pl.need,
        where: pl.where,
        cap: pl.cap,
      };
    }
    return { ...f, part: "", fmt: "", chips: 0, need: 0, where: "", cap: 0 };
  });
}

// ---------------------------------------------------------------------------
// Chapter 2: the batch sweep
// ---------------------------------------------------------------------------

export type SweepStep = Decode & { i: number; batch: number; fits: boolean };

export function maxBatch(
  pid: PartId,
  mid: ModelId,
  fmt: Fmt,
  ctx: number,
  chips: number,
): number {
  const pl = placement(pid, mid, fmt, 1, ctx);
  const free = chips * pl.cap - pl.weights;
  return free <= 0 ? 0 : Math.floor(free / (ctx * pl.kv_per_token));
}

export function batchSweep(
  pid: PartId,
  mid: ModelId,
  fmt: Fmt,
  ctx: number,
  chips: number,
): SweepStep[] {
  const cap = maxBatch(pid, mid, fmt, ctx, chips);
  return BATCHES.map((b, i) => ({
    ...decode(pid, mid, fmt, b, ctx, chips),
    i,
    batch: b,
    fits: b <= cap,
  }));
}

// ---------------------------------------------------------------------------
// Chapter 3: systolic arrays
// ---------------------------------------------------------------------------

export type Gemm = { name: string; k: number; n: number };

export function layerGemms(mid: ModelId): Gemm[] {
  const s = shape(mid);
  const d = s.d,
    kvd = s.kv_heads * s.head_dim;
  return [
    { name: "QKV", k: d, n: d + 2 * kvd },
    { name: "attention out", k: d, n: d },
    { name: "MLP up and gate", k: d, n: 2 * s.ffn },
    { name: "MLP down", k: s.ffn, n: d },
  ];
}

export type Mxu = {
  tiles: number;
  per_tile: number;
  cycles: number;
  macs: number;
  util: number;
};

export function mxu(
  rows: number,
  cols: number,
  m: number,
  k: number,
  n: number,
): Mxu {
  const tiles = Math.ceil(k / rows) * Math.ceil(n / cols);
  const perTile = Math.max(m, rows);
  const cycles = tiles * perTile + rows + cols - 2;
  const macs = m * k * n;
  return {
    tiles,
    per_tile: perTile,
    cycles,
    macs,
    util: macs / (rows * cols * cycles),
  };
}

export function layerMxu(
  rows: number,
  cols: number,
  mid: ModelId,
  m: number,
): { cycles: number; macs: number; util: number } {
  let cyc = 0;
  let macs = 0;
  for (const g of layerGemms(mid)) {
    const r = mxu(rows, cols, m, g.k, g.n);
    cyc += r.cycles;
    macs += r.macs;
  }
  return { cycles: cyc, macs, util: macs / (rows * cols * cyc) };
}

export type MxuStep = {
  i: number;
  tile: number;
  tr: number;
  tc: number;
  cycles: number;
  macs: number;
  util: number;
};

export function mxuSteps(
  rows: number,
  cols: number,
  m: number,
  k: number,
  n: number,
): MxuStep[] {
  const tr = Math.ceil(k / rows);
  const tc = Math.ceil(n / cols);
  const per = Math.max(m, rows);
  const out: MxuStep[] = [
    {
      i: 0,
      tile: -1,
      tr: -1,
      tc: -1,
      cycles: rows + cols - 2,
      macs: 0,
      util: 0.0,
    },
  ];
  let macs = 0;
  for (let t = 0; t < tr * tc; t++) {
    const a = Math.floor(t / tc);
    const c = t % tc;
    const kk = Math.min(rows, k - a * rows);
    const nn = Math.min(cols, n - c * cols);
    macs += m * kk * nn;
    const cycles = (t + 1) * per + rows + cols - 2;
    out.push({
      i: t + 1,
      tile: t,
      tr: a,
      tc: c,
      cycles,
      macs,
      util: macs / (rows * cols * cycles),
    });
  }
  return out;
}

// ---------------------------------------------------------------------------
// Chapter 4: wafer-scale
// ---------------------------------------------------------------------------

export function poissonZero(lam: number): number {
  return Math.exp(-lam);
}

export type WaferYield = {
  lam_wafer: number;
  lam_die: number;
  die_yield: number;
  wafer_perfect: number;
  spares: number;
  lost_share: number;
  spare_margin: number;
  die_cores: number;
};

export function waferYield(
  pid: PartId,
  d0PerCm2: number,
  dieMm2: number,
  spareFrac: number,
): WaferYield {
  const area = fval(pid, "area_mm2") as number;
  const cores = fval(pid, "units") as number;
  const lamWafer = (d0PerCm2 * area) / 100;
  const lamDie = (d0PerCm2 * dieMm2) / 100;
  const spares = Math.floor(cores * spareFrac);
  return {
    lam_wafer: lamWafer,
    lam_die: lamDie,
    die_yield: poissonZero(lamDie),
    wafer_perfect: poissonZero(lamWafer),
    spares,
    lost_share: lamWafer / cores,
    spare_margin: spares / lamWafer,
    die_cores: cores / (fval(pid, "dies") as number),
  };
}

export function xorshift32(x: number): number {
  x = (x ^ ((x << 13) >>> 0)) >>> 0;
  x = (x ^ (x >>> 17)) >>> 0;
  x = (x ^ ((x << 5) >>> 0)) >>> 0;
  return x >>> 0;
}

export type Defect = {
  i: number;
  col: number;
  row: number;
  u: number;
  v: number;
};

export function defects(seed: number, count: number): Defect[] {
  let x = seed;
  const out: Defect[] = [];
  for (let i = 0; i < count; i++) {
    x = xorshift32(x);
    const die = x % (DIE_COLS * DIE_ROWS);
    x = xorshift32(x);
    const cell = x % 100;
    out.push({
      i,
      col: die % DIE_COLS,
      row: Math.floor(die / DIE_COLS),
      u: cell % 10,
      v: Math.floor(cell / 10),
    });
  }
  return out;
}

export type DefectStep = { i: number; shown: number; dies_hit: number };

export function defectSteps(
  seed: number,
  count: number,
  perStep: number,
): DefectStep[] {
  const ds = defects(seed, count);
  const out: DefectStep[] = [];
  let k = 0;
  for (;;) {
    const shown = Math.min(k * perStep, count);
    const hit = new Set(ds.slice(0, shown).map((d) => `${d.col},${d.row}`));
    out.push({ i: k, shown, dies_hit: hit.size });
    if (shown === count) break;
    k += 1;
  }
  return out;
}

export type WafersNeeded = {
  wafers: number;
  need: number;
  fill: number;
  io: number;
  t_stream: number;
  stream_tok_s: number;
};

export function wafersNeeded(
  pid: PartId,
  mid: ModelId,
  fmt: Fmt,
  batch: number,
  ctx: number,
): WafersNeeded {
  const pl = placement(pid, mid, fmt, batch, ctx);
  const io = (fval(pid, "io_bps") as number) / 8;
  const tStream = pl.weights_read / io;
  return {
    wafers: pl.chips,
    need: pl.need,
    fill: pl.fill,
    io,
    t_stream: tStream,
    stream_tok_s: 1 / tStream,
  };
}

// ---------------------------------------------------------------------------
// Chapter 5: chains of compiler-scheduled chips
// ---------------------------------------------------------------------------

export type ChainStep = {
  i: number;
  stage: number;
  start: number;
  work_end: number;
  done: number;
  next: number;
};

export function chainSteps(
  pid: PartId,
  mid: ModelId,
  fmt: Fmt,
  ctx: number,
  chips: number,
  group: number,
): ChainStep[] {
  const d = decode(pid, mid, fmt, 1, ctx, chips, group);
  const st = d.stages;
  const s = shape(mid);
  const work = d.t_comp === null || d.t_mem >= d.t_comp ? d.t_mem : d.t_comp;
  const stageWork = work / st;
  const hop = fval(pid, "hop_s") || 0.0;
  const stageReduce = (d.group > 1 ? 2 * s.layers * hop : 0.0) / st;
  const out: ChainStep[] = [];
  let t = 0.0;
  for (let k = 0; k < st; k++) {
    const start = t;
    t = t + stageWork + stageReduce;
    const done = t;
    if (k < st - 1) t = t + hop;
    out.push({
      i: k,
      stage: k,
      start,
      work_end: start + stageWork,
      done,
      next: t,
    });
  }
  return out;
}

export function lpuCapacity(
  pid: PartId,
  mid: ModelId,
  fmt: Fmt,
  ctx: number,
): { chips: number; need: number; racks: number | null } {
  const pl = placement(pid, mid, fmt, 1, ctx);
  const perRack = fval(pid, "rack_chips");
  return {
    chips: pl.chips,
    need: pl.need,
    racks: perRack === null ? null : Math.ceil(pl.chips / perRack),
  };
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

export type BandKind = "capacity" | "throughput" | "latency";
export type Check = {
  id: string;
  kind: BandKind | "speculative";
  label: string;
  published: number;
  model: number;
  ratio: number;
  src: string;
};

export function validation(): Check[] {
  const P = (k: string) => PUBLISHED[k]!.v;
  const rows: Check[] = [];
  const add = (
    id: string,
    kind: Check["kind"],
    label: string,
    pub: number,
    model: number,
    src: string,
  ) =>
    rows.push({
      id,
      kind,
      label,
      published: pub,
      model,
      ratio: pub / model,
      src,
    });

  const cap70 = placement("wse-3", "llama3-70b", "bf16", 1, CEREBRAS_CTX).chips;
  add(
    "cerebras-70b-wafers",
    "capacity",
    "Llama 3.1 70B, 16-bit: CS-3 systems needed",
    P("cerebras-70b-systems"),
    cap70,
    "cerebras-inference",
  );
  add(
    "cerebras-20b-wafer",
    "capacity",
    "a 20B model, 16-bit: CS-3 systems needed (weights only)",
    P("cerebras-20b-one"),
    Math.ceil((20e9 * 2) / (fval("wse-3", "sram_bytes") as number)),
    "cerebras-inference",
  );
  const capg = placement("groqchip", "llama2-70b", "int8", 1, GROQ_CTX).chips;
  add(
    "groq-70b-chips",
    "capacity",
    "Llama 2 70B, INT8: GroqChips used vs the fewest that hold it",
    P("groq-70b-chips"),
    capg,
    "nextplatform-groq",
  );
  const nIn = P("mlperf-input-mean");
  const nOut = P("mlperf-tokens-per-sample");
  const ceilH100 = mlperfCeiling(
    "h100",
    "llama2-70b",
    "fp8",
    8,
    nIn,
    nOut,
  ).ceiling;
  add(
    "mlperf-h100",
    "throughput",
    "8x H100, Llama 2 70B, FP8, MLPerf Offline tokens/s vs the compute ceiling (prefill included)",
    P("mlperf-h100"),
    ceilH100,
    "mlperf-h100",
  );
  const ceilGb200 = mlperfCeiling(
    "b200",
    "llama2-70b",
    "fp4",
    P("mlperf-gb200-chips"),
    nIn,
    nOut,
  ).ceiling;
  add(
    "mlperf-gb200",
    "throughput",
    "4x GB200 (Blackwell), Llama 2 70B, FP4, MLPerf Offline tokens/s vs the compute ceiling (prefill included)",
    P("mlperf-gb200"),
    ceilGb200,
    "mlperf-gb200",
  );
  const g = decode(
    "groqchip",
    "llama2-70b",
    "int8",
    1,
    GROQ_CTX,
    P("groq-70b-chips"),
    fval("groqchip", "node_chips") as number,
  );
  add(
    "groq-70b-tps",
    "latency",
    "576 GroqChips, Llama 2 70B: tokens/s per user vs the bound",
    P("groq-70b-tps"),
    g.tok_s_user,
    "nextplatform-groq",
  );
  const c70 = decode("wse-3", "llama3-70b", "bf16", 1, CEREBRAS_CTX);
  add(
    "cerebras-70b-tps",
    "latency",
    "4x CS-3, Llama 3.1 70B: tokens/s per user vs the bound",
    P("cerebras-70b-tps"),
    c70.tok_s_user,
    "cerebras-inference",
  );
  const c8 = decode("wse-3", "llama3-8b", "bf16", 1, CEREBRAS_CTX);
  add(
    "cerebras-8b-tps",
    "latency",
    "1x CS-3, Llama 3.1 8B: tokens/s per user vs the bound",
    P("cerebras-8b-tps"),
    c8.tok_s_user,
    "cerebras-inference",
  );
  const n = P("sn40l-sockets");
  const s8 = decode("sn40l", "llama3-8b", "bf16", 1, SN40L_CTX, n);
  add(
    "sn40l-8b-tps",
    "latency",
    "16 SN40L sockets, Llama 3.1 8B, BF16: tokens/s per user vs the bound",
    P("sn40l-8b-tps"),
    s8.tok_s_user,
    "sn40l",
  );
  const s70 = decode("sn40l", "llama3-70b", "bf16", 1, SN40L_CTX, n);
  add(
    "sn40l-70b-tps",
    "speculative",
    "16 SN40L sockets, Llama 3.1 70B, BF16, speculative decoding: tokens/s per user vs the one-token-per-step bound",
    P("sn40l-70b-tps"),
    s70.tok_s_user,
    "sn40l",
  );
  return rows;
}

const OVERHEAD_MODEL: Record<string, ModelId> = {
  "groq-70b-tps": "llama2-70b",
  "cerebras-70b-tps": "llama3-70b",
  "cerebras-8b-tps": "llama3-8b",
  "sn40l-8b-tps": "llama3-8b",
};

export function impliedOverhead(checkId: string): {
  t_published: number;
  t_model: number;
  extra: number;
  per_layer: number;
} {
  const v = validation().find((r) => r.id === checkId);
  const mid = OVERHEAD_MODEL[checkId];
  if (!v || !mid) throw new Error(`no speed check ${checkId}`);
  const tPub = 1 / v.published;
  const tMod = 1 / v.model;
  const layers = shape(mid).layers;
  return {
    t_published: tPub,
    t_model: tMod,
    extra: tPub - tMod,
    per_layer: (tPub - tMod) / layers,
  };
}

// ---------------------------------------------------------------------------
// The per-layer latency floor of the SRAM machines (chapters 4, 5 and 10)
// ---------------------------------------------------------------------------

export const FLOOR_CHECKS = [
  "groq-70b-tps",
  "cerebras-70b-tps",
  "cerebras-8b-tps",
  "sn40l-8b-tps",
] as const;
export type FloorCheck = (typeof FLOOR_CHECKS)[number];
export const FLOOR_LABEL: Record<FloorCheck, string> = {
  "groq-70b-tps": "Groq, 576 chips, Llama 2 70B",
  "cerebras-70b-tps": "Cerebras, 4 CS-3, Llama 3.1 70B",
  "cerebras-8b-tps": "Cerebras, 1 CS-3, Llama 3.1 8B",
  "sn40l-8b-tps": "SambaNova, 16 SN40L, Llama 3.1 8B",
};

export type FloorRow = {
  id: FloorCheck;
  label: string;
  layers: number;
  t_published: number;
  t_model: number;
  extra: number;
  per_layer: number;
  tauLoo: number;
  tPred: number;
  ratioPred: number;
  ratioBound: number;
  floorShare: number;
};
export type FloorFit = {
  tau: number;
  min: number;
  max: number;
  rows: FloorRow[];
  worstPred: number;
  worstBound: number;
};

export function floorFit(): FloorFit {
  const rows: FloorRow[] = [];
  for (const cid of FLOOR_CHECKS) {
    const o = impliedOverhead(cid);
    const layers = shape(OVERHEAD_MODEL[cid]!).layers;
    rows.push({
      id: cid,
      label: FLOOR_LABEL[cid],
      layers,
      ...o,
      tauLoo: 0,
      tPred: 0,
      ratioPred: 0,
      ratioBound: 0,
      floorShare: 0,
    });
  }
  const n = rows.length;
  let sum = 0;
  for (const r of rows) sum += r.per_layer;
  const tau = sum / n;
  for (const r of rows) {
    let o = 0;
    let k = 0;
    for (const x of rows)
      if (x.id !== r.id) {
        o += x.per_layer;
        k += 1;
      }
    const tauLoo = o / k;
    const tPred = r.t_model + r.layers * tauLoo;
    r.tauLoo = tauLoo;
    r.tPred = tPred;
    r.ratioPred = r.t_published / tPred;
    r.ratioBound = r.t_published / r.t_model;
    r.floorShare = (r.layers * tauLoo) / tPred;
  }
  let lo = Infinity;
  let hi = -Infinity;
  let worstPred = 0;
  let worstBound = 0;
  for (const r of rows) {
    lo = Math.min(lo, r.per_layer);
    hi = Math.max(hi, r.per_layer);
    worstPred = Math.max(worstPred, Math.max(r.ratioPred, 1 / r.ratioPred));
    worstBound = Math.max(worstBound, r.ratioBound);
  }
  return { tau, min: lo, max: hi, rows, worstPred, worstBound };
}

export type FloorMode = "fitted" | "own" | "none";
export type PathStep = {
  i: number;
  layer: number;
  t: number;
  work: number;
  floor: number;
  layers: number;
};

export function criticalPathSteps(
  checkId: FloorCheck,
  mode: FloorMode,
): PathStep[] {
  const fit = floorFit();
  const r = fit.rows.find((x) => x.id === checkId)!;
  const L = r.layers;
  const work = r.t_model / L;
  const floor = { fitted: r.tauLoo, own: r.per_layer, none: 0.0 }[mode];
  const out: PathStep[] = [
    { i: 0, layer: 0, t: 0.0, work: 0.0, floor: 0.0, layers: L },
  ];
  let t = 0.0;
  let wAcc = 0.0;
  let fAcc = 0.0;
  for (let i = 1; i <= L; i++) {
    t = t + work + floor;
    wAcc = wAcc + work;
    fAcc = fAcc + floor;
    out.push({ i, layer: i, t, work: wAcc, floor: fAcc, layers: L });
  }
  return out;
}

export function band(): Record<
  BandKind,
  { min: number; max: number; n: number }
> {
  const out = {} as Record<BandKind, { min: number; max: number; n: number }>;
  for (const kind of ["capacity", "throughput", "latency"] as const) {
    const rs = validation()
      .filter((r) => r.kind === kind)
      .map((r) => r.ratio);
    out[kind] = { min: Math.min(...rs), max: Math.max(...rs), n: rs.length };
  }
  return out;
}

export function illustrative(key: string): number {
  return ILLUSTRATIVE[key]!.v;
}

// ---------------------------------------------------------------------------
// Chapter 6: reconfigurable dataflow
// ---------------------------------------------------------------------------

export type OpKind =
  | "vector"
  | "matmul"
  | "scores"
  | "values"
  | "attention"
  | "ethernet";
export type LayerOp = { name: string; kind: OpKind; w: number; out: number };

export function layerOps(mid: ModelId): LayerOp[] {
  const s = shape(mid);
  const d = s.d,
    ffn = s.ffn;
  const kvd = s.kv_heads * s.head_dim;
  return [
    { name: "RMSNorm", kind: "vector", w: 0, out: d },
    { name: "QKV", kind: "matmul", w: d * (d + 2 * kvd), out: d + 2 * kvd },
    { name: "scores QK^T", kind: "scores", w: 0, out: 0 },
    { name: "softmax, x V", kind: "values", w: 0, out: d },
    { name: "attention out", kind: "matmul", w: d * d, out: d },
    { name: "residual add", kind: "vector", w: 0, out: d },
    { name: "RMSNorm", kind: "vector", w: 0, out: d },
    { name: "MLP up and gate", kind: "matmul", w: d * 2 * ffn, out: 2 * ffn },
    { name: "SiLU x gate", kind: "vector", w: 0, out: ffn },
    { name: "MLP down", kind: "matmul", w: ffn * d, out: d },
    { name: "residual add", kind: "vector", w: 0, out: d },
  ];
}

export type TrafficOp = {
  i: number;
  name: string;
  kind: OpKind;
  w_bytes: number;
  act_bytes: number;
  kv_bytes: number;
  bytes: number;
  flops: number;
  t: number;
};
export type Traffic = {
  ops: TrafficOp[];
  bytes: number;
  flops: number;
  t: number;
};

export function layerTraffic(
  pid: PartId,
  mid: ModelId,
  fmt: Fmt,
  tokens: number,
  seqs: number,
  ctx: number,
  causal: boolean,
  fused: boolean,
): Traffic {
  const s = shape(mid);
  const b = FORMATS[fmt];
  const kvb = kvBytesPerElem(fmt);
  const d = s.d,
    h = s.heads;
  const kvd = s.kv_heads * s.head_dim;
  const avg = causal ? ctx / 2 : ctx;
  const bw = fval(pid, "off_bw") as number;
  const pk = peak(pid, fmt).flops as number;
  const kvRead = seqs * ctx * kvd * kvb;
  const ops = layerOps(mid);
  const out: TrafficOp[] = [];
  let prev = d;
  let totBytes = 0.0;
  let totFlops = 0.0;
  let tSum = 0.0;
  ops.forEach((o, i) => {
    const wbytes = o.w * b;
    const width = o.kind === "scores" ? h * avg : o.out;
    let kv = 0.0;
    let flops = 0.0;
    if (o.kind === "matmul") flops = 2 * o.w * tokens;
    if (o.kind === "scores") {
      kv = kvRead;
      flops = 2 * tokens * d * avg;
    }
    if (o.kind === "values") {
      kv = kvRead;
      flops = 2 * tokens * d * avg;
    }
    if (o.name === "QKV") kv = kv + tokens * 2 * kvd * kvb;
    let act: number;
    if (fused) {
      act = 0.0;
      if (i === 0) act = act + tokens * d * ACT_BYTES;
      if (i === ops.length - 1) act = act + tokens * d * ACT_BYTES;
    } else {
      act = (prev + width) * tokens * ACT_BYTES;
    }
    const nbytes = wbytes + act + kv;
    const tMem = nbytes / bw;
    const tComp = flops / pk;
    const t = tMem >= tComp ? tMem : tComp;
    out.push({
      i,
      name: o.name,
      kind: o.kind,
      w_bytes: wbytes,
      act_bytes: act,
      kv_bytes: kv,
      bytes: nbytes,
      flops,
      t,
    });
    totBytes = totBytes + nbytes;
    totFlops = totFlops + flops;
    tSum = tSum + t;
    prev = width;
  });
  let tLayer: number;
  if (fused) {
    const tm = totBytes / bw;
    const tc = totFlops / pk;
    tLayer = tm >= tc ? tm : tc;
  } else {
    tLayer = tSum;
  }
  return { ops: out, bytes: totBytes, flops: totFlops, t: tLayer };
}

export type DataflowStep = {
  i: number;
  name: string;
  kind: OpKind;
  u_bytes: number;
  u_t: number;
  f_bytes: number;
  f_t: number;
  u_op: number;
  f_op: number;
};

export function dataflowSteps(
  pid: PartId,
  mid: ModelId,
  fmt: Fmt,
  tokens: number,
  seqs: number,
  ctx: number,
  causal: boolean,
): DataflowStep[] {
  const u = layerTraffic(pid, mid, fmt, tokens, seqs, ctx, causal, false);
  const f = layerTraffic(pid, mid, fmt, tokens, seqs, ctx, causal, true);
  const bw = fval(pid, "off_bw") as number;
  const pk = peak(pid, fmt).flops as number;
  const out: DataflowStep[] = [];
  let ub = 0.0;
  let ut = 0.0;
  let fb = 0.0;
  let ff = 0.0;
  for (let i = 0; i < u.ops.length; i++) {
    const uo = u.ops[i]!;
    const fo = f.ops[i]!;
    ub = ub + uo.bytes;
    ut = ut + uo.t;
    fb = fb + fo.bytes;
    ff = ff + fo.flops;
    const tm = fb / bw;
    const tc = ff / pk;
    out.push({
      i,
      name: uo.name,
      kind: uo.kind,
      u_bytes: ub,
      u_t: ut,
      f_bytes: fb,
      f_t: tm >= tc ? tm : tc,
      u_op: uo.bytes,
      f_op: fo.bytes,
    });
  }
  return out;
}

export type Tiers = {
  weights: number;
  sram: number;
  hbm: number;
  ddr: number;
  sram_layers: number;
  hbm_models: number;
  ddr_models: number;
  switch_s: number;
};

export function tiers(
  pid: PartId,
  mid: ModelId,
  fmt: Fmt,
  sockets: number,
): Tiers {
  const s = shape(mid);
  const b = FORMATS[fmt];
  const w = s.params * b;
  const layer = s.per_layer * b;
  const sram = sockets * (fval(pid, "sram_bytes") as number);
  const hbm = sockets * (fval(pid, "off_bytes") as number);
  const ddr = sockets * (fval(pid, "ddr_bytes") as number);
  return {
    weights: w,
    sram,
    hbm,
    ddr,
    sram_layers: Math.floor(sram / layer),
    hbm_models: Math.floor(hbm / w),
    ddr_models: Math.floor(ddr / w),
    switch_s: w / (sockets * (fval(pid, "ddr_bw") as number)),
  };
}

// ---------------------------------------------------------------------------
// Chapter 7: a mesh of small cores
// ---------------------------------------------------------------------------

export type MeshOp = {
  name: string;
  kind: OpKind;
  bytes: number;
  per_core: number;
  fits_core: boolean;
  flops: number;
  t: number;
  i: number;
};
export type MeshLayer = {
  ops: MeshOp[];
  t: number;
  t_dram: number;
  t_eth: number;
  core_sram: number;
  cores: number;
  cards: number;
  layer_per_core: number;
  layer_fits: boolean;
  tok_s_user: number;
};

export function meshLayer(
  pid: PartId,
  mid: ModelId,
  fmt: Fmt,
  batch: number,
  ctx: number,
  cards: number,
): MeshLayer {
  const s = shape(mid);
  const b = FORMATS[fmt];
  const kvb = kvBytesPerElem(fmt);
  const kvd = s.kv_heads * s.head_dim;
  const cores = fval(pid, "units") as number;
  const coreSram = (fval(pid, "sram_bytes") as number) / cores;
  const bw = fval(pid, "off_bw") as number;
  const pk = peak(pid, fmt).flops as number;
  const link = (fval(pid, "c2c_bidir") as number) / 2;
  const g = layerGemms(mid);
  const arBytes =
    cards === 1 ? 0.0 : ((2 * (cards - 1)) / cards) * batch * s.d * ACT_BYTES;

  const gemm = (x: Gemm): Omit<MeshOp, "i"> => {
    const wc = (x.k * x.n * b) / cards;
    const fl = (2 * x.k * x.n * batch) / cards;
    const tm = wc / bw;
    const tc = fl / pk;
    return {
      name: x.name,
      kind: "matmul",
      bytes: wc,
      per_core: wc / cores,
      fits_core: wc / cores <= coreSram,
      flops: fl,
      t: tm >= tc ? tm : tc,
    };
  };

  const kv = (2 * batch * ctx * kvd * kvb) / cards;
  const afl = (2 * 2 * s.d * ctx * batch) / cards;
  const am = kv / bw;
  const ac = afl / pk;
  const attn: Omit<MeshOp, "i"> = {
    name: "attention",
    kind: "attention",
    bytes: kv,
    per_core: kv / cores,
    fits_core: kv / cores <= coreSram,
    flops: afl,
    t: am >= ac ? am : ac,
  };
  const ar: Omit<MeshOp, "i"> = {
    name: "all-reduce",
    kind: "ethernet",
    bytes: arBytes,
    per_core: 0.0,
    fits_core: true,
    flops: 0.0,
    t: arBytes / link,
  };
  const pre = [
    gemm(g[0]!),
    attn,
    gemm(g[1]!),
    ar,
    gemm(g[2]!),
    gemm(g[3]!),
    ar,
  ];
  let t = 0.0;
  let eth = 0.0;
  let dram = 0.0;
  let perCore = 0.0;
  const ops: MeshOp[] = pre.map((o, i) => {
    const op = { ...o, i };
    t = t + op.t;
    if (op.kind === "ethernet") eth = eth + op.t;
    else dram = dram + op.t;
    if (op.kind === "matmul") perCore = perCore + op.per_core;
    return op;
  });
  return {
    ops,
    t,
    t_dram: dram,
    t_eth: eth,
    core_sram: coreSram,
    cores,
    cards,
    layer_per_core: perCore,
    layer_fits: perCore <= coreSram,
    tok_s_user: 1 / (t * s.layers),
  };
}

export type MeshStep = {
  i: number;
  name: string;
  kind: OpKind;
  bytes: number;
  per_core: number;
  fits_core: boolean;
  t_op: number;
  t: number;
};

export function meshSteps(
  pid: PartId,
  mid: ModelId,
  fmt: Fmt,
  batch: number,
  ctx: number,
  cards: number,
): MeshStep[] {
  const m = meshLayer(pid, mid, fmt, batch, ctx, cards);
  let t = 0.0;
  return m.ops.map((o) => {
    t = t + o.t;
    return {
      i: o.i,
      name: o.name,
      kind: o.kind,
      bytes: o.bytes,
      per_core: o.per_core,
      fits_core: o.fits_core,
      t_op: o.t,
      t,
    };
  });
}

// ---------------------------------------------------------------------------
// Chapter 8: FPGAs and edge NPUs
// ---------------------------------------------------------------------------

/** A field of a modelled part or of a quoted one. */
export function anyval(pid: string, key: string): number | null {
  const r =
    PART_RECORDS.find((p) => p.id === pid) ??
    OTHER_PART_RECORDS.find((p) => p.id === pid);
  if (!r) throw new Error(`unknown part ${pid}`);
  const f = r.fields[key];
  return f === undefined ? null : (f.v as number | null);
}

/** The name of a modelled, quoted or in-memory part. */
export function anyName(pid: string): string {
  const r =
    PART_RECORDS.find((p) => p.id === pid) ??
    OTHER_PART_RECORDS.find((p) => p.id === pid) ??
    IMC_RECORDS[pid];
  if (!r) throw new Error(`unknown part ${pid}`);
  return r.name;
}

export type EdgeBound = {
  bits: number;
  weights: number;
  bytes: number;
  t_mem: number;
  t_comp: number | null;
  t: number;
  tok_s: number;
  fits: boolean | null;
  j_per_token: number | null;
};

export function edgeBound(
  pid: string,
  mid: ModelId,
  bits: number,
  ctx: number,
): EdgeBound {
  const s = shape(mid);
  const wb = bits / 8;
  const kvb = wb < 1 ? 1.0 : wb;
  const weights = s.params * wb;
  const nbytes = s.matmul_params * wb + ctx * s.kv_elems_per_token * kvb;
  const bw = anyval(pid, "off_bw") as number;
  const tMem = nbytes / bw;
  const flops = 2 * s.matmul_params + s.attn_flops_per_pos * ctx;
  const pk = bits <= 8 ? anyval(pid, "peak_int8") : anyval(pid, "peak_bf16");
  const tComp = pk === null ? null : flops / pk;
  const t = tComp === null || tMem >= tComp ? tMem : tComp;
  const cap = anyval(pid, "off_bytes");
  const pw = anyval(pid, "power_w");
  return {
    bits,
    weights,
    bytes: nbytes,
    t_mem: tMem,
    t_comp: tComp,
    t,
    tok_s: 1 / t,
    fits:
      cap === null ? null : weights + ctx * s.kv_elems_per_token * kvb <= cap,
    j_per_token: pw === null ? null : pw * t,
  };
}

export type EdgeStep = EdgeBound & { i: number };

export function edgeSteps(pid: string, mid: ModelId, ctx: number): EdgeStep[] {
  return BITS.map((bits, i) => ({ ...edgeBound(pid, mid, bits, ctx), i }));
}

// ---------------------------------------------------------------------------
// Chapter 9: analogue in-memory compute
// ---------------------------------------------------------------------------

export function ival(key: string): number {
  return IMC_RECORDS.hermes!.fields[key]!.v as number;
}

export type ImcGemm = {
  name: string;
  k: number;
  n: number;
  tiles: number;
  util: number;
};
export type ImcLayer = {
  gemms: ImcGemm[];
  layer_tiles: number;
  layer_chips: number;
  tiles: number;
  chips: number;
  psums: number;
  lat: number;
  t_layer: number;
  t_token: number;
  e_token: number;
  weights: number;
  capacity: number;
};

export function imcLayer(mid: ModelId, phase: number): ImcLayer {
  const s = shape(mid);
  const rows = ival("rows");
  const cols = ival("cols");
  const cores = ival("cores");
  const lat = phase === 1 ? ival("lat_1p") : ival("lat_4p");
  const eCore = (phase === 1 ? ival("e_1p") : ival("e_4p")) / cores;
  const gemms: ImcGemm[] = [];
  let layerTiles = 0;
  let psums = 0;
  for (const g of layerGemms(mid)) {
    const tk = Math.ceil(g.k / rows);
    const tn = Math.ceil(g.n / cols);
    const tiles = tk * tn;
    layerTiles += tiles;
    psums += (tk - 1) * g.n;
    gemms.push({
      name: g.name,
      k: g.k,
      n: g.n,
      tiles,
      util: (g.k * g.n) / (tiles * rows * cols),
    });
  }
  const head = Math.ceil(s.d / rows) * Math.ceil(s.vocab / cols);
  const tiles = s.layers * layerTiles + head;
  const chips = Math.ceil(tiles / cores);
  const tLayer = 4 * lat;
  const tToken = s.layers * tLayer + lat;
  return {
    gemms,
    layer_tiles: layerTiles,
    layer_chips: layerTiles / cores,
    tiles,
    chips,
    psums,
    lat,
    t_layer: tLayer,
    t_token: tToken,
    e_token: tiles * eCore,
    weights: s.matmul_params,
    capacity: chips * ival("weights"),
  };
}

export type ImcStep = {
  i: number;
  name: string;
  gemm: number;
  tiles: number;
  t: number;
  e: number;
};

export function imcSteps(mid: ModelId, phase: number): ImcStep[] {
  const m = imcLayer(mid, phase);
  const eCore = (phase === 1 ? ival("e_1p") : ival("e_4p")) / ival("cores");
  const order: [string, number][] = [
    ["program", -1],
    ["QKV", 0],
    ["attention", -1],
    ["attention out", 1],
    ["MLP up and gate", 2],
    ["MLP down", 3],
  ];
  const out: ImcStep[] = [];
  let t = 0.0;
  let e = 0.0;
  order.forEach(([name, gi], i) => {
    let tiles = 0;
    if (gi >= 0) {
      tiles = m.gemms[gi]!.tiles;
      t = t + m.lat;
      e = e + tiles * eCore;
    }
    out.push({ i, name, gemm: gi, tiles, t, e });
  });
  return out;
}

export function adcBits(inBits: number, wBits: number, rows: number): number {
  return inBits + wBits + Math.ceil(Math.log2(rows));
}

// ---------------------------------------------------------------------------
// Chapter 10: every family on the same decoder model
// ---------------------------------------------------------------------------

export type Choice = {
  i: number;
  family: string;
  name: string;
  part: string;
  fmt: Fmt;
  chips: number;
  where: string;
  t_decode: number;
  t_layer: number;
  tok_s_user: number;
  t_prefill: number;
  j_per_token: number | null;
  util: number | null;
  bound: string;
  basis: Basis;
};

export function choose(
  mid: ModelId,
  batch: number,
  ctx: number,
  prompt: number,
): Choice[] {
  const s = shape(mid);
  return FAMILIES.map((f, i) => {
    const [pid, fmt] = CHOICE[f.id]!;
    if (pid === "hermes") {
      const m = imcLayer(mid, 1);
      const ops = 2 * s.matmul_params;
      const tPf = (prompt + 4 * s.layers) * m.lat;
      return {
        i,
        family: f.id,
        name: f.name,
        part: pid,
        fmt,
        chips: m.chips,
        where: "in-array",
        t_decode: m.t_token,
        t_layer: m.t_layer,
        tok_s_user: 1 / m.t_token,
        t_prefill: tPf,
        j_per_token: m.e_token,
        util: ops / (m.t_token * m.chips * ival("tops_1p")),
        bound: "crossbar-only",
        basis: "dense",
      };
    }
    const p = pid as PartId;
    const d = decode(p, mid, fmt, batch, ctx);
    const pf = prefill(p, mid, fmt, prompt);
    const pk = peak(p, fmt);
    return {
      i,
      family: f.id,
      name: f.name,
      part: pid,
      fmt,
      chips: d.chips,
      where: d.where,
      t_decode: d.t_step,
      t_layer: d.t_step / s.layers,
      tok_s_user: d.tok_s_user,
      t_prefill: pf.t,
      j_per_token: d.j_per_token,
      util:
        pk.flops === null ? null : d.flops / (d.t_step * d.chips * pk.flops),
      bound: d.bound,
      basis: pk.basis,
    };
  });
}
