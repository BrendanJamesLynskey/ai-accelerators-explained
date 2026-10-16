/**
 * Every number the chapters quote comes from here, so prose cannot drift
 * from the tested model. Two kinds of path:
 *
 *   - a record path: "part.h100.off_bw", "shape.llama3-70b.params",
 *     "val.mlperf-h100.ratio", "band.latency.max", "pub.groq-70b-tps",
 *     "overhead.groq-70b-tps.per_layer", "ill.d0_per_cm2";
 *   - a model call and a field: "decode(h100,llama3-70b,fp8,1,4096,8).t_step"
 *     (the arguments are the model function's, in order; numbers are parsed).
 *
 * tests/unit/values.test.ts checks every path the chapters use resolves.
 */
import {
  fmtBits,
  fmtBytes,
  fmtFlops,
  fmtInt,
  fmtJoules,
  fmtRate,
  fmtTime,
  fmtWatts,
  pct,
  trim,
} from "@/lib/format";

import {
  ILLUSTRATIVE,
  IMC_RECORDS,
  MODEL_ORDER,
  OTHER_PART_RECORDS,
  PART_ORDER,
  PUBLISHED,
  adcBits,
  band,
  choose,
  decode,
  edgeBound,
  imcLayer,
  layerTraffic,
  meshLayer,
  tiers,
  impliedOverhead,
  layerMxu,
  lpuCapacity,
  maxBatch,
  mxu,
  part,
  placement,
  prefill,
  ridge,
  shape,
  throughputCeiling,
  validation,
  floorFit,
  mlperfCeiling,
  waferYield,
  wafersNeeded,
} from "./model";

export type Fmt =
  | "rate"
  | "bits"
  | "bytes"
  | "flops"
  | "time"
  | "watts"
  | "joules"
  | "pct"
  | "int"
  | "num"
  | "ops"
  | "opsw"
  | "raw";

type Arg = string | number | boolean;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Fn = (...a: any[]) => unknown;

/** The model functions a path may call. */
const CALLS: Record<string, Fn> = {
  placement,
  decode,
  prefill,
  ridge,
  ceiling: throughputCeiling,
  maxBatch,
  mxu,
  layerMxu,
  waferYield,
  wafersNeeded,
  lpuCapacity,
  layerTraffic,
  tiers,
  meshLayer,
  edgeBound,
  imcLayer,
  adcBits,
  choose,
};

function build(): Record<string, unknown> {
  const t: Record<string, unknown> = {};
  const p: Record<string, unknown> = {};
  for (const id of PART_ORDER) p[id] = part(id);
  t.part = p;
  const s: Record<string, unknown> = {};
  for (const id of MODEL_ORDER) s[id] = shape(id);
  t.shape = s;
  const v: Record<string, unknown> = {};
  for (const c of validation()) v[c.id] = c;
  t.val = v;
  t.band = band();
  const pub: Record<string, unknown> = {};
  for (const [k, f] of Object.entries(PUBLISHED)) pub[k] = f.v;
  t.pub = pub;
  const o: Record<string, unknown> = {};
  for (const k of [
    "groq-70b-tps",
    "cerebras-70b-tps",
    "cerebras-8b-tps",
    "sn40l-8b-tps",
  ])
    o[k] = impliedOverhead(k);
  t.overhead = o;
  const ill: Record<string, unknown> = {};
  for (const [k, f] of Object.entries(ILLUSTRATIVE)) ill[k] = f.v;
  t.ill = ill;
  const other: Record<string, unknown> = {};
  for (const r of OTHER_PART_RECORDS) {
    const f: Record<string, unknown> = { avail: r.avail.v };
    for (const [k, x] of Object.entries(r.fields)) f[k] = x.v;
    other[r.id] = f;
  }
  t.other = other;
  const imc: Record<string, unknown> = {};
  for (const [id, r] of Object.entries(IMC_RECORDS)) {
    const f: Record<string, unknown> = { avail: r.avail.v };
    for (const [k, x] of Object.entries(r.fields)) f[k] = x.v;
    imc[id] = f;
  }
  t.imc = imc;
  // brief 42: the latency floor and the MLPerf ceilings with prefill
  const f = floorFit();
  const rowsById: Record<string, unknown> = {};
  for (const r of f.rows) rowsById[r.id] = r;
  t.floor = { ...f, row: rowsById };
  const P = (k: string) => PUBLISHED[k]!.v;
  const nIn = P("mlperf-input-mean");
  const nOut = P("mlperf-tokens-per-sample");
  const h100 = mlperfCeiling("h100", "llama2-70b", "fp8", 8, nIn, nOut);
  t.mlperf = {
    h100: { ...h100, decodeRatio: P("mlperf-h100") / h100.decodeOnly },
    gb200: mlperfCeiling(
      "b200",
      "llama2-70b",
      "fp4",
      P("mlperf-gb200-chips"),
      nIn,
      nOut,
    ),
  };
  return t;
}

let TREE: Record<string, unknown> | null = null;

function walk(cur: unknown, keys: string[], path: string): unknown {
  for (const k of keys) {
    if (cur === null || typeof cur !== "object" || !(k in cur))
      throw new Error(`no value at ${path}`);
    cur = (cur as Record<string, unknown>)[k];
  }
  return cur;
}

const CALL = /^([A-Za-z]+)\(([^)]*)\)(?:\.(.+))?$/;

export function lookup(path: string): number | string {
  let cur: unknown;
  const m = CALL.exec(path);
  if (m) {
    const fn = CALLS[m[1]!];
    if (!fn) throw new Error(`no model function ${m[1]} in ${path}`);
    const args: Arg[] = m[2]!
      .split(",")
      .filter((x) => x !== "")
      .map((x) =>
        x === "true"
          ? true
          : x === "false"
            ? false
            : /^-?[\d.]+(e[+-]?\d+)?$/.test(x)
              ? Number(x)
              : x,
      );
    const res = fn(...args);
    cur = m[3] ? walk(res, m[3].split("."), path) : res;
  } else {
    if (!TREE) TREE = build();
    cur = walk(TREE, path.split("."), path);
  }
  if (typeof cur !== "number" && typeof cur !== "string")
    throw new Error(`${path} is not a number or string`);
  return cur;
}

export function formatValue(v: number | string, fmt: Fmt): string {
  if (typeof v === "string") return v;
  switch (fmt) {
    case "rate":
      return fmtRate(v);
    case "bits":
      return fmtBits(v);
    case "bytes":
      return fmtBytes(v);
    case "flops":
      return fmtFlops(v);
    case "time":
      return fmtTime(v);
    case "watts":
      return fmtWatts(v);
    case "joules":
      return fmtJoules(v);
    case "pct":
      return pct(v, v < 0.01 ? 2 : v < 0.1 ? 1 : 0);
    case "int":
      return fmtInt(v);
    case "ops":
      return fmtFlops(v).replace("FLOP/s", "OP/s");
    case "opsw":
      return `${trim(v / 1e12)} TOPS/W`;
    case "raw":
      return String(v);
    default:
      return trim(v);
  }
}
