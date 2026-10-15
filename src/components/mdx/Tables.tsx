/**
 * Sourced tables for the chapters and the /specs page. Server Components.
 *
 * <PartTable ids="h100,b200" fields="off_bytes,off_bw" /> prints each field
 * of each part with its status, and links its source (title and access date
 * on hover). <FamilyTable /> prints the map's positions with their basis.
 * <ValidationTable kinds="capacity,latency" /> prints the model against
 * published figures. <DecodeTable /> prints decode bounds for a list of cases.
 */
import {
  fmtBits,
  fmtBytes,
  fmtFlops,
  fmtInt,
  fmtRate,
  fmtTime,
  fmtWatts,
  trim,
} from "@/lib/format";
import {
  FAMILIES,
  GRAIN_LABELS,
  IMC_RECORDS,
  OTHER_PART_RECORDS,
  PART_RECORDS,
  PLACEMENT_LABELS,
  SCHEDULE_LABELS,
  SOURCES,
  anyName,
  choose,
  decode,
  validation,
  type Field,
  type Fmt,
  type ModelId,
  type PartId,
} from "@/lib/accel/model";

/** How each field is printed, and its row label. */
export const FIELD_INFO: Record<
  string,
  { label: string; fmt: (v: number) => string }
> = {
  peak_bf16: { label: "BF16, dense", fmt: fmtFlops },
  peak_fp16: { label: "FP16, dense", fmt: fmtFlops },
  peak_fp16_sparse: { label: "FP16, sparse (vendor figure)", fmt: fmtFlops },
  peak_fp8: { label: "FP8, dense", fmt: fmtFlops },
  peak_int8: {
    label: "INT8, dense",
    fmt: (v) => fmtFlops(v).replace("FLOP/s", "OP/s"),
  },
  sram_bytes: { label: "On-chip SRAM", fmt: fmtBytes },
  sram_bw: { label: "SRAM bandwidth", fmt: fmtRate },
  off_bytes: {
    label: "Off-chip memory",
    fmt: (v) => (v === 0 ? "none" : fmtBytes(v)),
  },
  off_bw: {
    label: "Off-chip bandwidth",
    fmt: (v) => (v === 0 ? "none" : fmtRate(v)),
  },
  c2c_bidir: { label: "Chip-to-chip (both directions)", fmt: fmtRate },
  hop_s: { label: "Chip-to-chip hop latency", fmt: fmtTime },
  power_w: { label: "Power", fmt: fmtWatts },
  units: { label: "Cores / SMs / TensorCores", fmt: fmtInt },
  array_rows: { label: "Systolic array rows", fmt: fmtInt },
  array_cols: { label: "Systolic array columns", fmt: fmtInt },
  io_bps: { label: "System I/O", fmt: fmtBits },
  area_mm2: { label: "Silicon area", fmt: (v) => `${fmtInt(v)} mm²` },
  dies: { label: "Dies on the wafer", fmt: fmtInt },
  transistors: {
    label: "Transistors",
    fmt: (v) => `${trim(v / 1e12)} trillion`,
  },
  core_um2: { label: "Area per core", fmt: (v) => `${fmtInt(v)} µm²` },
  core_sram: { label: "SRAM per core", fmt: fmtBytes },
  fabric_bps: { label: "On-wafer fabric", fmt: fmtBits },
  redundancy: { label: "Defect tolerance", fmt: String },
  process: { label: "Process", fmt: String },
  c2c_links: { label: "Chip-to-chip links", fmt: fmtInt },
  link_bps: { label: "One link, per direction", fmt: fmtBits },
  node_chips: { label: "Chips per node", fmt: fmtInt },
  clock_hz: { label: "Clock", fmt: (v) => `${trim(v / 1e6)} MHz` },
  rack_chips: { label: "Chips per rack", fmt: fmtInt },
  ddr_bytes: { label: "DDR (third tier)", fmt: fmtBytes },
  ddr_bw: { label: "DDR bandwidth", fmt: fmtRate },
  tile_sram: { label: "SRAM per tile", fmt: fmtBytes },
  riscv: { label: "'Big RISC-V' cores", fmt: fmtInt },
  peak_int4: {
    label: "INT4, dense",
    fmt: (v) => fmtFlops(v).replace("FLOP/s", "OP/s"),
  },
  scale_up: { label: "Scale-up, chips", fmt: fmtInt },
  cores: { label: "Crossbar cores", fmt: fmtInt },
  rows: { label: "Crossbar rows", fmt: fmtInt },
  cols: { label: "Crossbar columns", fmt: fmtInt },
  weights: { label: "Weights on chip", fmt: fmtInt },
  devices_per_weight: { label: "PCM devices per weight", fmt: fmtInt },
  lat_1p: { label: "MVM latency, 1-phase", fmt: fmtTime },
  lat_4p: { label: "MVM latency, 4-phase", fmt: fmtTime },
  e_1p: {
    label: "Energy, 64 MVMs, 1-phase",
    fmt: (v) => `${trim(v * 1e6)} µJ`,
  },
  e_4p: {
    label: "Energy, 64 MVMs, 4-phase",
    fmt: (v) => `${trim(v * 1e6)} µJ`,
  },
  tops_1p: {
    label: "Peak MVM rate, 1-phase",
    fmt: (v) => fmtFlops(v).replace("FLOP/s", "OP/s"),
  },
  tops_4p: {
    label: "Peak MVM rate, 4-phase",
    fmt: (v) => fmtFlops(v).replace("FLOP/s", "OP/s"),
  },
  topsw_1p: {
    label: "Efficiency, 1-phase",
    fmt: (v) => `${trim(v / 1e12)} TOPS/W`,
  },
  topsw_4p: {
    label: "Efficiency, 4-phase",
    fmt: (v) => `${trim(v / 1e12)} TOPS/W`,
  },
  io_bits: { label: "Input/output bits", fmt: fmtInt },
  die_mm2: { label: "Die area", fmt: (v) => `${fmtInt(v)} mm²` },
};

const STATUS_NOTE: Record<string, string> = {
  spec: "",
  derived: " (derived)",
  reported: " (reported)",
  illustrative: " (illustrative)",
  undisclosed: "",
};

export function FieldCell({ f, k }: { f: Field; k: string }): JSX.Element {
  const src = SOURCES[f.src];
  const info = FIELD_INFO[k];
  const text =
    f.v === null
      ? "not disclosed"
      : typeof f.v === "number"
        ? info
          ? info.fmt(f.v)
          : trim(f.v)
        : f.v;
  return (
    <span data-field={k} data-status={f.st}>
      {text}
      {STATUS_NOTE[f.st]}
      {src && (
        <a
          href={src.url}
          title={`${src.title}, read ${src.accessed}. ${f.ref}`}
          className="ml-1 align-super text-[0.65rem]"
        >
          [{f.src}]
        </a>
      )}
    </span>
  );
}

function Region({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}): JSX.Element {
  return (
    <div
      className="my-6 overflow-x-auto"
      tabIndex={0}
      role="region"
      aria-label={label}
    >
      {children}
    </div>
  );
}

/** Parts side by side, one row per field. */
export function PartTable({
  ids,
  fields,
}: {
  ids: string;
  fields: string;
}): JSX.Element {
  const recs = ids.split(",").map((id) => {
    const r =
      PART_RECORDS.find((x) => x.id === id) ??
      OTHER_PART_RECORDS.find((x) => x.id === id) ??
      IMC_RECORDS[id];
    if (!r) throw new Error(`no part ${id}`);
    return r;
  });
  const ks = fields.split(",");
  return (
    <Region label="Specifications">
      <table className="w-full text-sm">
        <thead>
          <tr>
            <th className="text-left">&nbsp;</th>
            {recs.map((r) => (
              <th key={r.id} className="text-left">
                {r.name}
                {r.avail.v !== "product" && (
                  <span
                    className="ml-1 text-xs font-normal"
                    data-avail={r.avail.v as string}
                  >
                    ({r.avail.v})
                  </span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {ks.map((k) => (
            <tr key={k}>
              <th scope="row" className="text-left font-normal">
                {FIELD_INFO[k]?.label ?? k}
              </th>
              {recs.map((r) => {
                const f = r.fields[k];
                return (
                  <td key={r.id}>
                    {f ? <FieldCell f={f} k={k} /> : <span>–</span>}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </Region>
  );
}

/** The families on the map's axes, each position with its basis. */
export function FamilyTable(): JSX.Element {
  const cell = (f: Field<number>, labels: readonly string[], k: string) => {
    const src = SOURCES[f.src]!;
    return (
      <span data-field={k} data-status={f.st}>
        {labels[f.v]}
        <a
          href={src.url}
          title={`${src.title}, read ${src.accessed}. ${f.ref}`}
          className="ml-1 align-super text-[0.65rem]"
        >
          [{f.src}]
        </a>
      </span>
    );
  };
  return (
    <Region label="Families">
      <table className="w-full text-sm">
        <thead>
          <tr>
            <th className="text-left">Family</th>
            <th className="text-left">Weights live in</th>
            <th className="text-left">Who schedules</th>
            <th className="text-left">Grain</th>
          </tr>
        </thead>
        <tbody>
          {FAMILIES.map((f) => (
            <tr key={f.id} data-family={f.id}>
              <th scope="row" className="text-left font-normal">
                {f.name}
                <span className="block text-xs text-neutral-600 dark:text-neutral-400">
                  {f.example}
                </span>
              </th>
              <td>{cell(f.placement, PLACEMENT_LABELS, "placement")}</td>
              <td>{cell(f.schedule, SCHEDULE_LABELS, "schedule")}</td>
              <td>{cell(f.grain, GRAIN_LABELS, "grain")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Region>
  );
}

/** The model against published figures. */
export function ValidationTable({
  kinds,
  ids,
}: {
  kinds?: string;
  ids?: string;
}): JSX.Element {
  const want = kinds ? kinds.split(",") : null;
  const only = ids ? ids.split(",") : null;
  const rows = validation().filter(
    (v) => (!want || want.includes(v.kind)) && (!only || only.includes(v.id)),
  );
  return (
    <Region label="Validation">
      <table className="w-full text-sm">
        <thead>
          <tr>
            <th className="text-left">Check</th>
            <th className="text-left">Kind</th>
            <th className="text-right">Published</th>
            <th className="text-right">Model</th>
            <th className="text-right">Ratio</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((v) => {
            const src = SOURCES[v.src]!;
            return (
              <tr key={v.id} data-check={v.id}>
                <th scope="row" className="text-left font-normal">
                  {v.label}
                  <a
                    href={src.url}
                    title={`${src.title}, read ${src.accessed}`}
                    className="ml-1 align-super text-[0.65rem]"
                  >
                    [{v.src}]
                  </a>
                </th>
                <td>{v.kind}</td>
                <td className="text-right font-mono">{trim(v.published, 5)}</td>
                <td className="text-right font-mono">{trim(v.model, 4)}</td>
                <td className="text-right font-mono">{trim(v.ratio, 3)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </Region>
  );
}

/** Decode bounds for a list of cases: "part:model:fmt:batch:ctx[:chips[:group]]", separated by spaces. */
export function DecodeTable({ cases }: { cases: string }): JSX.Element {
  const rows = cases
    .trim()
    .split(/\s+/)
    .map((c) => {
      const [p, m, f, b, ctx, n, g] = c.split(":");
      const d = decode(
        p as PartId,
        m as ModelId,
        f as Fmt,
        Number(b),
        Number(ctx),
        n ? Number(n) : 0,
        g ? Number(g) : 1,
      );
      const name = PART_RECORDS.find((r) => r.id === p)!.name;
      return { key: c, name, m, f, d };
    });
  return (
    <Region label="Decode bounds">
      <table className="w-full text-sm">
        <thead>
          <tr>
            <th className="text-left">Part</th>
            <th className="text-left">Model, format</th>
            <th className="text-right">Chips</th>
            <th className="text-right">Step</th>
            <th className="text-left">Bound</th>
            <th className="text-right">Tokens/s per user</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key} data-case={r.key}>
              <th scope="row" className="text-left font-normal">
                {r.name}
              </th>
              <td>
                {r.m}, {(r.f ?? "").toUpperCase()}
              </td>
              <td className="text-right font-mono">{fmtInt(r.d.chips)}</td>
              <td className="text-right font-mono">{fmtTime(r.d.t_step)}</td>
              <td>
                {r.d.bound}
                {r.d.basis === "sparse" ? " (sparse peak)" : ""}
              </td>
              <td className="text-right font-mono">{trim(r.d.tok_s_user)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Region>
  );
}

/** Chapter 10: every family's representative part on the same model. */
export function ChooseTable({ model }: { model: string }): JSX.Element {
  const rows = choose(model as ModelId, 1, 1024, 2048);
  return (
    <Region label="Every family on the same model">
      <table className="w-full text-sm">
        <thead>
          <tr>
            <th className="text-left">Family</th>
            <th className="text-left">Part, format</th>
            <th className="text-right">Chips</th>
            <th className="text-right">Per token</th>
            <th className="text-right">Tokens/s per user</th>
            <th className="text-right">Prefill, 2,048</th>
            <th className="text-right">J per token</th>
            <th className="text-right">Decode utilisation</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.family} data-choice={r.family}>
              <th scope="row" className="text-left font-normal">
                {r.name}
              </th>
              <td>
                {anyName(r.part)}, {r.fmt.toUpperCase()}
              </td>
              <td className="text-right font-mono">{fmtInt(r.chips)}</td>
              <td className="text-right font-mono">
                {fmtTime(r.t_decode)}
                {r.bound === "crossbar-only" ? "*" : ""}
              </td>
              <td className="text-right font-mono">{trim(r.tok_s_user)}</td>
              <td className="text-right font-mono">{fmtTime(r.t_prefill)}</td>
              <td className="text-right font-mono">
                {r.j_per_token === null ? "not disclosed" : trim(r.j_per_token)}
              </td>
              <td className="text-right font-mono">
                {r.util === null ? "–" : `${(r.util * 100).toFixed(2)}%`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Region>
  );
}
