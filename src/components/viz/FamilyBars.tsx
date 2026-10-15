/**
 * The home page's picture: how many chips of each family hold Llama 3.1
 * 70B and one 1,024-token KV cache, log scale, from the model (`placement`).
 * Server Component.
 */
import { fmtBytes, fmtInt } from "@/lib/format";
import {
  FAMILIES,
  PART_RECORDS,
  placement,
  type Fmt,
  type PartId,
} from "@/lib/accel/model";
import { PLACEMENT_COLOUR } from "@/lib/viz/palette";

const CASES: [PartId, Fmt][] = [
  ["h100", "fp8"],
  ["tpu-7x", "fp8"],
  ["tpu-v6e", "bf16"],
  ["wse-3", "bf16"],
  ["groq3", "fp8"],
  ["groqchip", "int8"],
];

export function FamilyBars(): JSX.Element {
  const rows = CASES.map(([pid, fmt]) => {
    const p = PART_RECORDS.find((r) => r.id === pid)!;
    const fam = FAMILIES.find((f) => f.id === p.family)!;
    const pl = placement(pid, "llama3-70b", fmt, 1, 1024);
    return { pid, fmt, name: p.name, avail: p.avail.v, fam, pl };
  });
  const hi = Math.log10(Math.max(...rows.map((r) => r.pl.chips))) + 0.15;
  const lo = -1.6; // room for the label inside even a one-chip bar
  return (
    <ol
      className="space-y-1.5"
      aria-label="Chips needed to hold Llama 3.1 70B, log scale"
    >
      {rows.map((r) => (
        <li key={r.pid} className="text-xs" data-part={r.pid}>
          <div className="flex justify-between gap-2 text-neutral-700 dark:text-neutral-300">
            <span>
              {r.name
                .replace(" (first-generation LPU)", "")
                .replace(" (LPX rack)", "")}
              {r.avail === "announced" ? " (announced)" : ""}
            </span>
            <span className="font-mono">{r.fmt.toUpperCase()}</span>
          </div>
          <span className="relative mt-0.5 block h-5 rounded bg-neutral-100 dark:bg-neutral-800">
            <span
              className="absolute inset-y-0 left-0 rounded"
              style={{
                width: `${((Math.log10(r.pl.chips) - lo) / (hi - lo)) * 100}%`,
                background: PLACEMENT_COLOUR[r.fam.placement.v],
              }}
            />
            <span className="absolute inset-y-0 left-1 flex items-center font-mono text-[0.7rem] text-neutral-950">
              {fmtInt(r.pl.chips)} × {fmtBytes(r.pl.cap)}{" "}
              {r.pl.where === "sram" ? "SRAM" : "HBM"}
            </span>
          </span>
        </li>
      ))}
    </ol>
  );
}
