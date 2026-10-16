/**
 * This site's favicon glyph (src/app/icon.svg) and its card's graphic.
 * Glyph: the chip of the Hardware track, the letter A, #ea580c.
 */
import {
  FAMILIES,
  PART_RECORDS,
  placement,
  type Fmt,
  type PartId,
} from "@/lib/accel/model";
import { PLACEMENT_COLOUR } from "@/lib/viz/palette";

export function Glyph({ size }: { size: number }): JSX.Element {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32">
      <rect x="3.5" y="3.5" width="25" height="25" rx="2.5" fill="#ea580c" />
      <path
        d="M9 0h3v4h-3zM20 0h3v4h-3zM9 28h3v4h-3zM20 28h3v4h-3zM0 9h4v3h-4zM0 20h4v3h-4zM28 9h4v3h-4zM28 20h4v3h-4z"
        fill="#ea580c"
      />
      <path
        d="M19.57 23.03 18.55 19.7H13.45L12.43 23.03H9.24L14.18 8.97H17.8L22.76 23.03ZM16.82 13.95Q16.73 13.62 16.57 13.11Q16.41 12.6 16.25 12.07Q16.1 11.54 16 11.15Q15.9 11.54 15.74 12.12Q15.57 12.7 15.42 13.22Q15.27 13.74 15.2 13.95L14.2 17.21H17.84Z"
        fill="#fff"
      />
    </svg>
  );
}

/**
 * The landing hero's bars: how many chips of each family hold Llama 3.1 70B and one 1,024-token
 * KV cache, log scale, coloured by where the weights live, from the model at build time.
 */
const CASES: [PartId, Fmt][] = [
  ["h100", "fp8"],
  ["tpu-7x", "fp8"],
  ["tpu-v6e", "bf16"],
  ["wse-3", "bf16"],
  ["groq3", "fp8"],
  ["groqchip", "int8"],
];

export function Motif(): JSX.Element {
  const rows = CASES.map(([pid, fmt]) => {
    const p = PART_RECORDS.find((r) => r.id === pid)!;
    const fam = FAMILIES.find((f) => f.id === p.family)!;
    return {
      pid,
      fam,
      chips: placement(pid, "llama3-70b", fmt, 1, 1024).chips,
    };
  });
  const hi = Math.log10(Math.max(...rows.map((r) => r.chips))) + 0.15;
  const lo = -0.6;
  return (
    <svg width="440" height="420" viewBox="0 0 440 420">
      {rows.map((r, i) => (
        <g key={r.pid}>
          <rect
            x={10}
            y={22 + i * 66}
            width={420}
            height={46}
            rx={8}
            fill="#1e293b"
          />
          <rect
            x={10}
            y={22 + i * 66}
            width={(420 * (Math.log10(r.chips) - lo)) / (hi - lo)}
            height={46}
            rx={8}
            fill={PLACEMENT_COLOUR[r.fam.placement.v]}
          />
        </g>
      ))}
    </svg>
  );
}
