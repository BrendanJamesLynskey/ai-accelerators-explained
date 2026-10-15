"use client";

/**
 * Chapter 10's hero: the same decoder model on every family, side by side.
 * One row per family (its representative part and format), one metric at a
 * time on a log scale: the bound on one user's tokens per second, a 2,048-
 * token prefill, energy per token, or the chips it takes to hold the model.
 * The animation adds the families one by one. Every row is a state of
 * `choose` (checked against the Python reference); bounds are not measured
 * speeds, and the chapter's validation table says how far each family's
 * published figures fall short of its bound.
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Hatch } from "@/components/interactive/Hatch";
import { Segmented, Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { fmtInt, fmtTime, pct, trim } from "@/lib/format";
import { chooseCaption, partShort } from "@/lib/accel/captions";
import {
  FAMILIES,
  choose,
  shape,
  type Choice,
  type ModelId,
} from "@/lib/accel/model";
import { PLACEMENT_COLOUR, STATE_COLOUR } from "@/lib/viz/palette";

const W = 360;
const SHORT: Record<string, string> = {
  gpu: "GPU",
  systolic: "Systolic",
  wafer: "Wafer",
  lpu: "LPU",
  dataflow: "Dataflow",
  mesh: "Mesh",
  "fpga-npu": "NPU",
  imc: "In-memory",
};

type Metric = "tps" | "prefill" | "energy" | "chips";
const METRICS: { value: Metric; label: string }[] = [
  { value: "tps", label: "Tokens/s, one user" },
  { value: "prefill", label: "Prefill time" },
  { value: "energy", label: "Energy per token" },
  { value: "chips", label: "Chips" },
];

function metric(r: Choice, m: Metric): number | null {
  if (m === "tps") return r.tok_s_user;
  if (m === "prefill") return r.t_prefill;
  if (m === "energy") return r.j_per_token;
  return r.chips;
}

function show(v: number, m: Metric): string {
  if (m === "tps") return `${trim(v)}/s`;
  if (m === "prefill") return fmtTime(v);
  if (m === "energy") return `${trim(v)} J`;
  return fmtInt(v);
}

export default function ChooseWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [mid, setMid] = useState<ModelId>("llama3-70b");
  const [met, setMet] = useState<Metric>("tps");
  const [hover, setHover] = useState<string | null>(null);
  const font = useSvgFont(W);
  const fs = font.fs;
  const rows = useMemo(() => choose(mid, 1, 1024, 2048), [mid]);
  const st = useStepper(rows.length, { stepMs: 1500, resetKey: mid });
  const k = st.step;
  const r = rows[k]!;
  const vals = rows.map((x) => metric(x, met));
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of vals)
    if (v !== null && v > 0) {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
  const l0 = Math.floor(Math.log10(lo));
  const l1 = Math.max(l0 + 1, Math.ceil(Math.log10(hi)));
  const LABEL = fs(10) * 5.6 + 4;
  const ROWH = Math.max(22, fs(16));
  const TOP = fs(11) + 8;
  const x0 = LABEL + 6;
  const xw = W - x0 - fs(52);
  const sx = (v: number) => ((Math.log10(v) - l0) / (l1 - l0)) * xw;
  const H = TOP + rows.length * ROWH + fs(11) + 10;
  const decades = [];
  for (let d = l0; d <= l1; d++) decades.push(d);
  const step = decades.length > 6 ? 2 : 1;

  const visual = (
    <div className="mx-auto max-w-lg">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full"
        role="img"
        aria-label={`${METRICS.find((x) => x.value === met)!.label} for ${rows.length} families, log scale; ${r.name} highlighted.`}
      >
        <defs>
          <Hatch id="choose-hatch" />
        </defs>
        {decades.map((d, i) =>
          i % step === 0 ? (
            <g key={d}>
              <line
                x1={x0 + sx(10 ** d)}
                x2={x0 + sx(10 ** d)}
                y1={TOP - 2}
                y2={TOP + rows.length * ROWH}
                stroke="#a3a3a3"
                strokeDasharray="2 3"
              />
              <text
                x={x0 + sx(10 ** d)}
                y={TOP - 5}
                textAnchor="middle"
                style={{ fontSize: fs(9) }}
                className="fill-neutral-600 dark:fill-neutral-400"
              >
                {show(10 ** d, met)}
              </text>
            </g>
          ) : null,
        )}
        {rows.map((x, i) => {
          const y = TOP + i * ROWH;
          const v = vals[i] ?? null;
          const state = i === k ? "active" : i < k ? "done" : "waiting";
          const fam = FAMILIES[i]!;
          return (
            <g key={x.family} data-family={x.family} data-state={state}>
              <text
                x={LABEL}
                y={y + ROWH * 0.66}
                textAnchor="end"
                style={{ fontSize: fs(10) }}
                className={
                  state === "active"
                    ? "fill-neutral-950 font-semibold dark:fill-white"
                    : "fill-neutral-600 dark:fill-neutral-400"
                }
              >
                {SHORT[x.family]}
              </text>
              {i <= k &&
                (v === null ? (
                  <g>
                    <rect
                      x={x0}
                      y={y + 3}
                      width={xw * 0.3}
                      height={ROWH - 7}
                      fill="url(#choose-hatch)"
                    />
                    <text
                      x={x0 + xw * 0.3 + 4}
                      y={y + ROWH * 0.66}
                      style={{ fontSize: fs(9) }}
                      className="fill-neutral-700 dark:fill-neutral-300"
                    >
                      not disclosed
                    </text>
                  </g>
                ) : (
                  <g>
                    <rect
                      x={x0}
                      y={y + 3}
                      width={Math.max(2, sx(v))}
                      height={ROWH - 7}
                      rx={1.5}
                      fill={
                        state === "active"
                          ? STATE_COLOUR.active
                          : PLACEMENT_COLOUR[fam.placement.v]
                      }
                      opacity={state === "active" ? 1 : 0.8}
                    />
                    <text
                      x={x0 + Math.max(2, sx(v)) + 4}
                      y={y + ROWH * 0.66}
                      style={{ fontSize: fs(9) }}
                      className="fill-neutral-800 dark:fill-neutral-200"
                    >
                      {show(v, met)}
                    </text>
                  </g>
                ))}
            </g>
          );
        })}
        <text
          x={x0}
          y={H - 6}
          style={{ fontSize: fs(9) }}
          className="fill-neutral-700 dark:fill-neutral-300"
        >
          {font.narrow
            ? "log scale; colour: where weights live"
            : "log scale; colour = where the weights live (chapter 1)"}
        </text>
      </svg>
      <p className="mt-1 text-[0.7rem] text-neutral-600 dark:text-neutral-400">
        One row per family, its representative part running {shape(mid).name} at
        batch 1. Bars are the model&apos;s bounds, not measured speeds: SRAM
        designs fall furthest short of theirs (see the validation below).
        Hatched: the vendor does not give the power.
      </p>
    </div>
  );

  const stats = (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
      <Stat
        label="Part"
        value={partShort(r.part)}
        hint={`${r.fmt.toUpperCase()}, ${fmtInt(r.chips)} chip${r.chips > 1 ? "s" : ""}`}
      />
      <Stat
        label="Per token"
        value={fmtTime(r.t_decode)}
        hint={`${fmtTime(r.t_layer)} per layer`}
      />
      <Stat label="Prefill" value={fmtTime(r.t_prefill)} hint="2,048 tokens" />
      <Stat
        label="Decode utilisation"
        value={r.util === null ? "–" : pct(r.util, 2)}
        hint="of the chips' peak"
      />
    </div>
  );

  return (
    <AnimationPanel
      testId="choose-widget"
      title="Every family on the same decoder model"
      summary="The fewest chips of each family's representative part that hold the model, one user at a time: the model's bounds on decode and prefill, energy where the power is given, and the chips it takes."
      stepper={st}
      stepLabel="family"
      caption={chooseCaption(
        r,
        rows.length,
        shape(mid).name,
        partShort(r.part),
      )}
      visual={visual}
      stats={stats}
      equation={children}
      hl={hover ?? "mem comp"}
      onEquationHover={setHover}
      params={
        <div className="flex flex-wrap gap-4">
          <Segmented
            label="Show"
            value={met}
            options={METRICS}
            onChange={(v) => setMet(v as Metric)}
          />
          <Segmented
            label="Running model"
            value={mid}
            options={[
              { value: "llama3-8b", label: "Llama 3.1 8B" },
              { value: "llama3-70b", label: "Llama 3.1 70B" },
            ]}
            onChange={(v) => setMid(v as ModelId)}
          />
        </div>
      }
    />
  );
}
