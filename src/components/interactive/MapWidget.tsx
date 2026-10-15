"use client";

/**
 * Chapter 1's hero: the families of AI chips on two axes, where the weights
 * live (left to right, from off-chip DRAM to inside the memory array) and
 * who decides when data moves (top to bottom, from hardware at run time to
 * the compiler ahead of time), with each family's grain drawn as its marker
 * (one big engine, a few cores, a carpet of tiny ones). The animation steps
 * through the families; for the four with parts in chapters 2-5 a strip
 * underneath counts the chips the running model's weights need. Every frame
 * is a state of `mapSteps` (checked against the Python reference).
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Segmented, Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { fmtBytes, fmtInt } from "@/lib/format";
import { mapCaption } from "@/lib/accel/captions";
import {
  PLACEMENT_LABELS,
  SCHEDULE_LABELS,
  mapSteps,
  part,
  shape,
  type MapStep,
  type ModelId,
} from "@/lib/accel/model";
import { PLACEMENT_COLOUR, STATE_COLOUR } from "@/lib/viz/palette";

const W = 360;
const LEFT = 74;
const TOP = 26;
const COLW = (W - LEFT - 6) / 4;
const ROWH = 66;
const H = TOP + 3 * ROWH + 8;

const SHORT_PLACE = ["DRAM", "SRAM + DRAM", "SRAM", "in memory"];
const SHORT_SCHED = ["hardware", "mixed", "compiler"];
const SHORT_NAME: Record<string, string> = {
  gpu: "GPU",
  systolic: "Systolic",
  wafer: "Wafer",
  lpu: "LPU",
  dataflow: "Dataflow",
  mesh: "Mesh",
  "fpga-npu": "FPGA/NPU",
  imc: "In-memory",
};

/** The marker's dots: one big engine, a 3 x 3 of cores, or a 6 x 6 carpet. */
function Marker({
  x,
  y,
  grain,
  colour,
  active,
  muted,
}: {
  x: number;
  y: number;
  grain: number;
  colour: string;
  active: boolean;
  muted: boolean;
}): JSX.Element {
  const size = 26;
  const n = grain === 0 ? 1 : grain === 1 ? 3 : 6;
  const gap = grain === 0 ? 0 : 1.2;
  const cell = (size - gap * (n - 1)) / n;
  const cells = [];
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      cells.push(
        <rect
          key={`${i}-${j}`}
          x={x - size / 2 + j * (cell + gap)}
          y={y - size / 2 + i * (cell + gap)}
          width={cell}
          height={cell}
          rx={grain === 0 ? 3 : 0.6}
          fill={colour}
          opacity={muted ? 0.35 : 1}
        />,
      );
  return (
    <g>
      {active && (
        <rect
          x={x - size / 2 - 4}
          y={y - size / 2 - 4}
          width={size + 8}
          height={size + 8}
          rx={5}
          fill="none"
          stroke={STATE_COLOUR.active}
          strokeWidth={2.5}
        />
      )}
      {cells}
    </g>
  );
}

/** Where each family sits: its cell, offset when two share one. */
function positions(
  steps: MapStep[],
): { x: number; y: number; above: boolean }[] {
  const seen: Record<string, number> = {};
  const count: Record<string, number> = {};
  for (const s of steps) {
    const k = `${s.placement},${s.schedule}`;
    count[k] = (count[k] ?? 0) + 1;
  }
  return steps.map((s) => {
    const k = `${s.placement},${s.schedule}`;
    const idx = seen[k] ?? 0;
    seen[k] = idx + 1;
    const n = count[k]!;
    const cx = LEFT + (s.placement + 0.5) * COLW;
    const dx = n === 1 ? 0 : (idx - (n - 1) / 2) * 34;
    return {
      x: cx + dx,
      y: TOP + (s.schedule + 0.5) * ROWH - 4,
      above: n > 1 && idx === 0,
    };
  });
}

export default function MapWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [mid, setMid] = useState<ModelId>("llama3-70b");
  const [hover, setHover] = useState<string | null>(null);
  const font = useSvgFont(W);
  const fs = font.fs;
  const steps = useMemo(() => mapSteps(mid, 1, 1024), [mid]);
  const pos = useMemo(() => positions(steps), [steps]);
  const st = useStepper(steps.length, { stepMs: 1800, resetKey: mid });
  const k = st.step;
  const s = steps[k]!;
  const modelName = shape(mid).name;
  const partName = s.part
    ? (part(s.part as never).name as string).split(" (")[0]!
    : "";

  const visual = (
    <div className="mx-auto max-w-lg">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full"
        role="img"
        aria-label={`Map of ${steps.length} chip families by where the weights live and who schedules data movement; family ${k + 1}, ${s.name}, highlighted.`}
      >
        {/* columns: where the weights live */}
        {PLACEMENT_LABELS.map((_, c) => (
          <g key={c}>
            <rect
              x={LEFT + c * COLW + 1}
              y={TOP}
              width={COLW - 2}
              height={3 * ROWH}
              fill={PLACEMENT_COLOUR[c]}
              opacity={0.1}
            />
            <text
              x={LEFT + (c + 0.5) * COLW}
              y={TOP - 8}
              textAnchor="middle"
              style={{ fontSize: fs(10) }}
              className="fill-neutral-700 dark:fill-neutral-300"
            >
              {SHORT_PLACE[c]}
            </text>
          </g>
        ))}
        {/* rows: who schedules */}
        {SCHEDULE_LABELS.map((_, r) => (
          <g key={r}>
            <line
              x1={LEFT}
              x2={W - 6}
              y1={TOP + r * ROWH}
              y2={TOP + r * ROWH}
              stroke="#a3a3a3"
              strokeDasharray="2 3"
            />
            <text
              x={LEFT - 6}
              y={TOP + (r + 0.5) * ROWH}
              textAnchor="end"
              style={{ fontSize: fs(10) }}
              className="fill-neutral-700 dark:fill-neutral-300"
            >
              {SHORT_SCHED[r]}
            </text>
          </g>
        ))}
        {steps.map((f, i) => (
          <g key={f.id} data-family={f.id} data-active={i === k}>
            <Marker
              x={pos[i]!.x}
              y={pos[i]!.y}
              grain={f.grain}
              colour={PLACEMENT_COLOUR[f.placement]!}
              active={i === k}
              muted={i > k}
            />
            <text
              x={pos[i]!.x}
              y={pos[i]!.above ? pos[i]!.y - 19 : pos[i]!.y + 27}
              textAnchor="middle"
              style={{ fontSize: fs(9.5) }}
              className={
                i === k
                  ? "fill-neutral-950 font-semibold dark:fill-white"
                  : "fill-neutral-600 dark:fill-neutral-400"
              }
            >
              {SHORT_NAME[f.id] ?? f.name}
            </text>
          </g>
        ))}
      </svg>
      <ChipStrip s={s} partName={partName} />
      <p className="mt-1 text-[0.7rem] text-neutral-600 dark:text-neutral-400">
        Columns: where the weights live while a token is generated. Rows: who
        decides when data moves. A marker&apos;s squares show the grain: one
        large engine, tens to hundreds of cores, or a carpet of tiny ones.
        Families ahead of the highlighted one are faded.
      </p>
    </div>
  );

  const stats = (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
      <Stat label="Family" value={s.name} hint={`chapter ${s.chapter}`} />
      <Stat label="Weights live in" value={PLACEMENT_LABELS[s.placement]!} />
      <Stat
        label="Model + 1k KV"
        value={s.part ? fmtBytes(s.need) : "–"}
        hint={s.part ? s.fmt.toUpperCase() : "numbers in a later chapter"}
      />
      <Stat
        label="Chips to hold it"
        value={s.part ? fmtInt(s.chips) : "–"}
        hint={s.part ? `${fmtBytes(s.cap)} each` : undefined}
      />
    </div>
  );

  return (
    <AnimationPanel
      testId="map-widget"
      title="The families of AI chips, on two axes"
      summary="Each family placed by where its weights live and who schedules the data; for the families of chapters 2 to 5, how many chips hold the running model."
      stepper={st}
      stepLabel="family"
      caption={mapCaption(s, steps.length, modelName, partName)}
      visual={visual}
      stats={stats}
      equation={children}
      hl={hover ?? (s.part ? "need cap" : "")}
      onEquationHover={setHover}
      params={
        <Segmented
          label="Running model"
          value={mid}
          options={[
            { value: "llama3-8b", label: "Llama 3.1 8B" },
            { value: "llama3-70b", label: "Llama 3.1 70B" },
          ]}
          onChange={(v) => setMid(v as ModelId)}
        />
      }
    />
  );
}

/** The chips a family's part needs to hold the model: one square per chip. */
function ChipStrip({
  s,
  partName,
}: {
  s: MapStep;
  partName: string;
}): JSX.Element {
  const n = s.chips;
  const per = n > 120 ? 40 : n > 40 ? 20 : 10;
  const rows = Math.max(1, Math.ceil(n / per));
  const cell = n > 120 ? 6 : 12;
  const w = per * (cell + 2);
  const h = rows * (cell + 2);
  return (
    <div
      className="mt-2 min-h-16 rounded bg-white p-2 ring-1 ring-neutral-200 dark:bg-neutral-950 dark:ring-neutral-800"
      data-chips={n}
    >
      {s.part ? (
        <>
          <p className="text-xs text-neutral-700 dark:text-neutral-300">
            {fmtInt(n)} × {partName}
          </p>
          <svg
            viewBox={`0 0 ${w} ${h}`}
            className="mt-1 h-auto max-h-28"
            style={{ width: `${Math.min(100, (w / 400) * 100 + 10)}%` }}
            role="img"
            aria-label={`${n} chips`}
          >
            {Array.from({ length: n }, (_, i) => (
              <rect
                key={i}
                x={(i % per) * (cell + 2)}
                y={Math.floor(i / per) * (cell + 2)}
                width={cell}
                height={cell}
                rx={1}
                fill={PLACEMENT_COLOUR[s.placement]}
              />
            ))}
          </svg>
        </>
      ) : (
        <p className="text-xs text-neutral-600 dark:text-neutral-400">
          This family&apos;s parts and numbers come in chapter {s.chapter}.
        </p>
      )}
    </div>
  );
}
