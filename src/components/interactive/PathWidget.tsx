"use client";

/**
 * Chapter 10's critical-path animation: one token through every layer of a
 * model on an SRAM machine. Each layer costs its share of the bound (the
 * weight reads or compute the model counts: tiny once weights sit in SRAM)
 * plus a fixed floor per layer (synchronisation, collectives and mesh
 * traversal, which the model does not split up), fitted on the other three
 * systems. Every frame is a state of `criticalPathSteps`; the log-scale
 * panel underneath sets the bound, the bound plus the floor and the
 * published time per token side by side for all four systems
 * (`floorFit`, checked against the Python reference).
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Hatch } from "@/components/interactive/Hatch";
import { Segmented, Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { fmtTime, trim } from "@/lib/format";
import { pathCaption } from "@/lib/accel/captions";
import {
  FLOOR_CHECKS,
  criticalPathSteps,
  floorFit,
  type FloorCheck,
  type FloorMode,
} from "@/lib/accel/model";
import { MUTED, STATE_COLOUR, TERM_COLOUR } from "@/lib/viz/palette";

const W = 360;
const SHORT: Record<FloorCheck, string> = {
  "groq-70b-tps": "Groq 70B",
  "cerebras-70b-tps": "Cerebras 70B",
  "cerebras-8b-tps": "Cerebras 8B",
  "sn40l-8b-tps": "SN40L 8B",
};

export default function PathWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [id, setId] = useState<FloorCheck>("cerebras-70b-tps");
  const [mode, setMode] = useState<FloorMode>("fitted");
  const [hover, setHover] = useState<string | null>(null);
  const font = useSvgFont(W);
  const fs = font.fs;
  const fit = useMemo(() => floorFit(), []);
  const r = fit.rows.find((x) => x.id === id)!;
  const steps = useMemo(() => criticalPathSteps(id, mode), [id, mode]);
  const st = useStepper(steps.length, {
    stepMs: Math.max(80, Math.round(8000 / steps.length)),
    resetKey: `${id}-${mode}`,
  });
  const s = steps[st.step]!;
  const L = r.layers;
  const total = steps[L]!.t;

  // the layers: one square each, split work | floor
  const perRow = L > 40 ? 20 : 16;
  const rows = Math.ceil(L / perRow);
  const cw = (W - 12) / perRow;
  const ch = Math.min(cw, 20);
  const gridH = rows * ch + 4;
  const workFrac =
    r.t_model /
    L /
    (r.t_model / L +
      (mode === "none" ? 0 : mode === "own" ? r.per_layer : r.tauLoo));
  // the token's time bar
  const barY = gridH + 24;
  const scale = Math.max(total, r.t_published);
  const bx = (t: number) => (t / scale) * (W - 12);
  // the log panel: all four systems
  const logY = barY + 44;
  const rowH = font.narrow ? 26 : 20;
  const lo = 1e-7;
  const hi = 1e-2;
  const lx = (t: number) =>
    86 +
    ((Math.log10(Math.max(t, lo)) - Math.log10(lo)) /
      (Math.log10(hi) - Math.log10(lo))) *
      (W - 96);
  const H = logY + FLOOR_CHECKS.length * rowH + 30;

  const visual = (
    <div className="mx-auto max-w-lg">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full"
        role="img"
        aria-label={`${r.label}: layer ${s.layer} of ${L}, ${fmtTime(s.t)} into a token; published ${fmtTime(r.t_published)} per token.`}
        data-layer={s.layer}
      >
        <defs>
          <Hatch id="path-hatch" />
        </defs>
        {Array.from({ length: L }, (_, i) => {
          const x = 6 + (i % perRow) * cw + 0.6;
          const y = 2 + Math.floor(i / perRow) * ch + 0.6;
          const w = cw - 1.2;
          const done = i < s.layer - 1;
          const cur = i === s.layer - 1;
          return (
            <g
              key={i}
              data-l={i}
              data-state={cur ? "active" : done ? "done" : "waiting"}
              opacity={cur || done ? 1 : 0.3}
            >
              <rect
                x={x}
                y={y}
                width={w}
                height={ch - 1.2}
                rx={1}
                fill="url(#path-hatch)"
                stroke={cur ? STATE_COLOUR.active : "none"}
                strokeWidth={1.5}
              />
              <rect
                x={x}
                y={y}
                width={Math.max(0.6, w * workFrac)}
                height={ch - 1.2}
                rx={1}
                fill={TERM_COLOUR.sram}
              />
            </g>
          );
        })}
        <text
          x={6}
          y={gridH + fs(12)}
          style={{ fontSize: fs(10) }}
          className="fill-neutral-800 font-medium dark:fill-neutral-200"
        >
          this token so far: {fmtTime(s.t)}
        </text>
        <rect
          x={6}
          y={barY}
          width={W - 12}
          height={14}
          fill="none"
          stroke={MUTED.light}
        />
        <rect
          x={6}
          y={barY}
          width={bx(s.work)}
          height={14}
          fill={TERM_COLOUR.sram}
          data-term="work"
        />
        <rect
          x={6 + bx(s.work)}
          y={barY}
          width={bx(s.floor)}
          height={14}
          fill="url(#path-hatch)"
          data-term="floor"
        />
        <line
          x1={6 + bx(r.t_published)}
          x2={6 + bx(r.t_published)}
          y1={barY - 4}
          y2={barY + 18}
          stroke="currentColor"
          strokeWidth={1.5}
        />
        <text
          x={Math.min(6 + bx(r.t_published), W - 60)}
          y={barY + 30}
          style={{ fontSize: fs(9) }}
          className="fill-neutral-700 dark:fill-neutral-300"
        >
          published
        </text>
        {/* every system, time per token, log axis */}
        {[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2].map((t) => (
          <g key={t}>
            <line
              x1={lx(t)}
              x2={lx(t)}
              y1={logY - 4}
              y2={logY + FLOOR_CHECKS.length * rowH}
              stroke={MUTED.light}
              strokeWidth={0.4}
            />
            <text
              x={lx(t)}
              y={logY + FLOOR_CHECKS.length * rowH + 12}
              textAnchor={t === hi ? "end" : t === lo ? "start" : "middle"}
              style={{ fontSize: fs(8) }}
              className="fill-neutral-600 dark:fill-neutral-400"
            >
              {fmtTime(t)}
            </text>
          </g>
        ))}
        {fit.rows.map((x, j) => {
          const y = logY + j * rowH + rowH / 2;
          const on = x.id === id;
          return (
            <g key={x.id} data-system={x.id} opacity={on ? 1 : 0.55}>
              <text
                x={2}
                y={y + 3}
                style={{ fontSize: fs(9) }}
                className={
                  on
                    ? "fill-neutral-900 font-semibold dark:fill-neutral-100"
                    : "fill-neutral-700 dark:fill-neutral-300"
                }
              >
                {SHORT[x.id]}
              </text>
              <line
                x1={lx(x.t_model)}
                x2={lx(x.t_published)}
                y1={y}
                y2={y}
                stroke={MUTED.light}
                strokeWidth={1}
              />
              <circle cx={lx(x.t_model)} cy={y} r={3.5} fill={TERM_COLOUR.sram}>
                <title>{`bound ${fmtTime(x.t_model)}`}</title>
              </circle>
              <rect
                x={lx(x.tPred) - 3.5}
                y={y - 3.5}
                width={7}
                height={7}
                fill={TERM_COLOUR.hop}
                transform={`rotate(45 ${lx(x.tPred)} ${y})`}
              >
                <title>{`bound + floor (fitted on the others) ${fmtTime(x.tPred)}`}</title>
              </rect>
              <line
                x1={lx(x.t_published)}
                x2={lx(x.t_published)}
                y1={y - 6}
                y2={y + 6}
                stroke="currentColor"
                strokeWidth={2}
              >
                <title>{`published ${fmtTime(x.t_published)}`}</title>
              </line>
            </g>
          );
        })}
      </svg>
      <p className="mt-1 text-[0.7rem] text-neutral-600 dark:text-neutral-400">
        Squares: the token&apos;s layers, green the bound&apos;s share of each,
        hatched the per-layer floor. Below, time per token on a log axis: green
        dot the bound, orange diamond the bound plus the floor fitted on the
        other three systems, black tick the published time.
      </p>
    </div>
  );

  return (
    <AnimationPanel
      testId="path-widget"
      title="One token's critical path, layer by layer"
      summary={`${r.label}: the bound's work per layer against a fixed per-layer floor. The floor is fitted on the other three systems, so each prediction is a test.`}
      stepper={st}
      stepLabel="layer"
      caption={pathCaption(s, r, mode)}
      visual={visual}
      equation={children}
      hl={hover ?? (mode === "none" ? "sram" : "hop")}
      onEquationHover={setHover}
      countFrom={0}
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="Bound" value={fmtTime(r.t_model)} hint="per token" />
          <Stat
            label="Bound + floor"
            value={fmtTime(r.tPred)}
            hint={`floor ${fmtTime(r.tauLoo)} a layer`}
          />
          <Stat label="Published" value={fmtTime(r.t_published)} />
          <Stat
            label="Published / predicted"
            value={`${trim(r.ratioPred)}×`}
            hint={`bound alone: ${trim(r.ratioBound)}×`}
          />
        </div>
      }
      params={
        <>
          <Segmented
            label="System"
            value={id}
            options={FLOOR_CHECKS.map((c) => ({ value: c, label: SHORT[c] }))}
            onChange={setId}
          />
          <Segmented
            label="Floor per layer"
            value={mode}
            options={[
              { value: "fitted", label: "fitted on the others" },
              { value: "own", label: "this system's own" },
              { value: "none", label: "none (the bound)" },
            ]}
            onChange={setMode}
          />
        </>
      }
    />
  );
}
