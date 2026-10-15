"use client";

/**
 * Chapter 9's hero: one decoder layer for one token on analogue in-memory
 * crossbars. Left, one crossbar, drawn 8 x 8 for the chip's 256 x 256: the
 * weights sit in it as conductances, inputs arrive on the rows through
 * DACs, the products sum as currents down the columns and an ADC at the
 * foot of each column reads the result. Right, the layer's four weight
 * matrices as counts of crossbar tiles. The animation programs the weights,
 * then runs the four matrix stages; attention runs off the crossbars, in
 * digital logic, because the KV cache changes every token. Every frame is a
 * state of `imcSteps` (checked against the Python reference).
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Segmented, Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { fmtInt, fmtTime, trim } from "@/lib/format";
import { imcCaption } from "@/lib/accel/captions";
import {
  imcLayer,
  imcSteps,
  shape,
  xorshift32,
  type ModelId,
} from "@/lib/accel/model";
import { MUTED, STATE_COLOUR, TERM_COLOUR } from "@/lib/viz/palette";

const W = 360;
const N = 8;
const SHORT = ["QKV", "out", "up+gate", "down"];

/** Fixed, seeded conductance shades for the drawing (not data). */
function shades(): number[] {
  let x = 7;
  const out: number[] = [];
  for (let i = 0; i < N * N; i++) {
    x = xorshift32(x);
    out.push(0.25 + (x % 1000) / 1333);
  }
  return out;
}
const SHADES = shades();

export default function CrossbarWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [mid, setMid] = useState<ModelId>("llama3-8b");
  const [phase, setPhase] = useState<number>(1);
  const [hover, setHover] = useState<string | null>(null);
  const font = useSvgFont(W);
  const fs = font.fs;
  const m = useMemo(() => imcLayer(mid, phase), [mid, phase]);
  const steps = useMemo(() => imcSteps(mid, phase), [mid, phase]);
  const st = useStepper(steps.length, {
    stepMs: 1500,
    resetKey: `${mid}-${phase}`,
  });
  const k = st.step;
  const s = steps[k]!;
  const mvm = s.gemm >= 0;
  const attn = s.name === "attention";

  const XB = 150;
  const X0 = 26;
  const Y0 = fs(12) + 18;
  const cell = (XB - 24) / N;
  const adcY = Y0 + N * cell + 6;
  const RX = X0 + XB + 26;
  const RW = W - RX - 6;
  let maxTiles = 0;
  for (const g of m.gemms) if (g.tiles > maxTiles) maxTiles = g.tiles;
  const rowH = Math.max(22, fs(12) + 10);
  const digY = adcY + 30;
  const H = Math.max(digY + 30, Y0 + 4 * rowH + 2 * fs(11) + 10);

  const xbar = [];
  for (let r = 0; r < N; r++)
    for (let c = 0; c < N; c++)
      xbar.push(
        <rect
          key={`${r}-${c}`}
          x={X0 + 12 + c * cell + 1}
          y={Y0 + r * cell + 1}
          width={cell - 2}
          height={cell - 2}
          rx={1}
          fill={TERM_COLOUR.comp}
          opacity={k === 0 ? SHADES[r * N + c]! * 0.6 : SHADES[r * N + c]!}
        />,
      );

  const visual = (
    <div className="mx-auto max-w-lg">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full"
        role="img"
        aria-label={`A crossbar and the layer's weight matrices as crossbar tiles; step ${k + 1}, ${s.name}.`}
      >
        <text
          x={X0}
          y={fs(11)}
          style={{ fontSize: fs(10.5) }}
          className="fill-neutral-800 font-medium dark:fill-neutral-200"
        >
          one crossbar, drawn 8 x 8
        </text>
        {/* DACs on the rows, lit when inputs are applied */}
        {Array.from({ length: N }, (_, r) => (
          <g key={r} data-row={r}>
            <polygon
              points={`${X0},${Y0 + r * cell + 2} ${X0 + 9},${Y0 + (r + 0.5) * cell} ${X0},${Y0 + (r + 1) * cell - 2}`}
              fill={mvm ? STATE_COLOUR.active : MUTED.light}
            />
            <line
              x1={X0 + 9}
              x2={X0 + 12 + N * cell}
              y1={Y0 + (r + 0.5) * cell}
              y2={Y0 + (r + 0.5) * cell}
              stroke={mvm ? STATE_COLOUR.active : MUTED.light}
              strokeWidth={mvm ? 1.2 : 0.6}
              opacity={0.6}
            />
          </g>
        ))}
        <g data-crossbar={mvm ? "active" : attn ? "idle" : "programming"}>
          {xbar}
        </g>
        {/* column currents and ADCs */}
        {Array.from({ length: N }, (_, c) => (
          <g key={c} data-col={c}>
            <line
              x1={X0 + 12 + (c + 0.5) * cell}
              x2={X0 + 12 + (c + 0.5) * cell}
              y1={Y0}
              y2={adcY}
              stroke={mvm ? TERM_COLOUR.hop : MUTED.light}
              strokeWidth={mvm ? 1.4 : 0.6}
              opacity={0.7}
            />
            <rect
              x={X0 + 12 + c * cell + 2}
              y={adcY}
              width={cell - 4}
              height={10}
              rx={1}
              fill={mvm ? TERM_COLOUR.hop : MUTED.light}
            />
          </g>
        ))}
        <text
          x={X0 + 12}
          y={adcY + 10 + fs(11)}
          style={{ fontSize: fs(9.5) }}
          className="fill-neutral-700 dark:fill-neutral-300"
        >
          ADCs, one per output line
        </text>
        {/* the digital unit where attention runs */}
        <rect
          x={X0}
          y={digY}
          width={XB}
          height={26}
          rx={3}
          fill={attn ? TERM_COLOUR.kv : "none"}
          stroke={attn ? TERM_COLOUR.kv : MUTED.light}
          data-digital={attn ? "active" : "idle"}
        />
        <text
          x={X0 + XB / 2}
          y={digY + 17}
          textAnchor="middle"
          style={{ fontSize: fs(9.5) }}
          className={
            attn ? "fill-neutral-950" : "fill-neutral-600 dark:fill-neutral-400"
          }
        >
          digital: attention
        </text>
        {/* the layer's matrices as tile counts */}
        <text
          x={RX}
          y={fs(11)}
          style={{ fontSize: fs(10.5) }}
          className="fill-neutral-800 font-medium dark:fill-neutral-200"
        >
          tiles per layer
        </text>
        {m.gemms.map((g, i) => {
          const y = Y0 + i * rowH;
          const on = s.gemm === i;
          const done = steps.findIndex((x) => x.gemm === i) < k;
          const w = (g.tiles / maxTiles) * (RW - 4);
          return (
            <g
              key={g.name}
              data-gemm={i}
              data-state={on ? "active" : done ? "done" : "waiting"}
            >
              <rect
                x={RX}
                y={y + fs(11) + 2}
                width={Math.max(2, w)}
                height={rowH - fs(11) - 6}
                rx={1.5}
                fill={on ? STATE_COLOUR.active : TERM_COLOUR.comp}
                opacity={on ? 1 : done ? 0.75 : 0.35}
              />
              <text
                x={RX}
                y={y + fs(10)}
                style={{ fontSize: fs(9.5) }}
                className={
                  on
                    ? "fill-neutral-950 font-semibold dark:fill-white"
                    : "fill-neutral-600 dark:fill-neutral-400"
                }
              >
                {SHORT[i]} {fmtInt(g.tiles)}
              </text>
            </g>
          );
        })}
        <text
          x={RX}
          y={Y0 + 4 * rowH + fs(11)}
          style={{ fontSize: fs(9.5) }}
          className="fill-neutral-700 dark:fill-neutral-300"
        >
          {trim(m.layer_chips)} chips per layer
        </text>
        <text
          x={RX}
          y={Y0 + 4 * rowH + 2 * fs(11) + 4}
          style={{ fontSize: fs(9.5) }}
          className="fill-neutral-700 dark:fill-neutral-300"
        >
          {fmtInt(m.chips)} for the model
        </text>
      </svg>
      <p className="mt-1 text-[0.7rem] text-neutral-600 dark:text-neutral-400">
        Yellow cells are weights stored as conductances (shades are for the
        drawing). Blue rows are inputs through DACs; orange columns are the
        summed currents and their ADCs. Bars on the right count the 256 x 256
        crossbar tiles each of the layer&apos;s matrices takes.
      </p>
    </div>
  );

  const stats = (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
      <Stat
        label="Chips for the model"
        value={fmtInt(m.chips)}
        hint="64 crossbars each"
      />
      <Stat label="Tiles per layer" value={fmtInt(m.layer_tiles)} />
      <Stat
        label="Crossbar time"
        value={fmtTime(s.t)}
        hint={`of ${fmtTime(m.t_layer)} per layer`}
      />
      <Stat
        label="Crossbar energy"
        value={`${trim(s.e * 1e6)} µJ`}
        hint="this layer so far"
      />
    </div>
  );

  return (
    <AnimationPanel
      testId="crossbar-widget"
      title="One decoder layer on analogue crossbars"
      summary={`${shape(mid).name} mapped onto the HERMES research chip's 256 x 256 phase-change crossbars: the weights never move, and each matrix stage is one analogue multiply. Times and energies are the paper's measured per-MVM figures; attention and chip-to-chip transfers are not counted.`}
      stepper={st}
      stepLabel="stage"
      countFrom={0}
      caption={imcCaption(s, m, shape(mid).name)}
      visual={visual}
      stats={stats}
      equation={children}
      hl={hover ?? (mvm ? "tiles adc" : "")}
      onEquationHover={setHover}
      params={
        <div className="flex flex-wrap gap-4">
          <Segmented
            label="Read mode"
            value={String(phase)}
            options={[
              { value: "1", label: "1-phase (fast)" },
              { value: "4", label: "4-phase (precise)" },
            ]}
            onChange={(v) => setPhase(Number(v))}
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
