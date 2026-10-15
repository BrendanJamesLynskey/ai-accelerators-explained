"use client";

/**
 * Chapter 3's hero: a matmul tiled onto one weight-stationary array. The
 * 1,024 x 1,024 weight matrix is cut into array-sized tiles that take turns
 * in the array; while one tile works, the next one's weights shift in behind
 * it. On the left, the array itself: its rows of PEs, and how many of them
 * the batch's input rows keep busy. Every frame is a state of `mxuSteps`
 * (checked against the Python reference).
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Segmented, Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { fmtInt, pct } from "@/lib/format";
import { mxuCaption } from "@/lib/accel/captions";
import { layerMxu, mxuSteps } from "@/lib/accel/model";
import {
  LEVEL_COLOUR,
  MUTED,
  STATE_COLOUR,
  TERM_COLOUR,
} from "@/lib/viz/palette";

const K = 1024;
const N = 1024;
const W = 360;
const H = 200;
const GX = 150; // the weight grid's left edge
const GS = 190; // the weight grid's size

export default function MxuWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [arr, setArr] = useState<string>("256");
  const [mS, setM] = useState<string>("1");
  const [hover, setHover] = useState<string | null>(null);
  const R = Number(arr);
  const m = Number(mS);
  const font = useSvgFont(W);
  const fs = font.fs;
  const steps = useMemo(() => mxuSteps(R, R, m, K, N), [R, m]);
  const layer = useMemo(() => layerMxu(R, R, "llama3-70b", m), [R, m]);
  const st = useStepper(steps.length, {
    stepMs: R === 128 ? 450 : 900,
    resetKey: `${R}|${m}`,
  });
  const s = steps[st.step]!;
  const tiles = steps.length - 1;
  const tc = Math.ceil(N / R);
  const cell = GS / tc;
  const busy = Math.min(m, R) / R;

  const visual = (
    <div className="mx-auto max-w-lg">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full"
        role="img"
        aria-label={`Weight matrix in ${tiles} tiles of ${R} by ${R}; tile ${Math.max(0, s.tile + 1)} in the array; ${pct(s.util, 1)} of the array used so far.`}
      >
        {/* the array: rows of PEs, the busy share shaded */}
        <text
          x={8}
          y={fs(12)}
          style={{ fontSize: fs(11) }}
          className="fill-neutral-800 font-medium dark:fill-neutral-200"
        >
          {font.narrow ? `array ${R}²` : `the array (${R} x ${R})`}
        </text>
        <rect
          x={8}
          y={24}
          width={110}
          height={110}
          fill="none"
          stroke="#737373"
        />
        <rect
          x={8}
          y={24}
          width={110}
          height={Math.max(1.5, 110 * busy)}
          fill={STATE_COLOUR.active}
          opacity={s.tile < 0 ? 0.25 : 0.85}
          data-busy={busy}
        />
        <text
          x={8}
          y={150}
          style={{ fontSize: fs(10) }}
          className="fill-neutral-700 dark:fill-neutral-300"
        >
          {font.narrow
            ? `busy ${Math.min(m, R)}/${Math.max(m, R)}`
            : `busy ${Math.min(m, R)} of every ${Math.max(m, R)} cycles`}
        </text>
        {!font.narrow && (
          <text
            x={8}
            y={150 + fs(14)}
            style={{ fontSize: fs(10) }}
            className="fill-neutral-700 dark:fill-neutral-300"
          >
            (batch {m}; a tile loads in {R})
          </text>
        )}
        {/* the weight matrix, tile by tile */}
        <text
          x={GX}
          y={fs(12)}
          style={{ fontSize: fs(11) }}
          className="fill-neutral-800 font-medium dark:fill-neutral-200"
        >
          {font.narrow ? "weights" : `weights, ${K} x ${N}`}
        </text>
        {Array.from({ length: tiles }, (_, t) => {
          const a = Math.floor(t / tc);
          const c = t % tc;
          const done = t < s.tile;
          const cur = t === s.tile;
          const next = t === s.tile + 1;
          return (
            <rect
              key={t}
              data-tile={t}
              data-state={
                cur ? "active" : done ? "done" : next ? "loading" : "waiting"
              }
              x={GX + c * cell + 1}
              y={24 + a * cell + 1}
              width={cell - 2}
              height={cell - 2}
              rx={1.5}
              fill={
                cur
                  ? STATE_COLOUR.active
                  : done
                    ? MUTED.light
                    : LEVEL_COLOUR.hbm
              }
              opacity={cur ? 1 : done ? 0.7 : next ? 0.75 : 0.25}
              stroke={next ? TERM_COLOUR.sram : "none"}
              strokeWidth={next ? 2 : 0}
            />
          );
        })}
      </svg>
      <p className="mt-1 text-[0.7rem] text-neutral-600 dark:text-neutral-400">
        Blue: the tile in the array. Green outline: the next tile, its weights
        shifting in behind it. Grey: done. Purple: still waiting in memory. The
        blue share of the array box is the share of each tile&apos;s turn spent
        multiplying: the batch&apos;s rows take m cycles to stream, the next
        tile takes {R} to load, and the slower of the two sets the turn.
      </p>
    </div>
  );

  const stats = (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
      <Stat label="Tiles" value={`${Math.max(0, s.tile + 1)} / ${tiles}`} />
      <Stat label="Cycles so far" value={fmtInt(s.cycles)} />
      <Stat label="Array used" value={pct(s.util, 1)} hint="this matmul" />
      <Stat
        label="One 70B layer"
        value={pct(layer.util, 1)}
        hint={`${fmtInt(layer.cycles)} cycles`}
      />
    </div>
  );

  return (
    <AnimationPanel
      testId="mxu-widget"
      title="A matmul, tile by tile, on one systolic array"
      summary="A weight-stationary array holds one tile of weights while the batch's input rows stream through it, and loads the next tile behind it. A batch smaller than the array leaves most of its rows idle."
      stepper={st}
      stepLabel="tile"
      caption={mxuCaption(s, tiles, R, R, m)}
      visual={visual}
      stats={stats}
      equation={children}
      hl={hover ?? (s.tile < 0 ? "fill" : "tiles per")}
      onEquationHover={setHover}
      countFrom={0}
      params={
        <>
          <Segmented
            label="Array"
            value={arr}
            options={[
              { value: "256", label: "256 x 256 (TPU v6e, TPU7x)" },
              { value: "128", label: "128 x 128 (Trainium2)" },
            ]}
            onChange={setArr}
          />
          <Segmented
            label="Batch (input rows)"
            value={mS}
            options={["1", "16", "128", "256", "512"].map((v) => ({
              value: v,
              label: v,
            }))}
            onChange={setM}
          />
        </>
      }
    />
  );
}
