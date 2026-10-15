"use client";

/**
 * Chapter 6's hero: one decoder layer run twice on one SN40L socket. In the
 * top lane each operation is its own kernel: it reads its input from HBM
 * and writes its result back, attention scores included. In the bottom lane
 * the layer is fused into one dataflow pipeline: results pass from stage to
 * stage through on-chip buffers, and HBM sees only the weights, the KV cache
 * and the layer's input and output. The bars under each operation are the
 * bytes it moves to and from HBM; the two totals underneath grow as the
 * animation steps through the layer. Every frame is a state of
 * `dataflowSteps` (checked against the Python reference).
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Segmented, Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { fmtBytes, fmtTime, pct } from "@/lib/format";
import { dataflowCaption } from "@/lib/accel/captions";
import {
  dataflowSteps,
  layerTraffic,
  part,
  type ModelId,
} from "@/lib/accel/model";
import { MUTED, STATE_COLOUR, TERM_COLOUR } from "@/lib/viz/palette";

const W = 360;
const LEFT = 6;
const N_OPS = 11;
const BOX = (W - 2 * LEFT) / N_OPS;
const SHORT = [
  "norm",
  "QKV",
  "QKᵀ",
  "·V",
  "out",
  "add",
  "norm",
  "up",
  "act",
  "down",
  "add",
];

/** (tokens, sequences, context, causal) */
const CASES: Record<string, [number, number, number, boolean, string]> = {
  prefill: [2048, 1, 2048, true, "Prefill, 2,048 tokens"],
  b64: [64, 64, 1024, false, "Decode, batch 64"],
  b1: [1, 1, 8192, false, "Decode, batch 1"],
};

export default function DataflowWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [mid, setMid] = useState<ModelId>("llama3-8b");
  const [ck, setCk] = useState<string>("prefill");
  const [hover, setHover] = useState<string | null>(null);
  const font = useSvgFont(W);
  const fs = font.fs;
  const [tokens, seqs, ctx, causal] = CASES[ck]!;
  const steps = useMemo(
    () => dataflowSteps("sn40l", mid, "bf16", tokens, seqs, ctx, causal),
    [mid, tokens, seqs, ctx, causal],
  );
  const u = useMemo(
    () => layerTraffic("sn40l", mid, "bf16", tokens, seqs, ctx, causal, false),
    [mid, tokens, seqs, ctx, causal],
  );
  const f = useMemo(
    () => layerTraffic("sn40l", mid, "bf16", tokens, seqs, ctx, causal, true),
    [mid, tokens, seqs, ctx, causal],
  );
  const st = useStepper(steps.length, {
    stepMs: 1100,
    resetKey: `${mid}-${ck}`,
  });
  const k = st.step;
  const s = steps[k]!;
  const partName = (part("sn40l").name as string).replace("SambaNova ", "");
  let maxOp = 0;
  for (const x of steps) if (x.u_op > maxOp) maxOp = x.u_op;
  const BAR = 34;
  // on a phone the labels turn sideways in taller boxes
  const BH = font.narrow ? 40 : 20;
  const lane = (top: number) => ({
    box: top + fs(13) + 2,
    bar: top + fs(13) + BH + 6,
  });
  const L1 = lane(2);
  const L2 = lane(2 + fs(13) + BH + 6 + BAR + 14);
  const totY = L2.bar + BAR + 16;
  const H = totY + 2 * (fs(12) + 14) + 6;
  const tw = W - 2 * LEFT;

  const laneDraw = (
    y: { box: number; bar: number },
    which: "u" | "f",
  ): JSX.Element[] =>
    steps.map((x, i) => {
      const state = i === k ? "active" : i < k ? "done" : "waiting";
      const v = which === "u" ? x.u_op : x.f_op;
      const h = maxOp > 0 ? (v / maxOp) * BAR : 0;
      const x0 = LEFT + i * BOX;
      return (
        <g
          key={`${which}-${i}`}
          data-lane={which}
          data-op={i}
          data-state={state}
        >
          <rect
            x={x0 + 1}
            y={y.box}
            width={BOX - 2}
            height={BH}
            rx={2}
            fill={
              state === "active"
                ? STATE_COLOUR.active
                : x.kind === "matmul"
                  ? STATE_COLOUR.compute
                  : MUTED.light
            }
            opacity={state === "waiting" ? 0.35 : state === "done" ? 0.7 : 1}
          />
          <text
            x={x0 + BOX / 2}
            y={font.narrow ? y.box + BH / 2 : y.box + 14}
            textAnchor="middle"
            dominantBaseline={font.narrow ? "central" : undefined}
            transform={
              font.narrow
                ? `rotate(-90 ${x0 + BOX / 2} ${y.box + BH / 2})`
                : undefined
            }
            style={{ fontSize: fs(8.5) }}
            className={
              state === "active"
                ? "fill-white font-semibold"
                : "fill-neutral-900"
            }
          >
            {SHORT[i]}
          </text>
          {/* fused: results pass to the next stage on chip */}
          {which === "f" && i < steps.length - 1 && (
            <line
              x1={x0 + BOX - 1}
              x2={x0 + BOX + 1}
              y1={y.box + BH / 2}
              y2={y.box + BH / 2}
              stroke={TERM_COLOUR.sram}
              strokeWidth={3}
            />
          )}
          {/* bytes to and from HBM */}
          <rect
            x={x0 + BOX * 0.2}
            y={y.bar + BAR - h}
            width={BOX * 0.6}
            height={Math.max(h, v > 0 ? 0.8 : 0)}
            fill={TERM_COLOUR.mem}
            opacity={i <= k ? 1 : 0.25}
          />
        </g>
      );
    });

  const total = (
    y: number,
    label: string,
    b: number,
    t: number,
    of: number,
  ) => (
    <g>
      <text
        x={LEFT}
        y={y}
        style={{ fontSize: fs(10) }}
        className="fill-neutral-800 dark:fill-neutral-200"
      >
        {label}: {fmtBytes(b)}, {fmtTime(t)}
      </text>
      <rect
        x={LEFT}
        y={y + 4}
        width={tw}
        height={8}
        fill="none"
        stroke="#a3a3a3"
      />
      <rect
        x={LEFT}
        y={y + 4}
        width={(b / of) * tw}
        height={8}
        fill={TERM_COLOUR.mem}
      />
    </g>
  );

  const visual = (
    <div className="mx-auto max-w-lg">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full"
        role="img"
        aria-label={`One decoder layer, ${N_OPS} operations, kernel by kernel and fused; operation ${k + 1}, ${s.name}, is running.`}
      >
        <text
          x={LEFT}
          y={L1.box - 4}
          style={{ fontSize: fs(10.5) }}
          className="fill-neutral-800 font-medium dark:fill-neutral-200"
        >
          Kernel by kernel: every result goes back to HBM
        </text>
        {laneDraw(L1, "u")}
        <line
          x1={LEFT}
          x2={W - LEFT}
          y1={L1.bar + BAR}
          y2={L1.bar + BAR}
          stroke={TERM_COLOUR.mem}
        />
        <text
          x={LEFT}
          y={L2.box - 4}
          style={{ fontSize: fs(10.5) }}
          className="fill-neutral-800 font-medium dark:fill-neutral-200"
        >
          Fused dataflow: results stay on chip
        </text>
        {laneDraw(L2, "f")}
        <line
          x1={LEFT}
          x2={W - LEFT}
          y1={L2.bar + BAR}
          y2={L2.bar + BAR}
          stroke={TERM_COLOUR.mem}
        />
        <g data-total="u">
          {total(totY, "Kernel by kernel", s.u_bytes, s.u_t, u.bytes)}
        </g>
        <g data-total="f">
          {total(totY + fs(12) + 14, "Fused", s.f_bytes, s.f_t, u.bytes)}
        </g>
      </svg>
      <p className="mt-1 text-[0.7rem] text-neutral-600 dark:text-neutral-400">
        Boxes are the layer&apos;s operations in order (yellow: matrix
        multiplies; blue: running now). Purple bars under each are the bytes it
        moves to and from HBM, on the same scale in both lanes; the green links
        in the lower lane are on-chip buffers between pipeline stages. The two
        bars at the bottom add up the HBM traffic so far.
      </p>
    </div>
  );

  const stats = (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
      <Stat
        label="Kernel by kernel"
        value={fmtBytes(u.bytes)}
        hint={fmtTime(u.t)}
      />
      <Stat label="Fused" value={fmtBytes(f.bytes)} hint={fmtTime(f.t)} />
      <Stat label="HBM traffic saved" value={pct(1 - f.bytes / u.bytes)} />
      <Stat
        label="Time saved"
        value={pct(1 - f.t / u.t)}
        hint="this layer, one socket"
      />
    </div>
  );

  return (
    <AnimationPanel
      testId="dataflow-widget"
      title="One decoder layer, kernel by kernel or fused"
      summary={`The same layer on one ${partName} (BF16), as separate kernels and as one dataflow pipeline. Bytes and times are the model's bound from the paper's HBM bandwidth and peak rate.`}
      stepper={st}
      stepLabel="operation"
      caption={dataflowCaption(s, steps.length, partName)}
      visual={visual}
      stats={stats}
      equation={children}
      hl={hover ?? (s.kind === "matmul" ? "mem comp" : "act")}
      onEquationHover={setHover}
      params={
        <div className="flex flex-wrap gap-4">
          <Segmented
            label="Work"
            value={ck}
            options={Object.entries(CASES).map(([v, c]) => ({
              value: v,
              label: c[4],
            }))}
            onChange={setCk}
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
