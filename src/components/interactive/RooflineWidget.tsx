"use client";

/**
 * Chapter 2's hero: one decode step on GPUs as the batch grows. Left, the
 * roofline (log-log): the step's arithmetic intensity against the flop rate
 * it reaches, under the slanted memory roof and the flat compute roof; the
 * point climbs the slope as the batch grows, until it meets the ridge. Right,
 * the step's two terms as bars: bytes over bandwidth and flops over the peak.
 * Every frame is a state of `batchSweep` (checked against the Python
 * reference).
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Hatch } from "@/components/interactive/Hatch";
import { Segmented, Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { fmtInt, fmtTime, trim } from "@/lib/format";
import { sweepCaption } from "@/lib/accel/captions";
import {
  batchSweep,
  part,
  peak,
  ridge,
  type Fmt,
  type ModelId,
  type PartId,
} from "@/lib/accel/model";
import { STATE_COLOUR, TERM_COLOUR } from "@/lib/viz/palette";

const W = 360;
const H = 230;
const PL = 44;
const PR = 10;
const PT = 20;
const PB = 34;
const XMIN = 0; // log10 intensity
const XMAX = 4;

const CASES: Record<
  string,
  { model: ModelId; fmt: Fmt; chips: number; label: string }
> = {
  small: {
    model: "llama3-8b",
    fmt: "bf16",
    chips: 1,
    label: "8B, BF16, 1 GPU",
  },
  big: { model: "llama3-70b", fmt: "fp8", chips: 8, label: "70B, FP8, 8 GPUs" },
};

export default function RooflineWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [pid, setPid] = useState<PartId>("h100");
  const [ck, setCase] = useState<string>("big");
  const [ctx, setCtx] = useState<string>("1024");
  const [hover, setHover] = useState<string | null>(null);
  const c = CASES[ck]!;
  const font = useSvgFont(W);
  const fs = font.fs;
  const sweep = useMemo(
    () => batchSweep(pid, c.model, c.fmt, Number(ctx), c.chips),
    [pid, c, ctx],
  );
  const st = useStepper(sweep.length, {
    stepMs: 1100,
    resetKey: `${pid}|${ck}|${ctx}`,
  });
  const s = sweep[st.step]!;
  const pk = peak(pid, c.fmt).flops as number;
  const rg = ridge(pid, c.fmt) as number;
  const bw = part(pid).off_bw as number;
  const g = s.group;
  const name = part(pid).name as string;
  const short = name.replace("NVIDIA ", "").replace(" (HGX B200)", "");

  // the chart's scales: x = log10 intensity, y = log10 flop/s for g chips
  const yTop = Math.ceil(Math.log10(g * pk) + 0.2);
  const yBot = yTop - 4;
  const px = (lx: number) => PL + ((lx - XMIN) / (XMAX - XMIN)) * (W - PL - PR);
  const py = (ly: number) => PT + ((yTop - ly) / (yTop - yBot)) * (H - PT - PB);
  const roof = (lx: number) =>
    Math.min(Math.log10(g * pk), Math.log10(g * bw) + lx);
  const pts = Array.from({ length: 41 }, (_, i) => XMIN + (i / 40) * XMAX);
  const roofPath = pts
    .map((lx, i) => `${i ? "L" : "M"}${px(lx)},${py(roof(lx))}`)
    .join(" ");
  const ach = s.flops / s.t_step;
  const lx = Math.log10(s.intensity);
  const memBound = s.bound === "memory";

  const tMax = Math.max(...sweep.map((x) => Math.max(x.t_mem, x.t_comp ?? 0)));
  const barW = 150;

  const visual = (
    <div className="grid min-w-0 gap-3 md:grid-cols-[minmax(0,1fr)_12rem]">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full"
        role="img"
        aria-label={`Roofline of ${g} ${short}: batch ${s.batch} at ${trim(s.intensity)} flop per byte, ${s.bound}-bound.`}
      >
        {/* axes */}
        <line x1={PL} x2={W - PR} y1={H - PB} y2={H - PB} stroke="#a3a3a3" />
        <line x1={PL} x2={PL} y1={PT} y2={H - PB} stroke="#a3a3a3" />
        {[0, 1, 2, 3, 4].map((e) => (
          <text
            key={e}
            x={px(e)}
            y={H - PB + fs(13)}
            textAnchor="middle"
            style={{ fontSize: fs(10) }}
            className="fill-neutral-600 dark:fill-neutral-400"
          >
            {e === 0 ? "1" : `10${["", "", "²", "³", "⁴"][e]}`}
          </text>
        ))}
        <text
          x={PL + 6}
          y={PT - 6}
          style={{ fontSize: fs(10) }}
          className="fill-neutral-700 dark:fill-neutral-300"
        >
          flop/s reached
        </text>
        <text
          x={(PL + W - PR) / 2}
          y={H - 4}
          textAnchor="middle"
          style={{ fontSize: fs(10) }}
          className="fill-neutral-700 dark:fill-neutral-300"
        >
          flop per byte read
        </text>
        {[yBot, yBot + 2, yTop].map((e) => (
          <text
            key={e}
            x={PL - 4}
            y={py(e) + 4}
            textAnchor="end"
            style={{ fontSize: fs(10) }}
            className="fill-neutral-600 dark:fill-neutral-400"
          >
            {e >= 15 ? `${10 ** (e - 15)}P` : `${10 ** (e - 12)}T`}
          </text>
        ))}
        {/* the roofs */}
        <path d={roofPath} fill="none" stroke="#525252" strokeWidth={2} />
        <line
          x1={px(Math.log10(rg))}
          x2={px(Math.log10(rg))}
          y1={py(Math.log10(g * pk))}
          y2={H - PB}
          stroke={TERM_COLOUR.comp}
          strokeDasharray="3 3"
        />
        <text
          x={px(Math.log10(rg)) + 3}
          y={H - PB - 5}
          style={{ fontSize: fs(10) }}
          className="fill-neutral-700 dark:fill-neutral-300"
        >
          ridge {trim(rg)}
        </text>
        {/* the batches so far */}
        {sweep.slice(0, st.step + 1).map((x) => (
          <circle
            key={x.batch}
            cx={px(Math.log10(x.intensity))}
            cy={py(Math.log10(x.flops / x.t_step))}
            r={x.i === st.step ? 6 : 3}
            fill={
              x.i === st.step
                ? STATE_COLOUR.active
                : x.bound === "memory"
                  ? TERM_COLOUR.mem
                  : TERM_COLOUR.comp
            }
            stroke={x.fits ? "none" : STATE_COLOUR.stalled}
            strokeWidth={2}
            data-batch={x.batch}
          />
        ))}
        <text
          x={Math.min(px(lx) + 8, W - PR - 70)}
          y={py(Math.log10(ach)) - 8}
          style={{ fontSize: fs(11) }}
          className="fill-neutral-900 font-medium dark:fill-neutral-100"
        >
          batch {fmtInt(s.batch)}
        </text>
      </svg>
      <div className="min-w-0 text-xs">
        <p className="font-medium text-neutral-800 dark:text-neutral-200">
          One step, batch {fmtInt(s.batch)}
        </p>
        <svg
          viewBox={`0 0 ${barW + 10} 70`}
          className="mt-1 h-auto w-full max-w-56"
          role="img"
          aria-label={`Memory term ${fmtTime(s.t_mem)}, compute term ${fmtTime(s.t_comp ?? 0)}.`}
        >
          <defs>
            <Hatch id="roof-hatch" />
          </defs>
          <rect
            x={0}
            y={6}
            width={Math.max(1, (s.t_mem / tMax) * barW)}
            height={18}
            fill={TERM_COLOUR.mem}
            opacity={memBound ? 1 : 0.45}
          />
          <rect
            x={0}
            y={40}
            width={Math.max(1, ((s.t_comp ?? 0) / tMax) * barW)}
            height={18}
            fill={TERM_COLOUR.comp}
            opacity={memBound ? 0.45 : 1}
          />
          {!s.fits && (
            <rect
              x={0}
              y={0}
              width={barW + 10}
              height={66}
              fill="url(#roof-hatch)"
              opacity={0.5}
            />
          )}
        </svg>
        <p className="mt-1 text-neutral-600 dark:text-neutral-400">
          <span style={{ color: TERM_COLOUR.mem }}>■</span> bytes ÷ bandwidth:{" "}
          {fmtTime(s.t_mem)}
          <br />
          <span style={{ color: TERM_COLOUR.comp }}>■</span> flops ÷ peak:{" "}
          {fmtTime(s.t_comp ?? 0)}
          {!s.fits && (
            <>
              <br />
              Hatched: the KV caches would not fit.
            </>
          )}
        </p>
      </div>
    </div>
  );

  const stats = (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
      <Stat label="Step" value={fmtTime(s.t_step)} hint={`${s.bound}-bound`} />
      <Stat label="Tokens/s, all users" value={trim(s.tok_s)} />
      <Stat label="Tokens/s, per user" value={trim(s.tok_s_user)} />
      <Stat
        label="Energy per token"
        value={
          s.j_per_token === null ? "not disclosed" : `${trim(s.j_per_token)} J`
        }
        hint={pid === "b200" ? "system share of DGX power" : "at TDP"}
      />
    </div>
  );

  return (
    <AnimationPanel
      testId="roofline-widget"
      title="One decode step on GPUs, batch by batch"
      summary="The step reads every weight once and each sequence's KV cache, and does two flops per weight per sequence. Bandwidth and peak rate are the vendor's figures."
      stepper={st}
      stepLabel="batch"
      caption={sweepCaption(s, sweep.length, short, rg)}
      visual={visual}
      stats={stats}
      equation={children}
      hl={hover ?? (memBound ? "mem kv" : "comp")}
      onEquationHover={setHover}
      params={
        <>
          <Segmented
            label="GPU"
            value={pid}
            options={[
              { value: "h100", label: "H100" },
              { value: "b200", label: "B200" },
            ]}
            onChange={(v) => setPid(v as PartId)}
          />
          <Segmented
            label="Model, format, GPUs"
            value={ck}
            options={Object.entries(CASES).map(([k, v]) => ({
              value: k,
              label: v.label,
            }))}
            onChange={setCase}
          />
          <Segmented
            label="Context per sequence"
            value={ctx}
            options={[
              { value: "1024", label: "1k tokens" },
              { value: "8192", label: "8k tokens" },
            ]}
            onChange={setCtx}
          />
        </>
      }
    />
  );
}
