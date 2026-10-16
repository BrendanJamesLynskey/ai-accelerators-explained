"use client";

/**
 * Chapter 8's hero: a decode step on a small memory budget, at narrower and
 * narrower weights. Each row is one weight width; its bar is the tokens per
 * second one user gets when every weight and the KV cache are read once per
 * token over the device's memory bandwidth. The strip underneath is the
 * weights against the device's memory, where the capacity is disclosed.
 * Every frame is a state of `edgeSteps` (checked against the Python
 * reference).
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Hatch } from "@/components/interactive/Hatch";
import { Segmented, Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { fmtBytes, fmtRate, fmtTime, trim } from "@/lib/format";
import { edgeCaption } from "@/lib/accel/captions";
import {
  anyName,
  anyval,
  edgeSteps,
  shape,
  type ModelId,
} from "@/lib/accel/model";
import { MUTED, STATE_COLOUR, TERM_COLOUR } from "@/lib/viz/palette";

const W = 360;
const DEVICES: Record<string, string> = {
  m4: "Apple M4 Neural Engine",
  v80: "AMD Alveo V80",
  v70: "AMD Alveo V70",
};

export default function EdgeWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [dev, setDev] = useState<string>("m4");
  const [mid, setMid] = useState<ModelId>("llama3-8b");
  const [hover, setHover] = useState<string | null>(null);
  const font = useSvgFont(W);
  const fs = font.fs;
  const steps = useMemo(() => edgeSteps(dev, mid, 1024), [dev, mid]);
  const st = useStepper(steps.length, {
    stepMs: 1300,
    resetKey: `${dev}-${mid}`,
  });
  const k = st.step;
  const s = steps[k]!;
  const name = anyName(dev).split(" (")[0]!;
  const cap = anyval(dev, "off_bytes");
  const bw = anyval(dev, "off_bw") as number;
  let max = 0;
  for (const x of steps) if (x.tok_s > max) max = x.tok_s;
  const LABEL = fs(42);
  const ROWH = Math.max(18, fs(15));
  const TOP = 6;
  const bw0 = W - LABEL - 8 - fs(48);
  const stripY = TOP + steps.length * ROWH + fs(12) + 12;
  const H = stripY + 18 + fs(11) + 8;
  const sx = (v: number) => (v / max) * bw0;
  const big = cap === null ? s.weights : Math.max(cap, s.weights);
  const tw = W - 12;

  const visual = (
    <div className="mx-auto max-w-lg">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full"
        role="img"
        aria-label={`Tokens per second on ${name} at ${steps.length} weight widths; ${s.bits}-bit highlighted.`}
      >
        <defs>
          <Hatch id="edge-hatch" />
        </defs>
        {steps.map((x, i) => {
          const y = TOP + i * ROWH;
          const state = i === k ? "active" : i < k ? "done" : "waiting";
          return (
            <g key={x.bits} data-bits={x.bits} data-state={state}>
              <text
                x={LABEL}
                y={y + ROWH * 0.68}
                textAnchor="end"
                style={{ fontSize: fs(10) }}
                className={
                  state === "active"
                    ? "fill-neutral-950 font-semibold dark:fill-white"
                    : "fill-neutral-600 dark:fill-neutral-400"
                }
              >
                {x.bits}-bit
              </text>
              <rect
                x={LABEL + 6}
                y={y + 2}
                width={Math.max(1, sx(x.tok_s))}
                height={ROWH - 5}
                rx={1.5}
                fill={
                  state === "active" ? STATE_COLOUR.active : TERM_COLOUR.mem
                }
                opacity={
                  state === "waiting" ? 0.25 : state === "done" ? 0.6 : 1
                }
              />
              {i <= k && (
                <text
                  x={LABEL + 10 + sx(x.tok_s)}
                  y={y + ROWH * 0.68}
                  style={{ fontSize: fs(9.5) }}
                  className="fill-neutral-800 dark:fill-neutral-200"
                >
                  {trim(x.tok_s)}/s
                </text>
              )}
            </g>
          );
        })}
        <text
          x={6}
          y={stripY - 5}
          style={{ fontSize: fs(10) }}
          className="fill-neutral-800 dark:fill-neutral-200"
        >
          weights {fmtBytes(s.weights)}
          {cap === null
            ? " (memory size not disclosed)"
            : ` of ${fmtBytes(cap)} memory`}
        </text>
        <rect
          x={6}
          y={stripY}
          width={tw}
          height={14}
          fill={cap === null ? "url(#edge-hatch)" : "none"}
          stroke="#a3a3a3"
        />
        <rect
          x={6}
          y={stripY}
          width={(s.weights / big) * tw}
          height={14}
          fill={s.fits === false ? STATE_COLOUR.stalled : TERM_COLOUR.mem}
          data-fits={String(s.fits)}
        />
        {cap !== null && cap < big && (
          <line
            x1={6 + (cap / big) * tw}
            x2={6 + (cap / big) * tw}
            y1={stripY - 2}
            y2={stripY + 16}
            stroke={MUTED.dark}
            strokeWidth={2}
          />
        )}
        <text
          x={6}
          y={stripY + 18 + fs(11)}
          style={{ fontSize: fs(9.5) }}
          className="fill-neutral-700 dark:fill-neutral-300"
        >
          tokens/s for one user at {fmtRate(bw)}
        </text>
      </svg>
      <p className="mt-1 text-[0.7rem] text-neutral-600 dark:text-neutral-400">
        One bar per weight width: the bound on one user&apos;s tokens per
        second, reading every weight and the KV cache once per token. The strip
        is the model&apos;s weights against the device&apos;s memory (red when
        they do not fit; hatched where the capacity is not disclosed).
      </p>
    </div>
  );

  const stats = (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
      <Stat label="Tokens/s, one user" value={trim(s.tok_s)} />
      <Stat
        label="Per token"
        value={fmtTime(s.t)}
        hint={`${s.bits}-bit weights`}
      />
      <Stat label="Read per token" value={fmtBytes(s.bytes)} />
      <Stat
        label="Energy"
        value={
          s.j_per_token === null ? "not disclosed" : `${trim(s.j_per_token)} J`
        }
        hint={s.j_per_token === null ? "power not given" : "per token, at TDP"}
      />
    </div>
  );

  return (
    <AnimationPanel
      testId="edge-widget"
      title="A decode step on a small memory budget"
      summary={`${shape(mid).name} at batch 1, every weight read once per token: the narrower the weights, the fewer bytes and the faster the token. Bounds from the device's stated memory bandwidth.`}
      stepper={st}
      stepLabel="width"
      caption={edgeCaption(s, steps.length, name)}
      visual={visual}
      stats={stats}
      equation={children}
      hl={hover ?? "bits mem"}
      onEquationHover={setHover}
      params={
        <div className="flex flex-wrap gap-4">
          <Segmented
            label="Device"
            value={dev}
            options={Object.entries(DEVICES).map(([v, l]) => ({
              value: v,
              label: l,
            }))}
            onChange={setDev}
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
