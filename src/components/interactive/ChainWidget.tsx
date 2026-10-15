"use client";

/**
 * Chapter 5's hero: one token down a chain of compiler-scheduled chips. The
 * model is cut at layer boundaries into stages; each stage reads its share of
 * the weights from SRAM, does its layers' all-reduces inside its group of
 * chips, and hands the activation to the next stage over one chip-to-chip
 * hop, all on a schedule fixed by the compiler. Every frame is a state of
 * `chainSteps` (checked against the Python reference); the time bar
 * underneath splits the token's time into SRAM reads, all-reduces and hops.
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Hatch } from "@/components/interactive/Hatch";
import { Segmented, Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { fmtInt, fmtTime, trim } from "@/lib/format";
import { chainCaption } from "@/lib/accel/captions";
import {
  CHAIN_CASES,
  CHAIN_CTX,
  chainSteps,
  decode,
  fval,
  part,
} from "@/lib/accel/model";
import { MUTED, STATE_COLOUR, TERM_COLOUR } from "@/lib/viz/palette";

const W = 360;
const LABELS: Record<string, string> = {
  demo: "576 chips, nodes of 8",
  "fewest-nodes": "304 chips, nodes of 8",
  "fewest-chain": "fewest chips, one per stage",
  groq3: "Groq 3 LPU (announced)",
};

export default function ChainWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [key, setKey] = useState<string>("demo");
  const [hover, setHover] = useState<string | null>(null);
  const c = CHAIN_CASES.find((x) => x.key === key)!;
  const font = useSvgFont(W);
  const fs = font.fs;
  const d = useMemo(
    () => decode(c.part, c.model, c.fmt, 1, CHAIN_CTX, c.chips, c.group),
    [c],
  );
  const steps = useMemo(
    () => chainSteps(c.part, c.model, c.fmt, CHAIN_CTX, c.chips, c.group),
    [c],
  );
  const st = useStepper(steps.length, {
    stepMs: Math.max(60, Math.round(9000 / steps.length)),
    resetKey: key,
  });
  const s = steps[st.step]!;
  const hop = fval(c.part, "hop_s");
  const name = (part(c.part).name as string).split(" (")[0]!;
  const S = steps.length;
  const perRow = S > 160 ? 30 : S > 60 ? 24 : 12;
  const rows = Math.ceil(S / perRow);
  const cw = (W - 12) / perRow;
  const ch = Math.min(cw, 22);
  const gridH = rows * ch + 4;
  const H = gridH + 70;
  const total = steps[S - 1]!.done;
  const work = s.work_end - s.start;
  const reduce = s.done - s.work_end;
  // the token's time so far, split into the three kinds
  const sumWork = (s.stage + 1) * work;
  const sumReduce = (s.stage + 1) * reduce;
  const sumHop = s.stage * (hop ?? 0);
  const bx = (t: number) => (t / total) * (W - 12);

  const visual = (
    <div className="mx-auto max-w-lg">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full"
        role="img"
        aria-label={`A chain of ${S} stages; the token is at stage ${s.stage + 1}, ${fmtTime(s.done)} into its ${fmtTime(total)}.`}
      >
        <defs>
          <Hatch id="chain-hatch" />
        </defs>
        {steps.map((x) => {
          const done = x.stage < s.stage;
          const cur = x.stage === s.stage;
          return (
            <rect
              key={x.stage}
              data-stage={x.stage}
              data-state={cur ? "active" : done ? "done" : "waiting"}
              x={6 + (x.stage % perRow) * cw + 0.6}
              y={2 + Math.floor(x.stage / perRow) * ch + 0.6}
              width={cw - 1.2}
              height={ch - 1.2}
              rx={1}
              fill={
                cur
                  ? STATE_COLOUR.active
                  : done
                    ? TERM_COLOUR.sram
                    : MUTED.light
              }
              opacity={cur ? 1 : done ? 0.75 : 0.35}
            />
          );
        })}
        <text
          x={6}
          y={gridH + fs(14)}
          style={{ fontSize: fs(11) }}
          className="fill-neutral-800 font-medium dark:fill-neutral-200"
        >
          the token&apos;s time so far: {fmtTime(s.done)} of {fmtTime(total)}
        </text>
        {/* the time bar: reads, all-reduces, hops */}
        <rect
          x={6}
          y={gridH + 26}
          width={W - 12}
          height={16}
          fill="none"
          stroke="#a3a3a3"
        />
        <rect
          x={6}
          y={gridH + 26}
          width={bx(sumWork)}
          height={16}
          fill={TERM_COLOUR.sram}
          data-term="sram"
        />
        <rect
          x={6 + bx(sumWork)}
          y={gridH + 26}
          width={bx(sumReduce)}
          height={16}
          fill={TERM_COLOUR.comp}
          data-term="reduce"
        />
        <rect
          x={6 + bx(sumWork + sumReduce)}
          y={gridH + 26}
          width={bx(sumHop)}
          height={16}
          fill="url(#chain-hatch)"
          data-term="hop"
        />
        <text
          x={6}
          y={gridH + 60}
          style={{ fontSize: fs(10) }}
          className="fill-neutral-700 dark:fill-neutral-300"
        >
          green SRAM reads · yellow all-reduces · hatched hops
        </text>
      </svg>
      <p className="mt-1 text-[0.7rem] text-neutral-600 dark:text-neutral-400">
        One square per pipeline stage, in order: blue is the stage the token is
        in, green the stages it has passed. The bar adds up where its time has
        gone. On a static schedule every stage takes exactly this long, every
        time.
      </p>
    </div>
  );

  const stats = (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
      <Stat
        label="Chips"
        value={fmtInt(d.chips)}
        hint={`${d.stages} stages of ${d.group}`}
      />
      <Stat
        label="Per stage"
        value={fmtTime(work + reduce)}
        hint="reads + all-reduces"
      />
      <Stat
        label="Per hop"
        value={hop === null ? "not disclosed" : fmtTime(hop)}
      />
      <Stat
        label="Bound, per user"
        value={`${trim(d.tok_s_user)} tokens/s`}
        hint={`${d.bound}-bound`}
      />
    </div>
  );

  return (
    <AnimationPanel
      testId="chain-widget"
      title="One token down a chain of SRAM-only chips"
      summary="The model cut into pipeline stages at layer boundaries, its weights spread over the chips' SRAM; the compiler fixes when each stage reads, reduces and hands on. Times are the model's bound (vendor bandwidths and hop latency)."
      stepper={st}
      stepLabel="stage"
      caption={chainCaption(s, d, name, hop)}
      visual={visual}
      stats={stats}
      equation={children}
      hl={hover ?? "sram stage"}
      onEquationHover={setHover}
      params={
        <Segmented
          label="Chain"
          value={key}
          options={CHAIN_CASES.map((x) => ({
            value: x.key,
            label: LABELS[x.key] ?? x.key,
          }))}
          onChange={setKey}
        />
      }
    />
  );
}
