"use client";

/**
 * Chapter 4's hero: defects landing on a wafer-scale chip. The wafer's 84
 * dies are drawn as a 12 x 7 grid (the vendor gives the count, not the
 * layout); defects arrive a few at a time, as many as the defect density
 * says to expect, each killing one core that a spare replaces. Beside it,
 * what the same density does to a single reticle-sized die and to a wafer
 * that had to be perfect. Every frame is a state of `defectSteps` and the
 * statistics are `waferYield` (both checked against the Python reference).
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Segmented, Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { fmtInt, pct, trim } from "@/lib/format";
import { waferCaption } from "@/lib/accel/captions";
import {
  DIE_COLS,
  DIE_ROWS,
  defectSteps,
  defects,
  fval,
  illustrative,
  waferYield,
} from "@/lib/accel/model";
import { LEVEL_COLOUR, STATE_COLOUR } from "@/lib/viz/palette";

const W = 360;
const PAD = 6;
const DIE = (W - 2 * PAD) / DIE_COLS;
const H = PAD * 2 + DIE * DIE_ROWS;
const SEED = 7;

export default function WaferWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [d0s, setD0] = useState<string>("0.1");
  const [hover, setHover] = useState<string | null>(null);
  const d0 = Number(d0s);
  const font = useSvgFont(W);
  const y = useMemo(
    () =>
      waferYield(
        "wse-3",
        d0,
        illustrative("gpu_die_mm2"),
        illustrative("spare_frac"),
      ),
    [d0],
  );
  const count = Math.round(y.lam_wafer);
  const per = Math.ceil(count / 23);
  const ds = useMemo(() => defects(SEED, count), [count]);
  const steps = useMemo(() => defectSteps(SEED, count, per), [count, per]);
  const st = useStepper(steps.length, { stepMs: 700, resetKey: d0s });
  const s = steps[st.step]!;
  const cores = fval("wse-3", "units") as number;
  const dies = fval("wse-3", "dies") as number;
  const shownIds = new Set(ds.slice(0, s.shown).map((d) => d.i));
  const latest = ds.slice(Math.max(0, s.shown - per), s.shown);
  const hitDies = new Set(ds.slice(0, s.shown).map((d) => `${d.col},${d.row}`));

  const visual = (
    <div className="mx-auto max-w-lg">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full"
        role="img"
        aria-label={`A wafer of ${dies} dies with ${s.shown} of ${count} defects placed, in ${s.dies_hit} dies.`}
      >
        {Array.from({ length: DIE_ROWS * DIE_COLS }, (_, i) => {
          const c = i % DIE_COLS;
          const r = Math.floor(i / DIE_COLS);
          const hit = hitDies.has(`${c},${r}`);
          return (
            <rect
              key={i}
              data-die={i}
              data-hit={hit}
              x={PAD + c * DIE + 0.8}
              y={PAD + r * DIE + 0.8}
              width={DIE - 1.6}
              height={DIE - 1.6}
              rx={1.5}
              fill={LEVEL_COLOUR.smem}
              opacity={hit ? 0.55 : 0.3}
            />
          );
        })}
        {ds
          .filter((d) => shownIds.has(d.i))
          .map((d) => {
            const x = PAD + d.col * DIE + 2 + ((d.u + 0.5) / 10) * (DIE - 4);
            const yy = PAD + d.row * DIE + 2 + ((d.v + 0.5) / 10) * (DIE - 4);
            const isNew = latest.some((l) => l.i === d.i);
            const r = isNew ? 3.2 : 2.2;
            return (
              <g key={d.i} data-defect={d.i}>
                <line
                  x1={x - r}
                  y1={yy - r}
                  x2={x + r}
                  y2={yy + r}
                  stroke={STATE_COLOUR.stalled}
                  strokeWidth={isNew ? 2 : 1.5}
                />
                <line
                  x1={x - r}
                  y1={yy + r}
                  x2={x + r}
                  y2={yy - r}
                  stroke={STATE_COLOUR.stalled}
                  strokeWidth={isNew ? 2 : 1.5}
                />
              </g>
            );
          })}
      </svg>
      <p className="mt-1 text-[0.7rem] text-neutral-600 dark:text-neutral-400">
        Green squares: the wafer&apos;s {dies} dies (about {fmtInt(y.die_cores)}{" "}
        cores each), darker once hit. A cross is a defect: one dead core, routed
        round to a spare. The newest defects are drawn larger.
      </p>
    </div>
  );

  const stats = (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
      <Stat
        label="Defects expected"
        value={trim(y.lam_wafer)}
        hint={`at ${d0} per cm² (illustrative)`}
        plain
      />
      <Stat
        label="Reticle-sized die works"
        value={pct(y.die_yield)}
        hint={`${illustrative("gpu_die_mm2")} mm², no spares`}
        plain
      />
      <Stat
        label="Wafer with no defect"
        value={
          y.wafer_perfect < 1e-6
            ? `${y.wafer_perfect.toExponential(1)}`
            : pct(y.wafer_perfect)
        }
        hint="what a monolithic wafer would need"
        plain
      />
      <Stat
        label="Spares per defect"
        value={trim(y.spare_margin)}
        hint={`${fmtInt(y.spares)} spare cores (illustrative share)`}
        plain
      />
    </div>
  );

  return (
    <AnimationPanel
      testId="wafer-widget"
      title="Defects on a wafer-scale chip"
      summary="The defects a wafer of this area should expect, placed by a seeded generator. Each takes out one small core, and the fabric routes round it, so the chip survives defects instead of having to avoid them."
      stepper={st}
      stepLabel="step"
      caption={waferCaption(s, count, dies, cores, y.spares)}
      visual={visual}
      stats={stats}
      equation={children}
      hl={hover ?? (s.shown === 0 ? "lam" : "lam die")}
      onEquationHover={setHover}
      countFrom={0}
      params={
        <Segmented
          label="Defect density (illustrative)"
          value={d0s}
          options={[
            { value: "0.05", label: "0.05 /cm²" },
            { value: "0.1", label: "0.1 /cm²" },
            { value: "0.2", label: "0.2 /cm²" },
          ]}
          onChange={setD0}
        />
      }
    />
  );
}
