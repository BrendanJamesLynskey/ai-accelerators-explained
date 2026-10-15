"use client";

/**
 * Chapter 7's hero: one decoder layer on a mesh of small cores. The card is
 * drawn as its 120 Tensix cores (a 12 x 10 grid; the vendor gives the count,
 * not this layout), with GDDR6 on either side and the four Ethernet ports
 * on top. On each matrix step every core receives its share of that
 * matrix's weights from GDDR6 (the fill shows how much of the core's SRAM
 * the share takes); on the two all-reduce steps the cores wait and the
 * cards exchange partial sums over Ethernet. The bar underneath adds up the
 * layer's time: DRAM streaming and Ethernet. Every frame is a state of
 * `meshSteps` (checked against the Python reference).
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { Hatch } from "@/components/interactive/Hatch";
import { Segmented, Stat } from "@/components/ui/Controls";
import { useSvgFont } from "@/components/viz/useSvgFont";
import { fmtBytes, fmtInt, fmtTime, pct, trim } from "@/lib/format";
import { meshCaption } from "@/lib/accel/captions";
import {
  meshLayer,
  meshSteps,
  placement,
  type ModelId,
} from "@/lib/accel/model";
import { MUTED, STATE_COLOUR, TERM_COLOUR } from "@/lib/viz/palette";

const W = 360;
const COLS = 12;
const ROWS = 10;
const CARDS = [1, 2, 3, 4, 8] as const;

export default function MeshWidget({
  children,
}: {
  children?: ReactNode;
}): JSX.Element {
  const [mid, setMid] = useState<ModelId>("llama3-70b");
  const [cards, setCards] = useState<number>(3);
  const [batch, setBatch] = useState<number>(1);
  const [hover, setHover] = useState<string | null>(null);
  const font = useSvgFont(W);
  const fs = font.fs;
  const m = useMemo(
    () => meshLayer("blackhole", mid, "fp8", batch, 1024, cards),
    [mid, batch, cards],
  );
  const steps = useMemo(
    () => meshSteps("blackhole", mid, "fp8", batch, 1024, cards),
    [mid, batch, cards],
  );
  const fewest = placement("blackhole", mid, "fp8", batch, 1024).chips;
  const st = useStepper(steps.length, {
    stepMs: 1300,
    resetKey: `${mid}-${cards}-${batch}`,
  });
  const k = st.step;
  const s = steps[k]!;
  const eth = s.kind === "ethernet";

  const SIDE = 22;
  const TOP = fs(12) + 22;
  const gw = W - 2 * SIDE - 16 - (cards > 1 ? 56 : 0);
  const cw = gw / COLS;
  const ch = Math.min(cw, 18);
  const gx = SIDE + 8;
  const gh = ROWS * ch;
  const barY = TOP + gh + fs(12) + 18;
  const H = barY + 16 + fs(11) + 10;
  const fill = Math.min(1, s.per_core / m.core_sram);
  const tw = W - 12;
  // time bar: DRAM so far, Ethernet so far
  let dram = 0;
  let net = 0;
  for (const x of steps.slice(0, k + 1)) {
    if (x.kind === "ethernet") net += x.t_op;
    else dram += x.t_op;
  }
  const bx = (t: number) => (t / m.t) * tw;

  const cores = [];
  for (let r = 0; r < ROWS; r++)
    for (let c = 0; c < COLS; c++) {
      const x = gx + c * cw + 0.8;
      const y = TOP + r * ch + 0.8;
      const w = cw - 1.6;
      const h = ch - 1.6;
      cores.push(
        <g key={`${r}-${c}`}>
          <rect
            x={x}
            y={y}
            width={w}
            height={h}
            rx={1}
            fill="none"
            stroke={eth ? MUTED.light : STATE_COLOUR.active}
            strokeWidth={0.8}
          />
          {!eth && (
            <rect
              x={x}
              y={y + h * (1 - fill)}
              width={w}
              height={h * fill}
              fill={TERM_COLOUR.sram}
              opacity={0.85}
            />
          )}
        </g>,
      );
    }

  const visual = (
    <div className="mx-auto max-w-lg">
      <svg
        ref={font.ref}
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full"
        role="img"
        aria-label={`One Blackhole card of ${COLS * ROWS} cores${cards > 1 ? `, one of ${cards}` : ""}; step ${k + 1}, ${s.name}.`}
      >
        <defs>
          <Hatch id="mesh-hatch" />
        </defs>
        <text
          x={6}
          y={fs(11)}
          style={{ fontSize: fs(10.5) }}
          className="fill-neutral-800 font-medium dark:fill-neutral-200"
        >
          {s.name}: {eth ? "Ethernet between cards" : "GDDR6 into every core"}
        </text>
        {/* Ethernet ports on top */}
        {[0, 1, 2, 3].map((p) => (
          <rect
            key={p}
            data-port={p}
            x={gx + (p + 0.5) * (gw / 4) - 9}
            y={TOP - 12}
            width={18}
            height={8}
            rx={1}
            fill={eth && cards > 1 ? TERM_COLOUR.hop : MUTED.light}
            opacity={eth && cards > 1 ? 1 : 0.5}
          />
        ))}
        {/* GDDR6 either side */}
        {[6, gx + gw + 4].map((x) => (
          <rect
            key={x}
            data-dram={x}
            x={x}
            y={TOP}
            width={14}
            height={gh}
            rx={2}
            fill={TERM_COLOUR.mem}
            opacity={eth ? 0.3 : 1}
          />
        ))}
        <g data-cores={COLS * ROWS} data-fill={trim(fill)}>
          {cores}
        </g>
        {/* the other cards, if any */}
        {cards > 1 && (
          <g data-cards={cards}>
            {Array.from({ length: Math.min(cards - 1, 7) }, (_, i) => (
              <rect
                key={i}
                x={W - 46 + (i % 2) * 20}
                y={TOP + Math.floor(i / 2) * 22}
                width={16}
                height={18}
                rx={2}
                fill="none"
                stroke={eth ? TERM_COLOUR.hop : MUTED.light}
                strokeWidth={1.5}
              />
            ))}
            <text
              x={W - 6}
              y={TOP + gh}
              textAnchor="end"
              style={{ fontSize: fs(9.5) }}
              className="fill-neutral-700 dark:fill-neutral-300"
            >
              +{cards - 1} card{cards > 2 ? "s" : ""}
            </text>
          </g>
        )}
        <text
          x={6}
          y={barY - 6}
          style={{ fontSize: fs(10) }}
          className="fill-neutral-800 dark:fill-neutral-200"
        >
          the layer so far: {fmtTime(s.t)} of {fmtTime(m.t)}
        </text>
        <rect
          x={6}
          y={barY}
          width={tw}
          height={14}
          fill="none"
          stroke="#a3a3a3"
        />
        <rect
          x={6}
          y={barY}
          width={bx(dram)}
          height={14}
          fill={TERM_COLOUR.mem}
          data-term="dram"
        />
        <rect
          x={6 + bx(dram)}
          y={barY}
          width={Math.max(bx(net), net > 0 ? 1.5 : 0)}
          height={14}
          fill="url(#mesh-hatch)"
          data-term="eth"
        />
        <text
          x={6}
          y={barY + 16 + fs(11)}
          style={{ fontSize: fs(9.5) }}
          className="fill-neutral-700 dark:fill-neutral-300"
        >
          purple: GDDR6 streaming · hatched: Ethernet
        </text>
      </svg>
      <p className="mt-1 text-[0.7rem] text-neutral-600 dark:text-neutral-400">
        Each square is one Tensix core; its green fill is the share of this
        step&apos;s weights it holds, against its {fmtBytes(m.core_sram)} of
        SRAM. Purple columns are the card&apos;s GDDR6, the grey tabs on top its
        four Ethernet ports (orange when the cards are exchanging data).
      </p>
    </div>
  );

  const stats = (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
      <Stat
        label="Cards"
        value={fmtInt(cards)}
        hint={`at least ${fewest} to hold it`}
      />
      <Stat label="One layer" value={fmtTime(m.t)} />
      <Stat
        label="Ethernet share"
        value={
          m.t_eth === 0
            ? "0%"
            : pct(m.t_eth / m.t, m.t_eth / m.t < 0.001 ? 3 : 1)
        }
        hint="bandwidth only"
      />
      <Stat
        label="Layer per core"
        value={fmtBytes(m.layer_per_core)}
        hint={m.layer_fits ? "fits in SRAM" : "more than its SRAM"}
      />
    </div>
  );

  return (
    <AnimationPanel
      testId="mesh-widget"
      title="One decoder layer on a mesh of small cores"
      summary="Tensor parallel over Tenstorrent Blackhole cards (FP8): each card streams its share of every weight matrix from GDDR6 into its cores, and the cards all-reduce over their built-in Ethernet. Times are the model's bound from the card's stated bandwidths and peak rate."
      stepper={st}
      stepLabel="step"
      caption={meshCaption(s, steps.length, cards, m.cores, m.core_sram)}
      visual={visual}
      stats={stats}
      equation={children}
      hl={hover ?? (eth ? "eth" : "mem")}
      onEquationHover={setHover}
      params={
        <div className="flex flex-wrap gap-4">
          <Segmented
            label="Cards"
            value={String(cards)}
            options={CARDS.map((c) => ({ value: String(c), label: String(c) }))}
            onChange={(v) => setCards(Number(v))}
          />
          <Segmented
            label="Batch"
            value={String(batch)}
            options={[
              { value: "1", label: "1" },
              { value: "32", label: "32" },
            ]}
            onChange={(v) => setBatch(Number(v))}
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
