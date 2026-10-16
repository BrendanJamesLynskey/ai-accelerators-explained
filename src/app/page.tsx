import type { Route } from "next";
import Link from "next/link";

import { CheckedStrip, StartButton } from "@/components/ui/Landing";
import { FamilyBars } from "@/components/viz/FamilyBars";
import { SECTIONS } from "@/lib/mdx/sections";
import { formatValue, lookup, type Fmt } from "@/lib/accel/values";
import {
  KERNELS_URL,
  NUMERICS_URL,
  RACKS_URL,
  SITE_NAME,
  SYSTOLIC_URL,
} from "@/lib/site";

const LINK =
  "focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent";

/**
 * Landing page: what the site is, the picture behind every chapter (how
 * many chips of each family hold the running model), and the ways in.
 * Server Component with no client JavaScript of its own.
 */
export default function HomePage(): JSX.Element {
  const v = (path: string, fmt: Fmt) => formatValue(lookup(path), fmt);
  return (
    <main className="mx-auto max-w-5xl px-4 py-12 sm:px-6 sm:py-20">
      <p className="font-mono text-xs uppercase tracking-widest text-accent dark:text-indigo-300">
        {SITE_NAME}
      </p>
      <h1 className="mt-4 text-4xl font-semibold tracking-tight sm:text-5xl">
        The families of AI silicon, on the same work.
      </h1>
      <div className="mt-6 grid items-start gap-8 md:grid-cols-[1fr_minmax(0,22rem)]">
        <div>
          <p className="max-w-2xl text-lg text-neutral-600 dark:text-neutral-300">
            An interactive comparison of the families of AI chips, from NVIDIA
            and AMD GPUs and TPUs to wafer-scale, compiler-scheduled and FPGA
            designs, running the same decoder.
          </p>
          <div className="mt-6">
            <StartButton href={`/learn/${SECTIONS[0]!.slug}` as Route} />
          </div>
        </div>
        <figure
          data-hero-figure
          className="rounded-lg border border-neutral-200 p-4 dark:border-neutral-800"
        >
          <FamilyBars />
          <figcaption className="mt-2 text-xs text-neutral-600 dark:text-neutral-400">
            Chips needed to hold Llama 3.1 70B and one 1,024-token KV cache, log
            scale: memory capacity is the first thing each family trades.
          </figcaption>
        </figure>
      </div>
      <CheckedStrip href="/about#validation">
        Every specification carries its source and the date it was read, vendor
        claims are never used as inputs, and the model is checked against
        published capacity, throughput and latency figures.
      </CheckedStrip>
      <section
        aria-labelledby="finding-heading"
        className="mt-12"
        data-floor-finding
      >
        <h2
          id="finding-heading"
          className="text-xl font-semibold tracking-tight"
        >
          The finding: once weights are in SRAM, a layer costs tens of
          microseconds
        </h2>
        <p className="mt-4 max-w-2xl text-neutral-600 dark:text-neutral-300">
          Groq&apos;s chain of chips, Cerebras&apos;s wafers and
          SambaNova&apos;s SN40L are built very differently, yet each pays{" "}
          {v("floor.min", "time")} to {v("floor.max", "time")} per layer beyond
          what reading its weights costs. On a wafer the weight reads take
          nanoseconds; the per-layer critical path (synchronisation,
          collectives, crossing the mesh) is the token. A single per-layer
          floor, fitted on three of the machines, predicts the fourth within{" "}
          {v("floor.worstPred", "num")}× where the bandwidth bound alone misses
          by up to {v("floor.worstBound", "int")}×.{" "}
          <Link href={"/learn/10-choosing" as Route} className={LINK}>
            Chapter 10
          </Link>{" "}
          animates it. The GPUs, with prefill counted, reach about a third of
          their tensor peak in MLPerf: {v("val.mlperf-h100.ratio", "pct")} for
          eight H100s and {v("val.mlperf-gb200.ratio", "pct")} for four GB200s.
        </p>
      </section>
      <section aria-labelledby="learn-heading" className="mt-12">
        <h2 id="learn-heading" className="text-xl font-semibold tracking-tight">
          What you&apos;ll learn
        </h2>
        <p className="mt-4 max-w-2xl text-neutral-600 dark:text-neutral-300">
          Every AI chip multiplies activations by weights. What separates a GPU
          from a TPU, a wafer-scale engine or an LPU is where the weights live,
          who decides when data moves, and how big the engines are. An H100
          reads its HBM at {v("part.h100.off_bw", "rate")}; a Cerebras wafer
          reads its SRAM at {v("part.wse-3.sram_bw", "rate")}, but holds{" "}
          {v("part.wse-3.sram_bytes", "bytes")}, so a 70B model needs{" "}
          {v("placement(wse-3,llama3-70b,bf16,1,1024).chips", "int")} wafers
          where it needs one GPU in 8-bit form.
        </p>
        <p className="mt-4 max-w-2xl text-neutral-600 dark:text-neutral-300">
          Each chapter runs the same decoder on one family, with an animation
          whose every frame comes from a small model tested against a Python
          reference. Every specification carries its source and the date it was
          read, announced parts are marked, vendor claims are quoted but never
          used as inputs, and the model is checked against published figures (
          <Link href="/about#validation" className={LINK}>
            its validation band
          </Link>
          ).
        </p>
      </section>
      <nav
        aria-label="Chapters"
        className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-3"
      >
        {SECTIONS.map((s, i) => (
          <Link
            key={s.slug}
            href={`/learn/${s.slug}`}
            className="focus-ring group rounded-lg border border-neutral-200 p-5 hover:border-accent dark:border-neutral-800 dark:hover:border-indigo-400"
          >
            <p className="font-mono text-xs text-neutral-500 dark:text-neutral-400">
              {String(i + 1).padStart(2, "0")}
            </p>
            <h2 className="mt-1 font-semibold">{s.title}</h2>
            <p className="mt-2 text-sm text-neutral-600 dark:text-neutral-400">
              {s.summary}
            </p>
          </Link>
        ))}
      </nav>
      <p className="mt-6 text-sm text-neutral-600 dark:text-neutral-400">
        Every figure the model uses, with its source and access date:{" "}
        <Link href="/specs" className={LINK}>
          the specifications
        </Link>
        .
      </p>
      <p className="mt-12 text-sm text-neutral-600 dark:text-neutral-400">
        Part of the companion sites&apos; Hardware track, with{" "}
        <a href={KERNELS_URL} className={LINK}>
          GPU Kernels Explained
        </a>{" "}
        (how a GPU executes the maths),{" "}
        <a href={SYSTOLIC_URL} className={LINK}>
          Systolic Arrays Explained
        </a>{" "}
        (inside a matrix engine) and{" "}
        <a href={RACKS_URL} className={LINK}>
          AI Racks Explained
        </a>{" "}
        (the machines these chips go into). The number formats they compute in:{" "}
        <a href={NUMERICS_URL} className={LINK}>
          Numerics
        </a>
        . How this site was built, and how to check it:{" "}
        <Link href="/about" className={LINK}>
          about
        </Link>
        .
      </p>
    </main>
  );
}
