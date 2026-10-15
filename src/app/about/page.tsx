/**
 * /about: what the site is, what the model computes and what it leaves
 * out, how figures are sourced, the validation band, and how the
 * animations are driven and checked. Server Component, static.
 */
import Link from "next/link";

import { ValidationTable } from "@/components/mdx/Tables";
import { ACCESSED, band, validation } from "@/lib/accel/model";
import { formatValue, lookup } from "@/lib/accel/values";
import { trim } from "@/lib/format";
import {
  AUTHOR,
  GITHUB_URL,
  GOOGLE_TPU_HUB,
  KERNELS_URL,
  NUMERICS_URL,
  NVIDIA_GPU_HUB,
  RACKS_URL,
  SITE_NAME,
  SYSTOLIC_URL,
  repoFile,
} from "@/lib/site";

export const metadata = {
  title: "About",
  description:
    "What AI Accelerators Explained's model computes, how every specification is sourced and dated, the model's validation band against published figures, and how the animations are checked.",
};

const A =
  "focus-ring rounded text-accent underline underline-offset-2 dark:text-indigo-300";

export default function AboutPage(): JSX.Element {
  const rows = validation();
  const b = band();
  const v = (p: string) => formatValue(lookup(p), "time");
  return (
    <main className="mx-auto max-w-3xl px-4 py-12 sm:px-6">
      <p className="font-mono text-xs uppercase tracking-widest text-accent dark:text-indigo-300">
        /about
      </p>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">
        About this site
      </h1>
      <div className="mdx-content mt-6">
        <p>
          {SITE_NAME} compares the families of AI chips on one piece of work: a
          Llama decoder&apos;s prefill and decode steps. The question is the
          same for every family: where do the weights and activations live, what
          moves on each step, and who decides when it moves. It sits in the
          companion sites&apos; Hardware track beside{" "}
          <a href={KERNELS_URL} className={A}>
            GPU Kernels Explained
          </a>
          ,{" "}
          <a href={SYSTOLIC_URL} className={A}>
            Systolic Arrays Explained
          </a>{" "}
          and{" "}
          <a href={RACKS_URL} className={A}>
            AI Racks Explained
          </a>
          , and leans on{" "}
          <a href={NUMERICS_URL} className={A}>
            Numerics Explained
          </a>{" "}
          for number formats. Chapters link the matching slides of the{" "}
          <a href={NVIDIA_GPU_HUB} className={A}>
            NVIDIA GPU
          </a>{" "}
          and{" "}
          <a href={GOOGLE_TPU_HUB} className={A}>
            Google TPU
          </a>{" "}
          series. It picks one or two representative parts per family; it is not
          a catalogue, and naming a product is not an endorsement.
        </p>

        <h2>The model</h2>
        <p>
          <a href={repoFile("accel_model/core.py")} className={A}>
            accel_model/
          </a>{" "}
          is a <strong>teaching model</strong>, not a simulator of any product.
          For each part it computes where a model&apos;s weights and KV cache
          must live (HBM when the part has it, SRAM when it does not) and the
          fewest chips that hold them; the bound on one decode step and one
          prefill, the larger of bytes over bandwidth and flops over the peak
          rate; and the chip-to-chip latency the layout forces, where the vendor
          gives it. Families differ explicitly, not through fudge factors: GPUs
          and systolic chips split every layer across their chips (tensor
          layout); wafers and LPUs cut the model at layer boundaries into a
          pipeline, so one token&apos;s time is the sum of the stages&apos;
          times plus a hop between each, and a static schedule reads every
          weight from SRAM exactly once. It also tiles a matmul on a
          weight-stationary array (chapter 3) and counts defects on a wafer
          (chapter 4).
        </p>
        <ul>
          <li>
            Every bound assumes the full stated bandwidth and the dense peak
            rate; kernel launches, synchronisation and achieved-versus-peak
            efficiency are not modelled.
          </li>
          <li>
            Where a vendor gives only a sparse peak (Cerebras), the model uses
            it and marks the result &ldquo;sparse&rdquo;: a compute term from it
            is a floor.
          </li>
          <li>
            Latency between chips is included only where the sources give it
            (Groq&apos;s 722 ns per hop, the CS-4&apos;s 2 µs between wafers);
            NVLink, ICI and NeuronLink latencies are not disclosed, so the GPU
            and systolic bounds have no collective time.
          </li>
          <li>
            Chip counts are capacity only: weights plus the KV cache, with no
            room for activations, runtime buffers or spares.
          </li>
          <li>
            Energy per token is the chips&apos; stated power over the
            bound&apos;s throughput; where a vendor gives only system power (DGX
            B200, CS-3) it is a system share, and where it gives none it is not
            shown.
          </li>
          <li>
            The defect density and spare share in chapter 4 are illustrative.
          </li>
        </ul>

        <h2>How figures are sourced</h2>
        <p>
          Every figure is a field with a value, a status and a source, read on{" "}
          {ACCESSED} (
          <Link href="/specs" className={A}>
            the full list
          </Link>
          ). Vendor datasheets, documentation and product pages come first, then
          papers (checked by arXiv ID) and benchmark logs. A figure only a third
          party reports is marked <em>reported</em>. Where the sources used do
          not give a figure, the site says &ldquo;not disclosed&rdquo; rather
          than estimating it. Announced parts (the CS-4&apos;s WSE-3 Turbo, the
          Groq 3 LPU) are labelled, and a retired one (the first-generation
          GroqChip) is kept because published results used it. Vendor claims
          (&ldquo;up to N×&rdquo;) are quoted and attributed, never used as
          model inputs.
        </p>

        <h2 id="validation">Validation band</h2>
        <p>
          The model is checked against {rows.length} published figures it does
          not take as inputs, of three kinds. <strong>Capacity</strong>: how
          many chips a model needs. Cerebras&apos;s &ldquo;as few as four
          systems&rdquo; for a 16-bit 70B model and one system for 20B match the
          model exactly; Groq&apos;s Llama 2 70B demonstration used{" "}
          {trim(b.capacity.max, 3)} times the model&apos;s minimum, which is a
          floor (no room for activations, and whole nodes of eight).{" "}
          <strong>Throughput</strong>: MLPerf&apos;s eight H100s reached{" "}
          {formatValue(lookup("val.mlperf-h100.ratio"), "pct")} of the compute
          ceiling, a typical share for a real serving stack that also runs
          prefill and attention. <strong>Latency</strong>: for one user on SRAM
          designs, the published speeds are {trim(b.latency.min * 100, 2)}% to{" "}
          {trim(b.latency.max * 100, 2)}% of the bandwidth bound. That miss is
          the model&apos;s main limit, and the chapters say so: once the weights
          are in SRAM, bandwidth stops being what limits one user, and the
          per-layer latency the sources do not break down takes over. The
          published speeds imply {v("overhead.cerebras-8b-tps.per_layer")} to{" "}
          {v("overhead.groq-70b-tps.per_layer")} per layer beyond the bound.
        </p>
        <ValidationTable />

        <h2>How it is checked</h2>
        <ul>
          <li>
            <strong>Reference tests.</strong>{" "}
            <a
              href={repoFile("tests/python/test_accel_model.py")}
              className={A}
            >
              tests/python/test_accel_model.py
            </a>{" "}
            checks every field has a known source, status and date, that every
            source is cited, the parameter counts against the model names, the
            placement sums, the decode terms, the tile model against its
            per-tile steps, the defect generator, the chain against the decode
            bound, and the validation checks.
          </li>
          <li>
            <strong>Exact parity.</strong> The TypeScript port,{" "}
            <a href={repoFile("src/lib/accel/model.ts")} className={A}>
              src/lib/accel/model.ts
            </a>
            , repeats every function with the same operations in the same order;{" "}
            <a href={repoFile("scripts/make_fixtures.py")} className={A}>
              scripts/make_fixtures.py
            </a>{" "}
            writes the reference&apos;s results over every parameter the widgets
            offer, and the unit tests require the port to reproduce all of them
            exactly (the two exponentials of the yield model to 1e-12). CI fails
            if the fixtures, the site data or{" "}
            <a href={repoFile("results.md")} className={A}>
              results.md
            </a>{" "}
            are out of date.
          </li>
          <li>
            <strong>Animations from the model.</strong> Every animation draws a
            sequence of states the model computes; a frame is a pure function of
            one state. The end-to-end tests set chosen frames of every animation
            and require the caption to match the caption built from the Python
            reference&apos;s state for that frame.
          </li>
          <li>
            <strong>Numbers in the prose</strong> are printed from the model
            when the page is built, not typed.
          </li>
        </ul>

        <h2>The animations</h2>
        <p>
          Every animation has play and pause, step back and forward, a scrub
          bar, speeds from 0.25× to 4× and reset; with the animation focused,
          Space plays or pauses and the arrow keys step. Each step has a
          one-line caption, also announced to screen readers. With{" "}
          <em>reduce motion</em> set in your system, nothing plays by itself.
          Animations pause when scrolled out of view. Colours come from Okabe
          and Ito&apos;s colour-blind-safe palette, the same in light and dark
          mode and on every companion site: off-chip memory (HBM) purple,
          on-chip SRAM green, tiered memory sky blue, compute in the memory
          array orange, compute yellow; waiting (a chip-to-chip hop, a defect)
          is always a hatched pattern or a mark as well as a warning colour.
        </p>

        <h2>Source</h2>
        <p>
          The code, the model and the tests are on{" "}
          <a href={GITHUB_URL} className={A}>
            GitHub
          </a>{" "}
          (MIT licence). The design system is copied from the companion sites;
          the README records where each piece came from.
        </p>
      </div>
      <p
        className="mt-12 text-sm text-neutral-600 dark:text-neutral-400"
        data-author-credit
      >
        Built by{" "}
        <a className={A} href={AUTHOR.url}>
          {AUTHOR.name}
        </a>
      </p>
    </main>
  );
}
