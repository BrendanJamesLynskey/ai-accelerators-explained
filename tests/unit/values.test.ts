/**
 * The numbers the chapters print: every <V of="…" /> path in the MDX
 * resolves in the model, and a few are recomputed here from the sources.
 */
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

import { formatValue, lookup, type Fmt } from "@/lib/accel/values";

const DIR = path.join(__dirname, "../../content/chapters");
const FILES = readdirSync(DIR).filter((f) => f.endsWith(".mdx"));

describe("model values quoted in the chapters", () => {
  const uses: { file: string; of: string; fmt: string }[] = [];
  for (const f of FILES) {
    const src = readFileSync(path.join(DIR, f), "utf8");
    for (const m of src.matchAll(/<V of="([^"]+)"(?: fmt="([^"]+)")? \/>/g))
      uses.push({ file: f, of: m[1]!, fmt: m[2] ?? "num" });
  }
  it("are used in every chapter", () => {
    expect(uses.length).toBeGreaterThan(60);
    for (const f of FILES)
      expect(uses.filter((u) => u.file === f).length, f).toBeGreaterThan(5);
  });
  it("all resolve to a number or string", () => {
    for (const u of uses) {
      const v = lookup(u.of);
      expect(["number", "string"], `${u.file}: ${u.of}`).toContain(typeof v);
      expect(formatValue(v, u.fmt as Fmt).length).toBeGreaterThan(0);
    }
  });
  it("print what the sources state", () => {
    expect(formatValue(lookup("part.h100.off_bw"), "rate")).toBe("3.35 TB/s");
    expect(formatValue(lookup("part.wse-3.sram_bw"), "rate")).toBe("21 PB/s");
    expect(formatValue(lookup("part.wse-3.sram_bytes"), "bytes")).toBe("44 GB");
    expect(formatValue(lookup("part.groqchip.hop_s"), "time")).toBe("722 ns");
    expect(formatValue(lookup("ridge(h100,fp8)"), "num")).toBe("591");
    expect(
      formatValue(
        lookup("placement(wse-3,llama3-70b,bf16,1,1024).chips"),
        "int",
      ),
    ).toBe("4");
    expect(formatValue(lookup("val.mlperf-h100.ratio"), "pct")).toBe("19%");
    expect(formatValue(lookup("pub.groq-70b-chips"), "int")).toBe("576");
  });
  it("formats every kind", () => {
    const kinds: [Fmt, number, string][] = [
      ["bits", 1.2e12, "1.2 Tb/s"],
      ["flops", 1979e12, "1.98 PFLOP/s"],
      ["watts", 27e3, "27 kW"],
      ["joules", 0.5, "0.5 J"],
      ["pct", 0.003, "0.30%"],
      ["pct", 0.05, "5.0%"],
      ["raw", 7, "7"],
      ["num", 3644.2, "3,640"],
    ];
    for (const [f, v, s] of kinds) expect(formatValue(v, f)).toBe(s);
    expect(formatValue("x", "rate")).toBe("x");
  });
  it("rejects paths that do not exist", () => {
    expect(() => lookup("part.nope.x")).toThrow();
    expect(() => lookup("nope(1)")).toThrow();
    expect(() => lookup("decode(h100,llama3-70b,fp8,1,1024)")).toThrow();
    expect(() => lookup("part.h100")).toThrow();
  });
});
