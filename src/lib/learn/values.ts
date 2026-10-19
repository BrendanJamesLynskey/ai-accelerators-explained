/**
 * The learning layer's numbers (brief 44) that are not already a path in
 * the chapters' value tree (`src/lib/accel/values.ts`): each is computed here
 * from the tested model's values, at build time. A self-check cites one as
 * `<Check v="…">`, prose as `<V v="…" />`; tests/unit/learn.test.ts checks
 * every one.
 */
import { lookup } from "@/lib/accel/values";

const num = (path: string): number => {
  const v = lookup(path);
  if (typeof v !== "number") throw new Error(`${path} is not a number`);
  return v;
};

const VALUES: Record<string, () => number> = {
  // short names for long model calls the self-checks print
  "ch05.oneChip": () =>
    num("decode(groqchip,llama2-70b,int8,1,1024,0,1).t_step"),
  "ch05.eightChips": () =>
    num("decode(groqchip,llama2-70b,int8,1,1024,576,8).t_step"),
  "ch06.prefillKernels": () =>
    num("layerTraffic(sn40l,llama3-8b,bf16,2048,1,2048,true,false).t"),
  "ch06.prefillFused": () =>
    num("layerTraffic(sn40l,llama3-8b,bf16,2048,1,2048,true,true).t"),
  "ch06.decodeKernels": () =>
    num("layerTraffic(sn40l,llama3-8b,bf16,1,1,8192,false,false).bytes"),
  "ch06.decodeFused": () =>
    num("layerTraffic(sn40l,llama3-8b,bf16,1,1,8192,false,true).bytes"),
  "ch07.eth": () => num("meshLayer(blackhole,llama3-70b,fp8,1,1024,3).t_eth"),
  "ch07.layer": () => num("meshLayer(blackhole,llama3-70b,fp8,1,1024,3).t"),
  // chapter 5: one chip per stage against eight (Llama 2 70B, INT8, 576 chips)
  "ch05.oneChipOverEight": () =>
    num("decode(groqchip,llama2-70b,int8,1,1024,0,1).t_step") /
    num("decode(groqchip,llama2-70b,int8,1,1024,576,8).t_step"),
  // chapter 6: a 2,048-token prefill layer, kernel by kernel over fused
  "ch06.prefillSpeedup": () =>
    num("layerTraffic(sn40l,llama3-8b,bf16,2048,1,2048,true,false).t") /
    num("layerTraffic(sn40l,llama3-8b,bf16,2048,1,2048,true,true).t"),
  // chapter 8: 4-bit over 8-bit weights on the M4
  "ch08.fourOverEight": () =>
    num("edgeBound(m4,llama3-8b,4,1024).tok_s") /
    num("edgeBound(m4,llama3-8b,8,1024).tok_s"),
};

export const LEARN_VALUE_NAMES = Object.keys(VALUES);

const CACHE = new Map<string, number>();

export function learnValue(name: string): number {
  const cached = CACHE.get(name);
  if (cached !== undefined) return cached;
  const f = VALUES[name];
  if (!f) throw new Error(`no learn value ${name}`);
  const v = f();
  if (!Number.isFinite(v)) throw new Error(`learn value ${name} is ${v}`);
  CACHE.set(name, v);
  return v;
}
