/** The site constants, the deck links and the palette the widgets share. */
import { describe, expect, it } from "vitest";

import { illustrative } from "@/lib/accel/model";
import { fmtBytes, fmtFlops, fmtInt, fmtTime, pct, trim } from "@/lib/format";
import { SECTIONS, getSectionMeta, isValidSlug } from "@/lib/mdx/sections";
import { PLACEMENT_COLOUR, TERM_COLOUR } from "@/lib/viz/palette";
import { SITE_NAME, SITE_TITLE, deck, repoFile } from "@/lib/site";
import { nameOf, placeOf } from "@/components/ui/SiteSwitch";

describe("site", () => {
  it("names itself as the shared switch does", () => {
    expect(SITE_TITLE.startsWith(`${SITE_NAME}:`)).toBe(true);
    expect(nameOf("accelerators")).toBe(SITE_NAME);
    expect(placeOf("accelerators")).toEqual({
      track: "Hardware",
      site: "Accelerators",
      title: SITE_TITLE,
    });
    // the Systolic site keeps its key; only its label and title changed
    expect(nameOf("silicon")).toBe("Systolic Arrays Explained");
    expect(placeOf("silicon").site).toBe("Systolic");
  });
  it("links deck slides and repository files", () => {
    expect(deck("Google_TPU_03_Systolic_Arrays", 7)).toBe(
      "https://brendanjameslynskey.github.io/Google_TPU_03_Systolic_Arrays/#slide-07",
    );
    expect(deck("Google_TPU_03_Systolic_Arrays")).toBe(
      "https://brendanjameslynskey.github.io/Google_TPU_03_Systolic_Arrays/",
    );
    expect(repoFile("results.md")).toBe(
      "https://github.com/BrendanJamesLynskey/ai-accelerators-explained/blob/main/results.md",
    );
  });
  it("lists ten chapters", () => {
    expect(SECTIONS).toHaveLength(10);
    expect(isValidSlug("04-wafer-scale")).toBe(true);
    expect(isValidSlug("10-choosing")).toBe(true);
    expect(isValidSlug("06-dataflow")).toBe(false);
    expect(getSectionMeta("04-wafer-scale").title).toBe("Wafer-scale");
  });
  it("colours memory by distance from the maths", () => {
    expect(PLACEMENT_COLOUR).toHaveLength(4);
    expect(new Set(Object.values(TERM_COLOUR)).size).toBe(5);
  });
  it("reads the illustrative parameters", () => {
    expect(illustrative("d0_per_cm2")).toBe(0.1);
  });
  it("formats", () => {
    expect(fmtBytes(230e6)).toBe("230 MB");
    expect(fmtFlops(125e15)).toBe("125 PFLOP/s");
    expect(fmtTime(0)).toBe("0 s");
    expect(fmtTime(2)).toBe("2 s");
    expect(fmtTime(2.5e-3)).toBe("2.5 ms");
    expect(fmtInt(1234.4)).toBe("1,234");
    expect(pct(0.5)).toBe("50%");
    expect(trim(0)).toBe("0");
  });
});
