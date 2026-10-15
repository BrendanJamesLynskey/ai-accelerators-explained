"use client";

/**
 * Until brief 37C adds Accelerators to the shared site switch, globals.css
 * draws "Hardware › Accelerators" on the switch's button and hides the
 * switch's own place text. A <summary> needs an accessible name, and CSS
 * generated text does not count for every checker, so this names the button
 * once the page has loaded. 37C deletes this file with the CSS block.
 */
import { useEffect } from "react";

export function PrelaunchLabel(): null {
  useEffect(() => {
    const s = document.querySelector("[data-switch-prelaunch] summary");
    s?.setAttribute(
      "aria-label",
      "Hardware › Accelerators: open the list of companion sites",
    );
  }, []);
  return null;
}
