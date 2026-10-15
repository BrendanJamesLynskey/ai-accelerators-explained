/**
 * Site-wide header.
 *
 * Server Component, adapted from transformer-explainer's `SiteHeader`
 * (same layout and classes), as on the other companion sites: no accounts,
 * so the sign-in controls are replaced by the cross-site switch. The brand
 * names the site; the switch's button says where it sits ("Track › Site").
 *
 * Until brief 37C adds Accelerators to the shared switch (SiteSwitch.tsx is
 * byte-identical on every site and has no entry for this site yet), the brand
 * comes from `SITE_NAME` and the switch is rendered for an unknown key, which
 * highlights no site; `data-switch-prelaunch` relabels its button "Hardware ›
 * Accelerators" (globals.css), and `PrelaunchLabel` gives it that name. 37C replaces this with `nameOf("accelerators")`
 * and `current="accelerators"`, like every other site.
 */
import Link from "next/link";

import { SITE_NAME, SITE_TITLE } from "@/lib/site";

import { PrelaunchLabel } from "./PrelaunchLabel";
import { SiteSwitch, type SiteKey } from "./SiteSwitch";

/** Not a key of the switch yet (see above): no entry is highlighted. */
const PRELAUNCH_KEY = "accelerators" as unknown as SiteKey;

const NAV = [
  { href: "/learn", label: "Learn" },
  { href: "/specs", label: "Specs" },
  { href: "/about", label: "About" },
] as const;

export function SiteHeader(): JSX.Element {
  return (
    <header className="border-b border-neutral-200 dark:border-neutral-800">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-x-4 gap-y-2 px-4 py-3 sm:px-6">
        <Link
          href="/"
          title={SITE_TITLE}
          data-site-brand
          className="focus-ring whitespace-nowrap rounded text-sm font-medium tracking-tight"
        >
          {SITE_NAME}
        </Link>
        <nav
          aria-label="Site"
          className="flex flex-wrap items-center gap-1 text-sm sm:gap-3"
        >
          {NAV.map((n) => (
            <Link
              key={n.href}
              href={n.href}
              className="focus-ring rounded px-2 py-1 text-neutral-700 hover:text-neutral-950 dark:text-neutral-300 dark:hover:text-white"
            >
              {n.label}
            </Link>
          ))}
          <span data-switch-prelaunch className="contents">
            <SiteSwitch current={PRELAUNCH_KEY} />
          </span>
          <PrelaunchLabel />
        </nav>
      </div>
    </header>
  );
}
