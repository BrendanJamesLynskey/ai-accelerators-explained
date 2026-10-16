/**
 * Root layout for the App Router.
 *
 * Server Component, copied from transformer-explainer's layout (via the companion sites): HTML
 * scaffold, the global stylesheet and the site header. Dark mode follows
 * the system setting (`darkMode: "media"` in tailwind.config.ts).
 */
import type { Metadata } from "next";
import type { ReactNode } from "react";

import { SiteHeader } from "@/components/ui/SiteHeader";
import { SITE_URL } from "@/lib/site";
import { withShare } from "@/lib/share";

import "./globals.css";

export const metadata: Metadata = withShare("/", {
  metadataBase: new URL(SITE_URL),
  title: {
    default: "AI Accelerators Explained",
    template: "%s · AI Accelerators Explained",
  },
  description:
    "The families of AI silicon (GPUs, systolic designs, wafer-scale engines, compiler-scheduled chips and more), each run on the same decoder layer by a tested model, with every specification sourced and dated.",
});

export default function RootLayout({
  children,
}: {
  children: ReactNode;
}): JSX.Element {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen font-sans antialiased">
        <SiteHeader />
        {children}
      </body>
    </html>
  );
}
