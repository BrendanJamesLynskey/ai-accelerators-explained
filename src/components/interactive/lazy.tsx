"use client";

/**
 * Code-split client widgets: each loads its own chunk after the page shell,
 * so pages stay light (the pattern of the companion sites' lazy.tsx). Each
 * widget takes its equation as server-rendered children.
 */
import dynamic from "next/dynamic";

function Placeholder({ what }: { what: string }): JSX.Element {
  return (
    <p
      data-pending-widget
      className="my-8 min-h-96 text-sm text-neutral-600 dark:text-neutral-400"
    >
      Loading the {what}…
    </p>
  );
}

const loading = (what: string) =>
  function Loading(): JSX.Element {
    return <Placeholder what={what} />;
  };

export const MapWidget = dynamic(() => import("./MapWidget"), {
  ssr: false,
  loading: loading("animation"),
});
export const RooflineWidget = dynamic(() => import("./RooflineWidget"), {
  ssr: false,
  loading: loading("animation"),
});
export const MxuWidget = dynamic(() => import("./MxuWidget"), {
  ssr: false,
  loading: loading("animation"),
});
export const WaferWidget = dynamic(() => import("./WaferWidget"), {
  ssr: false,
  loading: loading("animation"),
});
export const ChainWidget = dynamic(() => import("./ChainWidget"), {
  ssr: false,
  loading: loading("animation"),
});
