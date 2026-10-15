/**
 * Site-wide constants: this site's URL and name, its companion sites, the
 * owner's slide series it links into, and its repository.
 */

/** This site (production). */
export const SITE_URL = "https://ai-accelerators-explained.vercel.app";

/**
 * This site's name and title, for the home and About pages and the metadata.
 * The header takes them from the shared switch (`nameOf("accelerators")`,
 * `placeOf("accelerators").title`); a unit test keeps the two equal.
 */
export const SITE_NAME = "AI Accelerators Explained";
export const SITE_TITLE =
  "AI Accelerators Explained: the families of AI silicon";

/** The companion sites. */
export const KERNELS_URL = "https://gpu-kernels-explained.vercel.app";
export const SYSTOLIC_URL = "https://systolic-arrays-explained.vercel.app";
export const NUMERICS_URL = "https://numerics-explained.vercel.app";
export const RACKS_URL = "https://ai-racks-explained.vercel.app";
export const INFERENCE_URL = "https://llm-inference-explained.vercel.app";
export const TRANSIT_URL = "https://compute-in-transit-results.vercel.app";

export const GITHUB_URL =
  "https://github.com/BrendanJamesLynskey/ai-accelerators-explained";

/** The author, credited in one quiet line at the foot of the About page. */
export const AUTHOR = {
  name: "Brendan Lynskey",
  url: "https://github.com/BrendanJamesLynskey",
} as const;

/** The slide series the chapters link into ("go deeper"). */
export const GOOGLE_TPU_HUB =
  "https://brendanjameslynskey.github.io/LLM_Hub_Google_TPUs/";
export const NVIDIA_GPU_HUB =
  "https://brendanjameslynskey.github.io/LLM_Hub_NVIDIA_GPUs/";

/** A slide in one of the owner's decks (anchors are #slide-NN). */
export function deck(repo: string, slide?: number): string {
  const base = `https://brendanjameslynskey.github.io/${repo}/`;
  return slide === undefined
    ? base
    : `${base}#slide-${String(slide).padStart(2, "0")}`;
}

/** A file in this site's repository on GitHub. */
export function repoFile(path: string): string {
  return `${GITHUB_URL}/blob/main/${path}`;
}
