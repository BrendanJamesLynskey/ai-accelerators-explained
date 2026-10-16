/**
 * The site's link-preview card (Open Graph), drawn at build time: see ./_og/card.tsx.
 */
import { card, OG_SIZE, OG_TYPE } from "./_og/card";

export const alt =
  "AI Accelerators Explained: The families of AI silicon, on the same work";
export const size = OG_SIZE;
export const contentType = OG_TYPE;

export default function Image() {
  return card();
}
