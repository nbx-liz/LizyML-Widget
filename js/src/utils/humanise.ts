/**
 * Convert a snake_case identifier to a Title Case label.
 *
 * Used as a fallback when the backend contract does not provide a
 * human-readable label for a generic identifier — e.g. an unknown CV
 * strategy added in the backend without a corresponding entry in
 * `cv_strategy_labels`. Keeps the UI rendering generic so a backend
 * change alone surfaces in the dropdown without a JS edit (#119).
 */
export function humaniseSnake(value: string): string {
  if (!value) return value;
  return value
    .split("_")
    .filter((part) => part.length > 0)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
    .join(" ");
}
