/**
 * usePlot — request plots via msg:custom + receive via msg:custom.
 * Caches received plots by plot_type.
 *
 * B-1: Each request carries a monotonic request_id. Responses that don't
 * match the latest request_id for a given plot_type are discarded,
 * preventing out-of-order responses from corrupting the cache.
 */
import { useState, useCallback, useEffect, useRef } from "preact/hooks";

interface PlotCache {
  [plotType: string]: any; // parsed Plotly JSON spec
}

/** Options forwarded to Python's request_plot (P-026). */
export interface PlotRequestOptions {
  metrics?: string[];
}

export function usePlot(model: any) {
  const [plots, setPlots] = useState<PlotCache>({});
  const [loading, setLoading] = useState<Record<string, boolean>>({});
  // P-037 / #152: explicit error surface so PlotViewer can render
  // "Plot unavailable" instead of looping on the loading text. Cleared
  // on the next requestPlot for the same plot_type and on clearCache.
  const [errors, setErrors] = useState<Record<string, string>>({});
  // Track the latest request_id per plot_type for stale-response filtering
  const pendingRef = useRef<Record<string, string>>({});
  const nextRequestIdRef = useRef(1);
  // Track options used for cached plots to invalidate on change (P-026)
  const cachedOptionsRef = useRef<Record<string, string>>({});

  useEffect(() => {
    /** Check if a response should be accepted.
     * - If we have a pending request_id for this plot_type, the response
     *   must carry a matching request_id.  Stale or missing IDs are rejected.
     * - If no pending request_id exists (e.g. cache cleared, or legacy
     *   response without request_id support), accept the response. */
    const isAcceptable = (pt: string, rid: string | undefined): boolean => {
      const expected = pendingRef.current[pt];
      if (!expected) return true; // no pending request — accept (backward compat)
      return rid === expected; // strict match when we have a pending ID
    };

    const handler = (msg: any, buffers?: ArrayBuffer[]) => {
      if (msg.type === "plot_data") {
        const pt = msg.plot_type;
        if (!isAcceptable(pt, msg.request_id)) return;
        // D-1: Large plots arrive as binary buffer instead of inline JSON
        let jsonStr: string | undefined = msg.plotly_json;
        if (msg.binary && buffers && buffers.length > 0) {
          jsonStr = new TextDecoder().decode(buffers[0]);
        }
        if (!jsonStr) return;
        try {
          const spec = JSON.parse(jsonStr);
          setPlots((prev) => ({ ...prev, [pt]: spec }));
          // Successful response clears any stale error from a prior attempt.
          setErrors((prev) => {
            if (!(pt in prev)) return prev;
            const next = { ...prev };
            delete next[pt];
            return next;
          });
        } catch {
          // ignore parse errors
        }
        setLoading((prev) => ({ ...prev, [pt]: false }));
        delete pendingRef.current[pt];
      }
      if (msg.type === "plot_error") {
        const pt = msg.plot_type;
        if (!isAcceptable(pt, msg.request_id)) return;
        setLoading((prev) => ({ ...prev, [pt]: false }));
        // P-037: surface a non-empty message even when the backend forgets
        // to include one, so PlotViewer never falls back to a blank panel.
        const message =
          typeof msg.message === "string" && msg.message.length > 0
            ? msg.message
            : "Plot unavailable";
        setErrors((prev) => ({ ...prev, [pt]: message }));
        delete pendingRef.current[pt];
      }
    };
    model.on("msg:custom", handler);
    return () => model.off("msg:custom", handler);
  }, [model]);

  const requestPlot = useCallback(
    (plotType: string, options?: PlotRequestOptions) => {
      const optionsKey = options ? JSON.stringify(options) : "";
      const optionsChanged = cachedOptionsRef.current[plotType] !== optionsKey;
      // Skip if already cached with same options (P-026)
      if (plots[plotType] && !optionsChanged) return;
      // Skip if already loading with same options; allow re-request on options change
      if (loading[plotType] && !optionsChanged) return;
      const rid = `req-${nextRequestIdRef.current++}`;
      pendingRef.current[plotType] = rid;
      cachedOptionsRef.current[plotType] = optionsKey;
      setLoading((prev) => ({ ...prev, [plotType]: true }));
      // P-037: a fresh request invalidates any stale error so users never
      // see the previous failure message on top of a new loading panel.
      setErrors((prev) => {
        if (!(plotType in prev)) return prev;
        const next = { ...prev };
        delete next[plotType];
        return next;
      });
      const payload: Record<string, any> = { plot_type: plotType, request_id: rid };
      if (options) payload.options = options;
      model.send({
        type: "action",
        action_type: "request_plot",
        payload,
      });
    },
    [model, plots, loading],
  );

  const clearCache = useCallback(() => {
    setPlots({});
    setLoading({});
    setErrors({});
    pendingRef.current = {};
    cachedOptionsRef.current = {};
  }, []);

  return { plots, loading, errors, requestPlot, clearCache };
}
