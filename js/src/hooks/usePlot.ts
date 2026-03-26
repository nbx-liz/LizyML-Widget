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

export function usePlot(model: any) {
  const [plots, setPlots] = useState<PlotCache>({});
  const [loading, setLoading] = useState<Record<string, boolean>>({});
  // Track the latest request_id per plot_type for stale-response filtering
  const pendingRef = useRef<Record<string, string>>({});
  const nextRequestIdRef = useRef(1);

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

    const handler = (msg: any) => {
      if (msg.type === "plot_data" && msg.plotly_json) {
        const pt = msg.plot_type;
        if (!isAcceptable(pt, msg.request_id)) return;
        try {
          const spec = JSON.parse(msg.plotly_json);
          setPlots((prev) => ({ ...prev, [pt]: spec }));
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
        delete pendingRef.current[pt];
      }
    };
    model.on("msg:custom", handler);
    return () => model.off("msg:custom", handler);
  }, [model]);

  const requestPlot = useCallback(
    (plotType: string) => {
      if (plots[plotType]) return; // already cached
      if (loading[plotType]) return; // already loading
      const rid = `req-${nextRequestIdRef.current++}`;
      pendingRef.current[plotType] = rid;
      setLoading((prev) => ({ ...prev, [plotType]: true }));
      model.send({
        type: "action",
        action_type: "request_plot",
        payload: { plot_type: plotType, request_id: rid },
      });
    },
    [model, plots, loading],
  );

  const clearCache = useCallback(() => {
    setPlots({});
    setLoading({});
    pendingRef.current = {};
  }, []);

  return { plots, loading, requestPlot, clearCache };
}
