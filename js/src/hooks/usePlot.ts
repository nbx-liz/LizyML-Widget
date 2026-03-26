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

let _nextRequestId = 1;

export function usePlot(model: any) {
  const [plots, setPlots] = useState<PlotCache>({});
  const [loading, setLoading] = useState<Record<string, boolean>>({});
  // Track the latest request_id per plot_type for stale-response filtering
  const pendingRef = useRef<Record<string, string>>({});

  useEffect(() => {
    const handler = (msg: any) => {
      if (msg.type === "plot_data" && msg.plotly_json) {
        const rid = msg.request_id;
        const pt = msg.plot_type;
        // B-1: Discard stale responses (request_id mismatch)
        if (rid && pendingRef.current[pt] && rid !== pendingRef.current[pt]) {
          return;
        }
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
        const rid = msg.request_id;
        const pt = msg.plot_type;
        if (rid && pendingRef.current[pt] && rid !== pendingRef.current[pt]) {
          return;
        }
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
      const rid = `req-${_nextRequestId++}`;
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
