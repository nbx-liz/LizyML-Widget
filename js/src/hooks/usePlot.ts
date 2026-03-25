/**
 * usePlot — request plots via msg:custom + receive via msg:custom.
 * Caches received plots by plot_type.
 */
import { useState, useCallback, useEffect } from "preact/hooks";

interface PlotCache {
  [plotType: string]: any; // parsed Plotly JSON spec
}

export function usePlot(model: any) {
  const [plots, setPlots] = useState<PlotCache>({});
  const [loading, setLoading] = useState<Record<string, boolean>>({});

  useEffect(() => {
    const handler = (msg: any) => {
      if (msg.type === "plot_data" && msg.plotly_json) {
        try {
          const spec = JSON.parse(msg.plotly_json);
          setPlots((prev) => ({ ...prev, [msg.plot_type]: spec }));
        } catch {
          // ignore parse errors
        }
        setLoading((prev) => ({ ...prev, [msg.plot_type]: false }));
      }
      if (msg.type === "plot_error") {
        setLoading((prev) => ({ ...prev, [msg.plot_type]: false }));
      }
    };
    model.on("msg:custom", handler);
    return () => model.off("msg:custom", handler);
  }, [model]);

  const requestPlot = useCallback(
    (plotType: string) => {
      if (plots[plotType]) return; // already cached
      if (loading[plotType]) return; // already loading
      setLoading((prev) => ({ ...prev, [plotType]: true }));
      model.send({
        type: "action",
        action_type: "request_plot",
        payload: { plot_type: plotType },
      });
    },
    [model, plots, loading],
  );

  const clearCache = useCallback(() => {
    setPlots({});
    setLoading({});
  }, []);

  return { plots, loading, requestPlot, clearCache };
}
