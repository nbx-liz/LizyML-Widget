/** App — Header + Tab router. */
import { useState, useMemo, useEffect, useCallback } from "preact/hooks";
import { useTraitlet, useSendAction, useCustomMsg } from "./hooks/useModel";
import { useJobPolling } from "./hooks/useJobPolling";
import { usePlot } from "./hooks/usePlot";
import { useTheme } from "./hooks/useTheme";
import { Header } from "./components/Header";
import { DataTab } from "./tabs/DataTab";
import { ConfigTab } from "./tabs/ConfigTab";
import { ResultsTab } from "./tabs/ResultsTab";

const TABS = ["Data", "Model", "Results"] as const;
type Tab = (typeof TABS)[number];

interface AppProps {
  model: any;
  rootEl: HTMLElement;
}

export function App({ model, rootEl }: AppProps) {
  const [activeTab, setActiveTab] = useState<Tab>("Data");
  const backendInfo = useTraitlet<Record<string, any>>(model, "backend_info");
  const status = useTraitlet<string>(model, "status");
  const dfInfo = useTraitlet<Record<string, any>>(model, "df_info");
  const backendContract = useTraitlet<Record<string, any>>(model, "backend_contract");
  const config = useTraitlet<Record<string, any>>(model, "config");
  const jobType = useTraitlet<string>(model, "job_type");
  const jobIndex = useTraitlet<number>(model, "job_index");
  const progress = useTraitlet<Record<string, any>>(model, "progress");
  const elapsedSec = useTraitlet<number>(model, "elapsed_sec");
  const fitSummary = useTraitlet<Record<string, any>>(model, "fit_summary");
  const tuneSummary = useTraitlet<Record<string, any>>(model, "tune_summary");
  const availablePlots = useTraitlet<string[]>(model, "available_plots");
  const inferenceResult = useTraitlet<Record<string, any>>(model, "inference_result");
  const error = useTraitlet<Record<string, any>>(model, "error");
  const sendAction = useSendAction(model);
  const { resolved: theme, toggle: toggleTheme } = useTheme(rootEl);
  const polled = useJobPolling(model, status, elapsedSec);

  // Merge polled state over traitlet state (Colab fallback)
  const effectiveStatus = polled?.status ?? status;
  const effectiveProgress = polled?.progress ?? progress;
  const effectiveElapsedSec = polled?.elapsed_sec ?? elapsedSec;
  const effectiveJobType = polled?.job_type ?? jobType;
  const effectiveJobIndex = polled?.job_index ?? jobIndex;
  const effectiveError = polled?.error ?? error;
  const effectiveFitSummary = polled?.fit_summary ?? fitSummary;
  const effectiveTuneSummary = polled?.tune_summary ?? tuneSummary;
  const effectiveAvailablePlots = polled?.available_plots ?? availablePlots;

  const [columnStats, setColumnStats] = useState<Record<string, any> | null>(null);
  const [splitPreview, setSplitPreview] = useState<any | null>(null);
  const [exportLoading, setExportLoading] = useState(false);

  // Handle custom messages for column_stats, split_preview, code_export_download
  const handleCustomMsg = useCallback((msg: any, buffers?: ArrayBuffer[]) => {
    if (msg.type === "column_stats") {
      setColumnStats(msg);
    } else if (msg.type === "split_preview" || msg.type === "preview_splits") {
      setSplitPreview(msg);
    } else if (msg.type === "code_export_download" && buffers && buffers.length > 0) {
      const blob = new Blob([buffers[0]], { type: "application/zip" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = msg.filename || "exported_code.zip";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      setExportLoading(false);
    }
  }, []);
  useCustomMsg(model, handleCustomMsg);

  const { plots, loading: plotLoading, requestPlot, clearCache } = usePlot(model);

  // Clear plot cache when a new job starts
  useEffect(() => {
    if (effectiveStatus === "running") clearCache();
  }, [effectiveStatus, clearCache]);

  // Auto-switch to Results tab when job starts/completes
  useEffect(() => {
    if (effectiveStatus === "running" || effectiveStatus === "completed" || effectiveStatus === "failed") {
      setActiveTab("Results");
    }
  }, [effectiveStatus]);

  // All column names for target dropdown
  const allColumns = useMemo(() => {
    const shape = dfInfo.shape;
    if (!shape) return [];
    const base = (dfInfo.columns ?? []).map((c: any) => c.name);
    return dfInfo.target && !base.includes(dfInfo.target)
      ? [...base, dfInfo.target]
      : base;
  }, [dfInfo]);

  return (
    <div class="lzw-app">
      <Header backendInfo={backendInfo} status={effectiveStatus} theme={theme} onToggleTheme={toggleTheme} />

      {/* Tab bar */}
      <div class="lzw-tabs">
        {TABS.map((tab) => {
          const enabled =
            tab === "Data" ||
            (tab === "Model" && effectiveStatus !== "idle") ||
            (tab === "Results" && (effectiveJobIndex > 0 || effectiveStatus === "running" || effectiveStatus === "completed" || effectiveStatus === "failed"));
          return (
            <button
              key={tab}
              class={`lzw-tabs__btn ${activeTab === tab ? "lzw-tabs__btn--active" : ""} ${!enabled ? "lzw-tabs__btn--disabled" : ""}`}
              onClick={() => enabled && setActiveTab(tab)}
              type="button"
            >
              {tab}
            </button>
          );
        })}
      </div>

      {/* Tab content */}
      <div class="lzw-content">
        {activeTab === "Data" && (
          <DataTab
            dfInfo={dfInfo}
            allColumns={allColumns}
            columnStats={columnStats}
            splitPreview={splitPreview}
            sendAction={sendAction}
          />
        )}
        {activeTab === "Model" && (
          <ConfigTab
            backendContract={backendContract}
            config={config}
            dfInfo={dfInfo}
            status={effectiveStatus}
            sendAction={sendAction}
            model={model}
          />
        )}
        {activeTab === "Results" && (
          <ResultsTab
            status={effectiveStatus}
            jobType={effectiveJobType}
            jobIndex={effectiveJobIndex}
            progress={effectiveProgress}
            elapsedSec={effectiveElapsedSec}
            fitSummary={effectiveFitSummary}
            tuneSummary={effectiveTuneSummary}
            availablePlots={effectiveAvailablePlots}
            inferenceResult={inferenceResult}
            error={effectiveError}
            plots={plots}
            plotLoading={plotLoading}
            onRequestPlot={requestPlot}
            sendAction={sendAction}
            onSwitchToFit={() => setActiveTab("Model")}
            evaluationParams={config?.evaluation?.params}
            theme={theme}
            exportLoading={exportLoading}
            onExportCode={() => {
              setExportLoading(true);
              sendAction("export_code", {});
            }}
          />
        )}
      </div>
    </div>
  );
}
