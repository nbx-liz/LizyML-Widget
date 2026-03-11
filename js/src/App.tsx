/** App — Header + Tab router. */
import { useState, useMemo, useEffect } from "preact/hooks";
import { useTraitlet, useSendAction } from "./hooks/useModel";
import { usePlot } from "./hooks/usePlot";
import { Header } from "./components/Header";
import { DataTab } from "./tabs/DataTab";
import { ConfigTab } from "./tabs/ConfigTab";
import { ResultsTab } from "./tabs/ResultsTab";

const TABS = ["Data", "Model", "Results"] as const;
type Tab = (typeof TABS)[number];

interface AppProps {
  model: any;
}

export function App({ model }: AppProps) {
  const [activeTab, setActiveTab] = useState<Tab>("Data");
  const backendInfo = useTraitlet<Record<string, any>>(model, "backend_info");
  const status = useTraitlet<string>(model, "status");
  const dfInfo = useTraitlet<Record<string, any>>(model, "df_info");
  const configSchema = useTraitlet<Record<string, any>>(model, "config_schema");
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

  const { plots, loading: plotLoading, requestPlot, clearCache } = usePlot(model);

  // Clear plot cache when a new job starts
  useEffect(() => {
    if (status === "running") clearCache();
  }, [status, clearCache]);

  // Auto-switch to Results tab when job starts/completes
  useEffect(() => {
    if (status === "running" || status === "completed" || status === "failed") {
      setActiveTab("Results");
    }
  }, [status]);

  // All column names for target dropdown
  const allColumns = useMemo(() => {
    const shape = dfInfo.shape;
    if (!shape) return [];
    const cols = (dfInfo.columns ?? []).map((c: any) => c.name);
    if (dfInfo.target && !cols.includes(dfInfo.target)) {
      cols.push(dfInfo.target);
    }
    return cols;
  }, [dfInfo]);

  return (
    <div class="lzw-root">
      <Header backendInfo={backendInfo} status={status} />

      {/* Tab bar */}
      <div class="lzw-tabs">
        {TABS.map((tab) => {
          const enabled =
            tab === "Data" ||
            (tab === "Model" && status !== "idle") ||
            (tab === "Results" && (jobIndex > 0 || status === "running" || status === "completed" || status === "failed"));
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
            sendAction={sendAction}
          />
        )}
        {activeTab === "Model" && (
          <ConfigTab
            configSchema={configSchema}
            config={config}
            dfInfo={dfInfo}
            status={status}
            sendAction={sendAction}
            model={model}
          />
        )}
        {activeTab === "Results" && (
          <ResultsTab
            status={status}
            jobType={jobType}
            jobIndex={jobIndex}
            progress={progress}
            elapsedSec={elapsedSec}
            fitSummary={fitSummary}
            tuneSummary={tuneSummary}
            availablePlots={availablePlots}
            inferenceResult={inferenceResult}
            error={error}
            plots={plots}
            plotLoading={plotLoading}
            onRequestPlot={requestPlot}
            sendAction={sendAction}
            onSwitchToFit={() => setActiveTab("Model")}
          />
        )}
      </div>
    </div>
  );
}
