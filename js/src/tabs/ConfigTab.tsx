/**
 * ConfigTab — Coordinator for Fit/Tune sub-tabs.
 * Manages shared state (localConfig, debounce timer) and delegates rendering
 * to FitSubTab and TuneSubTab.
 */
import { useState, useRef, useCallback, useEffect, useMemo } from "preact/hooks";
import { useCustomMsg } from "../hooks/useModel";
import { FitSubTab } from "./FitSubTab";
import { TuneSubTab } from "./TuneSubTab";
import { diffToPatchOps } from "./configHelpers";

type SubTab = "fit" | "tune";

interface ConfigTabProps {
  backendContract: Record<string, any>;
  config: Record<string, any>;
  dfInfo: Record<string, any>;
  status: string;
  sendAction: (type: string, payload?: Record<string, any>) => void;
  model?: any;
}

export function ConfigTab({
  backendContract,
  config,
  dfInfo,
  status,
  sendAction,
  model,
}: ConfigTabProps) {
  const [subTab, setSubTab] = useState<SubTab>("fit");
  const [localConfig, setLocalConfig] = useState<Record<string, any>>(config);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastSentRef = useRef<Record<string, any>>(config);

  const uiSchema = backendContract?.ui_schema ?? {};
  const configSchema = backendContract?.config_schema ?? {};
  const optionSets: Record<string, Record<string, string[]>> = uiSchema.option_sets ?? {};

  // Sync from Python → local when Python config changes externally
  useEffect(() => {
    setLocalConfig(config);
    lastSentRef.current = config;
  }, [config]);

  // Cleanup debounce timer on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  const task = dfInfo?.task ?? "";

  // Auto-set tune evaluation metrics when task changes
  const evalMetricOpts = useMemo(
    () => optionSets.metric?.[task] ?? [],
    [optionSets, task],
  );
  const localConfigRef = useRef(localConfig);
  localConfigRef.current = localConfig;
  useEffect(() => {
    if (evalMetricOpts.length === 0) return;
    const cur = localConfigRef.current;
    const tuning = cur.tuning ?? {};
    const tuneEval = tuning.evaluation ?? {};
    const currentMetrics: string[] = tuneEval.metrics ?? [];
    if (currentMetrics.length === 0) {
      const updated = {
        ...cur,
        tuning: { ...tuning, evaluation: { ...tuneEval, metrics: [evalMetricOpts[0]] } },
      };
      setLocalConfig(updated);
      if (timerRef.current) clearTimeout(timerRef.current);
      timerRef.current = setTimeout(() => {
        const ops = diffToPatchOps(lastSentRef.current, updated);
        if (ops.length > 0) {
          sendAction("patch_config", { ops });
          lastSentRef.current = updated;
        }
      }, 300);
    }
  }, [task, evalMetricOpts, sendAction]);

  // Debounced send to Python via patch_config
  const handleChange = useCallback(
    (newConfig: Record<string, any>) => {
      // B-5: Reject config changes while job is running
      if (status === "running") return;
      setLocalConfig(newConfig);
      if (timerRef.current) clearTimeout(timerRef.current);
      timerRef.current = setTimeout(() => {
        const ops = diffToPatchOps(lastSentRef.current, newConfig);
        if (ops.length > 0) {
          sendAction("patch_config", { ops });
          lastSentRef.current = newConfig;
        }
      }, 300);
    },
    [sendAction, status],
  );

  const handleSectionChange = useCallback(
    (sectionKey: string, sectionValue: Record<string, any>) => {
      handleChange({ ...localConfig, [sectionKey]: sectionValue });
    },
    [localConfig, handleChange],
  );

  const canRun =
    status === "data_loaded" || status === "completed" || status === "failed";

  // Tune requires at least one range/choice param, unless backend allows empty space
  const tuneSpace = localConfig.tuning?.optuna?.space ?? {};
  const capabilities = backendContract?.capabilities ?? {};
  const allowEmptySpace = capabilities?.tune?.allow_empty_space ?? false;
  const hasSearchParam = allowEmptySpace || Object.values(tuneSpace).some(
    (p: any) => p.type === "float" || p.type === "int" || p.type === "categorical",
  );

  // Import/Export YAML via custom messages
  const [rawYaml, setRawYaml] = useState<string | null>(null);
  const [yamlExportCount, setYamlExportCount] = useState(0);

  useCustomMsg(model, useCallback((msg: any) => {
    if (msg.type === "yaml_export") {
      const blob = new Blob([msg.content], { type: "text/yaml" });
      // D-2: Try Blob URL first; fall back to data URL for Colab sandbox
      try {
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "config.yaml";
        a.click();
        URL.revokeObjectURL(url);
      } catch {
        // Colab sandbox may block Blob URL — fall back to data URL
        const reader = new FileReader();
        reader.onload = () => {
          const a = document.createElement("a");
          a.href = reader.result as string;
          a.download = "config.yaml";
          a.click();
        };
        reader.readAsDataURL(blob);
      }
      setYamlExportCount((prev) => prev + 1);
    } else if (msg.type === "raw_config") {
      setRawYaml(msg.content);
    } else if (msg.type === "raw_config_error") {
      setRawYaml(`# Error loading config\n# ${msg.message ?? "unknown error"}`);
    }
  }, []));

  return (
    <div class="lzw-config-tab">
      {/* Sub-tab bar */}
      <div class="lzw-subtabs">
        <button
          class={`lzw-subtabs__btn ${subTab === "fit" ? "lzw-subtabs__btn--active" : ""}`}
          onClick={() => setSubTab("fit")}
          type="button"
        >
          Fit
        </button>
        <button
          class={`lzw-subtabs__btn ${subTab === "tune" ? "lzw-subtabs__btn--active" : ""}`}
          onClick={() => setSubTab("tune")}
          type="button"
        >
          Tune
        </button>

        <div class="lzw-subtabs__right">
          {subTab === "fit" && (
            <button
              class="lzw-btn lzw-btn--primary"
              disabled={!canRun || status === "running"}
              onClick={() => sendAction("fit")}
              type="button"
            >
              {status === "running" ? "Running..." : "\u25B6 Fit"}
            </button>
          )}
          {subTab === "tune" && (
            <button
              class="lzw-btn lzw-btn--primary"
              disabled={!canRun || status === "running" || !hasSearchParam}
              onClick={() => sendAction("tune")}
              type="button"
            >
              {status === "running" ? "Running..." : "\u25B6 Tune"}
            </button>
          )}
        </div>
      </div>

      <div class="lzw-config-tab__body" style={status === "running" ? { pointerEvents: "none", opacity: 0.6 } : undefined}>
        {subTab === "fit" && (
          <FitSubTab
            localConfig={localConfig}
            configSchema={configSchema}
            uiSchema={uiSchema}
            task={task}
            dfInfo={dfInfo}
            handleChange={handleChange}
            handleSectionChange={handleSectionChange}
            sendAction={sendAction}
            rawYaml={rawYaml}
            setRawYaml={setRawYaml}
            yamlExportCount={yamlExportCount}
          />
        )}
        {subTab === "tune" && (
          <TuneSubTab
            localConfig={localConfig}
            configSchema={configSchema}
            uiSchema={uiSchema}
            task={task}
            dfInfo={dfInfo}
            handleChange={handleChange}
            sendAction={sendAction}
            rawYaml={rawYaml}
            setRawYaml={setRawYaml}
            yamlExportCount={yamlExportCount}
          />
        )}
      </div>
    </div>
  );
}
