/**
 * ConfigTab — Fit sub-tab with DynForm + Tune sub-tab with SearchSpace.
 * Form changes are debounced (300ms) before syncing to Python.
 */
import { useState, useRef, useCallback, useEffect } from "preact/hooks";
import { Accordion } from "../components/Accordion";
import { DynForm } from "../components/DynForm";
import { SearchSpace } from "../components/SearchSpace";

type SubTab = "fit" | "tune";

interface ConfigTabProps {
  configSchema: Record<string, any>;
  config: Record<string, any>;
  status: string;
  sendAction: (type: string, payload?: Record<string, any>) => void;
}

export function ConfigTab({
  configSchema,
  config,
  status,
  sendAction,
}: ConfigTabProps) {
  const [subTab, setSubTab] = useState<SubTab>("fit");
  const [localConfig, setLocalConfig] = useState<Record<string, any>>(config);
  const [searchSpace, setSearchSpace] = useState<Record<string, any>>({});
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Sync from Python → local when Python config changes externally
  useEffect(() => {
    setLocalConfig(config);
  }, [config]);

  // Debounced send to Python
  const handleChange = useCallback(
    (newConfig: Record<string, any>) => {
      setLocalConfig(newConfig);
      if (timerRef.current) clearTimeout(timerRef.current);
      timerRef.current = setTimeout(() => {
        sendAction("update_config", { config: newConfig });
      }, 300);
    },
    [sendAction],
  );

  const canRun =
    status === "data_loaded" || status === "completed" || status === "failed";

  // Tune requires at least one range/choice param
  const hasSearchParam = Object.values(searchSpace).some(
    (p: any) => p.mode === "range" || p.mode === "choice",
  );

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

        {/* Action button (sticky right) */}
        <div class="lzw-subtabs__right">
          {subTab === "fit" && (
            <button
              class="lzw-btn lzw-btn--primary"
              disabled={!canRun || status === "running"}
              onClick={() => sendAction("fit")}
              type="button"
            >
              {status === "running" ? "Running..." : "Fit"}
            </button>
          )}
          {subTab === "tune" && (
            <button
              class="lzw-btn lzw-btn--primary"
              disabled={!canRun || status === "running" || !hasSearchParam}
              onClick={() => sendAction("tune")}
              type="button"
            >
              {status === "running" ? "Running..." : "Tune"}
            </button>
          )}
        </div>
      </div>

      {/* Sub-tab content */}
      <div class="lzw-config-tab__body">
        {subTab === "fit" && (
          <DynForm
            schema={configSchema}
            value={localConfig}
            onChange={handleChange}
          />
        )}
        {subTab === "tune" && (
          <div class="lzw-tune-tab">
            <Accordion title="Tune Settings">
              <div class="lzw-form-row">
                <label class="lzw-label">n_trials</label>
                <input
                  class="lzw-input lzw-input--sm"
                  type="number"
                  min={1}
                  value={localConfig.tuning?.n_trials ?? 100}
                  onChange={(e) =>
                    handleChange({
                      ...localConfig,
                      tuning: {
                        ...(localConfig.tuning ?? {}),
                        n_trials: parseInt(
                          (e.target as HTMLInputElement).value,
                        ) || 100,
                      },
                    })
                  }
                />
              </div>
              <div class="lzw-form-row">
                <label class="lzw-label">timeout (sec)</label>
                <input
                  class="lzw-input lzw-input--sm"
                  type="number"
                  min={0}
                  value={localConfig.tuning?.timeout ?? 600}
                  onChange={(e) =>
                    handleChange({
                      ...localConfig,
                      tuning: {
                        ...(localConfig.tuning ?? {}),
                        timeout: parseInt(
                          (e.target as HTMLInputElement).value,
                        ) || 600,
                      },
                    })
                  }
                />
              </div>
            </Accordion>

            <Accordion title="Search Space">
              <SearchSpace
                schema={configSchema}
                value={searchSpace}
                onChange={setSearchSpace}
              />
            </Accordion>
          </div>
        )}
      </div>
    </div>
  );
}
