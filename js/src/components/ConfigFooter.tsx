/**
 * ConfigFooter — Import/Export YAML buttons + Raw Config modal.
 * Shared between Fit and Tune sub-tabs.
 */
import { useRef, useCallback, useState, useEffect } from "preact/hooks";

interface ConfigFooterProps {
  sendAction: (type: string, payload?: Record<string, any>) => void;
  rawYaml: string | null;
  setRawYaml: (value: string | null) => void;
  yamlExportCount?: number;
}

export function ConfigFooter({ sendAction, rawYaml, setRawYaml, yamlExportCount }: ConfigFooterProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [exportLoading, setExportLoading] = useState(false);

  // B-3: Reset loading state when export response arrives
  useEffect(() => {
    if (yamlExportCount && yamlExportCount > 0) {
      setExportLoading(false);
    }
  }, [yamlExportCount]);

  const handleImportFile = useCallback(
    (e: Event) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        sendAction("import_yaml", { content: reader.result as string });
      };
      reader.readAsText(file);
      if (fileInputRef.current) fileInputRef.current.value = "";
    },
    [sendAction],
  );

  return (
    <>
      <div class="lzw-config-tab__footer">
        <input
          ref={fileInputRef}
          type="file"
          accept=".yaml,.yml,.json"
          style="display:none"
          onChange={handleImportFile}
        />
        <button class="lzw-btn" type="button" onClick={() => fileInputRef.current?.click()}>
          Import YAML
        </button>
        <button
          class="lzw-btn"
          type="button"
          disabled={exportLoading}
          onClick={() => {
            setExportLoading(true);
            sendAction("export_yaml");
          }}
        >
          {exportLoading ? "Exporting..." : "Export YAML"}
        </button>
        <button class="lzw-btn" type="button" onClick={() => sendAction("raw_config")}>
          Raw Config
        </button>
      </div>
      {rawYaml !== null && (
        <div class="lzw-modal-overlay" onClick={() => setRawYaml(null)}>
          <div class="lzw-modal" onClick={(e) => e.stopPropagation()}>
            <div class="lzw-modal__header">
              <span>Raw Config</span>
              <button class="lzw-btn" type="button" onClick={() => setRawYaml(null)}>
                Close
              </button>
            </div>
            <pre class="lzw-pre">{rawYaml}</pre>
          </div>
        </div>
      )}
    </>
  );
}
