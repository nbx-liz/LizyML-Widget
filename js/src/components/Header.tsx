/** Header — backend badge + status indicator. */

interface HeaderProps {
  backendInfo: { name?: string; version?: string };
  status: string;
}

const STATUS_LABELS: Record<string, { label: string; cls: string }> = {
  idle: { label: "Idle", cls: "lzw-badge--muted" },
  data_loaded: { label: "Data Loaded", cls: "lzw-badge--info" },
  running: { label: "Running", cls: "lzw-badge--running" },
  completed: { label: "Completed", cls: "lzw-badge--success" },
  failed: { label: "Failed", cls: "lzw-badge--error" },
};

export function Header({ backendInfo, status }: HeaderProps) {
  const s = STATUS_LABELS[status] ?? STATUS_LABELS.idle;
  const backend = backendInfo.name
    ? `${backendInfo.name} v${backendInfo.version}`
    : "";

  return (
    <div class="lzw-header">
      <span class="lzw-header__title">LizyML Widget</span>
      <span class="lzw-header__right">
        {backend && <span class="lzw-badge lzw-badge--muted">{backend}</span>}
        <span class={`lzw-badge ${s.cls}`}>{s.label}</span>
      </span>
    </div>
  );
}
