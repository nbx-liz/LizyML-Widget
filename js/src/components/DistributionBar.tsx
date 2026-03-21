/** DistributionBar — horizontal bar chart for value distribution. */

interface DistValue {
  value: string;
  count: number;
}

interface DistributionBarProps {
  values: DistValue[];
}

export function DistributionBar({ values }: DistributionBarProps) {
  if (values.length === 0) return null;
  const maxCount = Math.max(...values.map((v) => v.count), 1);

  return (
    <div>
      {values.map((v) => {
        const pct = Math.round((v.count / maxCount) * 100);
        return (
          <div key={v.value} class="lzw-dist-row">
            <span>{v.value}</span>
            <div class="lzw-dist-bar-bg">
              <div class="lzw-dist-bar" style={{ width: `${pct}%` }} />
            </div>
            <span class="lzw-dist-count">{v.count.toLocaleString()}</span>
          </div>
        );
      })}
    </div>
  );
}
