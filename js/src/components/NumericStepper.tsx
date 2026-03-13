/** NumericStepper — large -/+ buttons with direct numeric input. */

interface NumericStepperProps {
  value: number | undefined;
  step?: number | string;
  min?: number;
  max?: number;
  placeholder?: string;
  class?: string;
  onChange: (value: number | undefined) => void;
}

export function NumericStepper({
  value,
  step = 1,
  min,
  max,
  placeholder,
  class: cls,
  onChange,
}: NumericStepperProps) {
  const numStep = typeof step === "string" ? 1 : step;

  const decrement = () => {
    const next = (value ?? 0) - numStep;
    if (min !== undefined && next < min) return;
    onChange(next);
  };

  const increment = () => {
    const next = (value ?? 0) + numStep;
    if (max !== undefined && next > max) return;
    onChange(next);
  };

  return (
    <div class={`lzw-stepper ${cls ?? ""}`}>
      <button type="button" class="lzw-stepper__btn" onClick={decrement} aria-label="Decrease">
        −
      </button>
      <input
        class="lzw-stepper__input"
        type="number"
        step={step}
        min={min}
        max={max}
        placeholder={placeholder}
        value={value ?? ""}
        onChange={(e) => {
          const raw = (e.target as HTMLInputElement).value;
          onChange(raw === "" ? undefined : parseFloat(raw));
        }}
      />
      <button type="button" class="lzw-stepper__btn" onClick={increment} aria-label="Increase">
        +
      </button>
    </div>
  );
}
