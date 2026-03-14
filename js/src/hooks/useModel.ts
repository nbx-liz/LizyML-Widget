/**
 * useModel — Preact hooks for anywidget model (traitlet) binding.
 *
 * Provides reactive access to Python→JS traitlets and a sendAction
 * helper for JS→Python communication via the `action` traitlet.
 */
import { useState, useEffect, useCallback, useRef } from "preact/hooks";

/** Read a single traitlet reactively. Re-renders on change. */
export function useTraitlet<T>(model: any, name: string): T {
  const [value, setValue] = useState<T>(() => model.get(name));

  useEffect(() => {
    const handler = () => setValue(model.get(name));
    model.on(`change:${name}`, handler);
    return () => model.off(`change:${name}`, handler);
  }, [model, name]);

  return value;
}

/** Send an action to the Python side via the `action` traitlet. */
export function useSendAction(model: any) {
  const tsRef = useRef(0);
  return useCallback(
    (type: string, payload: Record<string, any> = {}) => {
      tsRef.current += 1;
      model.set("action", { type, payload, _ts: tsRef.current });
      model.save_changes();
    },
    [model],
  );
}

/** Subscribe to custom messages from Python (widget.send()). */
export function useCustomMsg(
  model: any,
  handler: (msg: any) => void,
) {
  const handlerRef = useRef(handler);
  handlerRef.current = handler;

  useEffect(() => {
    const cb = (msg: any) => handlerRef.current(msg);
    model.on("msg:custom", cb);
    return () => model.off("msg:custom", cb);
  }, [model]);
}
