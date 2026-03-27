/**
 * useModel — Preact hooks for anywidget model (traitlet) binding.
 *
 * Provides reactive access to Python→JS traitlets and a sendAction
 * helper for JS→Python communication via msg:custom.
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

/** Send an action to the Python side via msg:custom.
 *
 * P-023: Switched from traitlet sync (model.set + save_changes) to
 * model.send() because Dict traitlet JS→Python sync breaks on
 * Google Colab (ipywidgets 7.x).  msg:custom is handled by
 * _handle_custom_msg on the Python main thread.
 */
export function useSendAction(model: any) {
  return useCallback(
    (type: string, payload: Record<string, any> = {}) => {
      model.send({ type: "action", action_type: type, payload });
    },
    [model],
  );
}

/** Subscribe to custom messages from Python (widget.send()). */
export function useCustomMsg(
  model: any,
  handler: (msg: any, buffers?: ArrayBuffer[]) => void,
) {
  const handlerRef = useRef(handler);
  handlerRef.current = handler;

  useEffect(() => {
    const cb = (msg: any, buffers?: ArrayBuffer[]) => handlerRef.current(msg, buffers);
    model.on("msg:custom", cb);
    return () => model.off("msg:custom", cb);
  }, [model]);
}
