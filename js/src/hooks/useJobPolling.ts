/**
 * useJobPolling — polls Python for job state via msg:custom.
 *
 * On environments where BG-thread traitlet writes don't reach JS (Google Colab),
 * this hook provides a fallback: JS sends {type:"poll"} at 1s intervals via
 * model.send(), and Python's main thread replies with the current job state via
 * self.send(). A 100ms client-side timer interpolates elapsed_sec for smooth display.
 *
 * Polling activates only when traitlet updates stall (Colab detection):
 * if elapsed_sec traitlet hasn't changed within 2s of status becoming "running",
 * polling kicks in as a fallback.
 */
import { useState, useEffect, useRef, useCallback } from "preact/hooks";

export interface JobState {
  status: string;
  progress: Record<string, any>;
  elapsed_sec: number;
  job_type: string;
  job_index: number;
  error: Record<string, any>;
  fit_summary?: Record<string, any>;
  tune_summary?: Record<string, any>;
  available_plots?: string[];
}

const POLL_INTERVAL_MS = 1000;
const INTERPOLATION_INTERVAL_MS = 100;
/** Wait this long before deciding traitlets are stalled and starting polling. */
const STALL_DETECT_MS = 2000;

export function useJobPolling(
  model: any,
  traitletStatus: string,
  traitletElapsed: number,
): JobState | null {
  const [polled, setPolled] = useState<JobState | null>(null);
  const lastPollTime = useRef<number>(0);
  const lastElapsed = useRef<number>(0);
  const pollingActive = useRef<boolean>(false);
  const needsPolling = useRef<boolean>(false);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const interpTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Handle poll responses from Python
  const handleMsg = useCallback((msg: any) => {
    const data = msg.content?.data ?? msg;
    if (data.type !== "job_state") return;

    lastPollTime.current = performance.now();
    lastElapsed.current = data.elapsed_sec;

    pollingActive.current = data.status === "running";

    setPolled({
      status: data.status,
      progress: data.progress ?? {},
      elapsed_sec: data.elapsed_sec,
      job_type: data.job_type ?? "",
      job_index: data.job_index ?? 0,
      error: data.error ?? {},
      fit_summary: data.fit_summary,
      tune_summary: data.tune_summary,
      available_plots: data.available_plots,
    });
  }, []);

  /** Stop polling and clean up timers/listeners. */
  const stopPolling = useCallback(() => {
    if (pollTimerRef.current !== null) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
    if (interpTimerRef.current !== null) {
      clearInterval(interpTimerRef.current);
      interpTimerRef.current = null;
    }
    if (needsPolling.current) {
      model.off("msg:custom", handleMsg);
    }
    pollingActive.current = false;
    needsPolling.current = false;
  }, [model, handleMsg]);

  /** Start polling: register listener + timers. */
  const startPolling = useCallback(() => {
    needsPolling.current = true;
    pollingActive.current = true;

    model.on("msg:custom", handleMsg);
    model.send({ type: "poll" });

    pollTimerRef.current = setInterval(() => {
      if (!pollingActive.current) return;
      model.send({ type: "poll" });
    }, POLL_INTERVAL_MS);

    interpTimerRef.current = setInterval(() => {
      if (lastPollTime.current === 0 || !pollingActive.current) return;
      const delta = (performance.now() - lastPollTime.current) / 1000;
      const interpolated =
        Math.round((lastElapsed.current + delta) * 10) / 10;
      setPolled((prev) =>
        prev ? { ...prev, elapsed_sec: interpolated } : prev,
      );
    }, INTERPOLATION_INTERVAL_MS);
  }, [model, handleMsg]);

  // Detect stalled traitlets and start polling as fallback
  useEffect(() => {
    if (traitletStatus !== "running") {
      stopPolling();
      return;
    }

    // Wait STALL_DETECT_MS: if traitletElapsed is still 0, traitlets are stalled
    const stallTimer = setTimeout(() => {
      if (traitletElapsed === 0) {
        startPolling();
      }
    }, STALL_DETECT_MS);

    return () => {
      clearTimeout(stallTimer);
      stopPolling();
    };
  }, [model, traitletStatus, traitletElapsed, startPolling, stopPolling]);

  // Reset polled state on status transitions
  useEffect(() => {
    if (traitletStatus === "running") {
      lastPollTime.current = 0;
      lastElapsed.current = 0;
      setPolled(null);
    } else if (traitletStatus === "data_loaded" || traitletStatus === "idle") {
      setPolled(null);
    }
  }, [traitletStatus]);

  return polled;
}
