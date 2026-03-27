/**
 * Mock implementation of the anywidget model interface.
 *
 * Provides get/set/on/off/send with an EventEmitter pattern,
 * simulating the traitlet sync and msg:custom channels that
 * anywidget uses between Python and JS.
 */

type Handler = (...args: any[]) => void;

export class MockModel {
  private _data: Record<string, any> = {};
  private _listeners: Map<string, Set<Handler>> = new Map();
  private _sentMessages: any[] = [];

  constructor(initialData: Record<string, any> = {}) {
    this._data = { ...initialData };
  }

  /** Read a traitlet value. */
  get(name: string): any {
    return this._data[name];
  }

  /** Write a traitlet value and fire change event. */
  set(name: string, value: any): void {
    this._data[name] = value;
    this._emit(`change:${name}`);
  }

  /** Register an event listener. */
  on(event: string, handler: Handler): void {
    if (!this._listeners.has(event)) {
      this._listeners.set(event, new Set());
    }
    this._listeners.get(event)!.add(handler);
  }

  /** Remove an event listener. */
  off(event: string, handler: Handler): void {
    this._listeners.get(event)?.delete(handler);
  }

  /** Send a msg:custom message to Python (captured for assertions). */
  send(msg: any): void {
    this._sentMessages.push(msg);
  }

  save_changes(): void {
    // no-op in tests
  }

  // ── Test helpers ─────────────────────────────────────────

  /** Simulate a traitlet change from Python side. */
  simulateTraitletChange(name: string, value: any): void {
    this._data[name] = value;
    this._emit(`change:${name}`);
  }

  /** Simulate a msg:custom message from Python side. */
  simulateCustomMessage(msg: any, buffers?: ArrayBuffer[]): void {
    this._emit("msg:custom", msg, buffers);
  }

  /** Get all messages sent via model.send(). */
  get sentMessages(): any[] {
    return this._sentMessages;
  }

  /** Clear sent messages. */
  clearSentMessages(): void {
    this._sentMessages = [];
  }

  /** Get count of listeners for an event. */
  listenerCount(event: string): number {
    return this._listeners.get(event)?.size ?? 0;
  }

  private _emit(event: string, ...args: any[]): void {
    const handlers = this._listeners.get(event);
    if (handlers) {
      for (const handler of handlers) {
        handler(...args);
      }
    }
  }
}

/** Create a MockModel with default widget traitlets. */
export function createMockModel(overrides: Record<string, any> = {}): MockModel {
  return new MockModel({
    status: "idle",
    job_type: "",
    job_index: 0,
    progress: {},
    elapsed_sec: 0,
    fit_summary: {},
    tune_summary: {},
    available_plots: [],
    error: {},
    config: {},
    df_info: {},
    backend_info: {},
    backend_contract: {},
    inference_result: {},
    ...overrides,
  });
}
