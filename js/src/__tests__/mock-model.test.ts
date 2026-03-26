import { describe, it, expect } from "vitest";
import { MockModel, createMockModel } from "./mock-model";

describe("MockModel", () => {
  it("get/set works", () => {
    const m = new MockModel({ status: "idle" });
    expect(m.get("status")).toBe("idle");
    m.set("status", "running");
    expect(m.get("status")).toBe("running");
  });

  it("fires change events on set", () => {
    const m = new MockModel({ status: "idle" });
    let received = false;
    m.on("change:status", () => { received = true; });
    m.set("status", "running");
    expect(received).toBe(true);
  });

  it("simulateCustomMessage fires msg:custom", () => {
    const m = new MockModel();
    const msgs: any[] = [];
    m.on("msg:custom", (msg: any) => msgs.push(msg));
    m.simulateCustomMessage({ type: "job_state", status: "running" });
    expect(msgs).toHaveLength(1);
    expect(msgs[0].status).toBe("running");
  });

  it("send captures messages", () => {
    const m = new MockModel();
    m.send({ type: "poll" });
    expect(m.sentMessages).toHaveLength(1);
    expect(m.sentMessages[0].type).toBe("poll");
  });

  it("createMockModel has default traitlets", () => {
    const m = createMockModel();
    expect(m.get("status")).toBe("idle");
    expect(m.get("job_index")).toBe(0);
  });
});
