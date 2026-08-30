// Display-only scheduling. Never gates sim_state or policy RGB-D capture.
export class PresentationBudget {
  constructor() {
    this.lastFrameAt = null;
    this.nextFrameAt = null;
    this.targetFps = 60;
    this.frames = [];
    this.skipped = 0;
    this.hidden = false;
  }

  shouldRender(now, { hidden = false, busy = false, idle = false, interacting = false } = {}) {
    this.hidden = hidden;
    const target = busy ? 30 : interacting || !idle ? 60 : 30;
    if (target !== this.targetFps) this.nextFrameAt = null;
    this.targetFps = target;
    if (hidden) {
      this.lastFrameAt = null;
      this.nextFrameAt = null;
      this.skipped += 1;
      return false;
    }
    if (this.nextFrameAt !== null && now < this.nextFrameAt - 1) {
      this.skipped += 1;
      return false;
    }
    this.lastFrameAt = now;
    const interval = 1000 / this.targetFps;
    // Preserve the fractional deadline on non-60Hz displays. Reset after a
    // long stall instead of creating a burst of catch-up rendering.
    this.nextFrameAt = this.nextFrameAt === null || now - this.nextFrameAt > interval
      ? now + interval : this.nextFrameAt + interval;
    return true;
  }

  record(now, submissionMs) {
    this.frames.push({ at: now, ms: submissionMs });
    if (this.frames.length > 300) this.frames.shift();
  }

  statistics(now) {
    const recent = this.frames.filter((frame) => now - frame.at <= 5000);
    const costs = recent.map((frame) => frame.ms).sort((a, b) => a - b);
    const elapsed = recent.length > 1 ? recent.at(-1).at - recent[0].at : 0;
    return {
      target_fps: this.targetFps,
      fps: elapsed > 0 ? (recent.length - 1) * 1000 / elapsed : 0,
      render_submission_p95_ms: costs.length ? costs[Math.ceil(costs.length * 0.95) - 1] : null,
      frames_in_window: recent.length,
      skipped_frames: this.skipped,
      hidden: this.hidden,
    };
  }
}
