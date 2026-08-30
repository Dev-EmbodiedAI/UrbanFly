const percentile = (values, fraction) => {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted.length ? sorted[Math.ceil(sorted.length * fraction) - 1] : null;
};

export class RenderBenchmark {
  constructor({ durationMs = 10000, warmupMs = 1000 } = {}) {
    this.durationMs = durationMs;
    this.warmupMs = warmupMs;
    this.active = false;
    this.samples = [];
  }

  start(now) {
    this.startAt = now;
    this.samples = [];
    this.active = true;
  }

  record(now, submissionMs, drawCalls, triangles) {
    if (!this.active) return null;
    if (now - this.startAt >= this.warmupMs && this.samples.length < 2400) {
      this.samples.push({ at: now, cpu: submissionMs, drawCalls, triangles });
    }
    if (now - this.startAt < this.warmupMs + this.durationMs) return null;
    this.active = false;
    const elapsed = this.samples.length > 1 ? this.samples.at(-1).at - this.samples[0].at : 0;
    const intervals = this.samples.slice(1).map((item, i) => item.at - this.samples[i].at);
    return {
      schema: 'urbanfly-display-benchmark-v1',
      warmup_ms: this.warmupMs, duration_ms: elapsed, frames: this.samples.length,
      fps: elapsed > 0 ? (this.samples.length - 1) * 1000 / elapsed : 0,
      frame_interval_p95_ms: percentile(intervals, 0.95),
      render_cpu_p95_ms: percentile(this.samples.map((item) => item.cpu), 0.95),
      draw_calls_median: percentile(this.samples.map((item) => item.drawCalls), 0.5),
      triangles_median: percentile(this.samples.map((item) => item.triangles), 0.5),
    };
  }

  cancel() { this.active = false; }
}
