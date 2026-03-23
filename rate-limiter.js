// rate-limiter.js — Token-bucket rate limiter for Gemini free tier (15 RPM)

const DEFAULT_RPM = Number(process.env.GEMINI_RPM_LIMIT || 15);
const DEFAULT_RETRY_LIMIT = 3;

export function createRateLimiter({
  rpm = DEFAULT_RPM,
  retryLimit = DEFAULT_RETRY_LIMIT,
} = {}) {
  const intervalMs = (60 * 1000) / rpm;
  let lastCall = 0;
  let pendingCount = 0;

  async function waitForSlot() {
    pendingCount += 1;
    const now = Date.now();
    const elapsed = now - lastCall;
    const waitMs = Math.max(0, intervalMs - elapsed) + (pendingCount - 1) * intervalMs;
    if (waitMs > 0) {
      await new Promise((resolve) => setTimeout(resolve, waitMs));
    }
    lastCall = Date.now();
    pendingCount -= 1;
  }

  async function call(fn) {
    let lastError;
    for (let attempt = 1; attempt <= retryLimit; attempt += 1) {
      await waitForSlot();
      try {
        return await fn();
      } catch (err) {
        lastError = err;
        const isRateLimit =
          err.status === 429
          || err.message?.includes("429")
          || err.message?.toLowerCase().includes("rate limit")
          || err.message?.toLowerCase().includes("quota");
        if (isRateLimit && attempt < retryLimit) {
          const backoff = Math.min(60_000, 2 ** attempt * 2000);
          console.warn(
            `[RateLimiter] 429 hit, backing off ${(backoff / 1000).toFixed(0)}s (attempt ${attempt}/${retryLimit})`
          );
          await new Promise((resolve) => setTimeout(resolve, backoff));
          continue;
        }
        throw err;
      }
    }
    throw lastError;
  }

  return { call, waitForSlot };
}
