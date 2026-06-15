const ALLOWED_ORIGINS = new Set([
  "https://kdhyun32.github.io",
  "https://kdhyun32.github.io/alpha-research-ops-dashboard",
  "https://alpha-research-ops-dashboard.pages.dev"
]);

const MAX_STRATEGIES = 10;
const MAX_PAYLOAD_BYTES = 120000;

function cors(origin) {
  const allowed = ALLOWED_ORIGINS.has(origin) ? origin : "https://kdhyun32.github.io";
  return {
    "access-control-allow-origin": allowed,
    "access-control-allow-methods": "POST, OPTIONS",
    "access-control-allow-headers": "content-type",
    "content-type": "application/json"
  };
}

function json(body, status, origin) {
  return new Response(JSON.stringify(body), { status, headers: cors(origin) });
}

function validatePayload(body) {
  if (body.mode !== "validate" && body.mode !== "backtest") {
    return "mode must be validate or backtest.";
  }

  const batch = body.strategy_batch;
  if (!batch || !Array.isArray(batch.strategies)) {
    return "strategy_batch.strategies array is required.";
  }
  if (batch.strategies.length < 1) {
    return "No strategies were provided.";
  }
  if (batch.strategies.length > MAX_STRATEGIES) {
    return `A maximum of ${MAX_STRATEGIES} strategies is allowed per request.`;
  }

  const bytes = new TextEncoder().encode(JSON.stringify(batch)).length;
  if (bytes > MAX_PAYLOAD_BYTES) {
    return `Request JSON is too large. Maximum payload bytes: ${MAX_PAYLOAD_BYTES}.`;
  }

  return "";
}

export default {
  async fetch(request, env) {
    const origin = request.headers.get("origin") || "";
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: cors(origin) });
    }
    if (request.method !== "POST") {
      return json({ ok: false, error: "Only POST is allowed." }, 405, origin);
    }

    let body;
    try {
      body = await request.json();
    } catch {
      return json({ ok: false, error: "Valid JSON is required." }, 400, origin);
    }

    const payloadError = validatePayload(body);
    if (payloadError) return json({ ok: false, error: payloadError }, 400, origin);

    const githubToken = env.GITHUB_TOKEN || env.GITHUB_DISPATCH_TOKEN;
    if (!githubToken) {
      return json({ ok: false, error: "GitHub dispatch token is not configured." }, 500, origin);
    }

    const mode = body.mode;
    const eventType = mode === "backtest" ? "external_strategy_backtest" : "external_strategy_validate";
    const response = await fetch("https://api.github.com/repos/kdhyun32/alpha-research-ops-dashboard/dispatches", {
      method: "POST",
      headers: {
        "accept": "application/vnd.github+json",
        "authorization": `Bearer ${githubToken}`,
        "content-type": "application/json",
        "user-agent": "alpha-research-ops-dashboard-dispatch-worker",
        "x-github-api-version": "2022-11-28"
      },
      body: JSON.stringify({
        event_type: eventType,
        client_payload: {
          strategy_batch: body.strategy_batch,
          requested_at: new Date().toISOString(),
          request_guard: { max_strategies: MAX_STRATEGIES, max_payload_bytes: MAX_PAYLOAD_BYTES }
        }
      })
    });

    if (!response.ok) {
      const text = await response.text();
      return json({
        ok: false,
        error: "GitHub Actions dispatch failed.",
        github_status: response.status,
        github_response: text.slice(0, 500)
      }, 502, origin);
    }

    return json({ ok: true, github_event_type: eventType, result_path: "external_strategy_results/latest.json" }, 202, origin);
  }
};
