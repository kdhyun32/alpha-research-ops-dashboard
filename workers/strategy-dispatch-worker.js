const ALLOWED_ORIGINS = new Set([
  "https://kdhyun32.github.io",
  "https://kdhyun32.github.io/alpha-research-ops-dashboard",
  "https://alpha-research-ops-dashboard.pages.dev"
]);

function cors(origin) {
  const allowed = ALLOWED_ORIGINS.has(origin) || origin === "https://kdhyun32.github.io" ? origin : "https://kdhyun32.github.io";
  return {
    "access-control-allow-origin": allowed,
    "access-control-allow-methods": "POST, OPTIONS",
    "access-control-allow-headers": "content-type, x-alpha-execution-token, authorization",
    "content-type": "application/json"
  };
}

function json(body, status, origin) {
  return new Response(JSON.stringify(body), { status, headers: cors(origin) });
}

function tokenFrom(request, body) {
  const auth = request.headers.get("authorization") || "";
  if (auth.toLowerCase().startsWith("bearer ")) return auth.slice(7).trim();
  return request.headers.get("x-alpha-execution-token") || body.execution_token || "";
}

function validatePayload(body) {
  const batch = body.strategy_batch;
  if (!batch || !Array.isArray(batch.strategies)) return "strategy_batch.strategies 배열이 필요합니다.";
  if (batch.strategies.length < 1) return "실행할 전략이 없습니다.";
  if (batch.strategies.length > 10) return "한 번에 최대 10개 전략만 실행할 수 있습니다.";
  const bytes = new TextEncoder().encode(JSON.stringify(batch)).length;
  if (bytes > 120000) return "요청 JSON이 너무 큽니다.";
  return "";
}

export default {
  async fetch(request, env) {
    const origin = request.headers.get("origin") || "";
    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: cors(origin) });
    if (request.method !== "POST") return json({ ok: false, error: "POST만 허용됩니다." }, 405, origin);
    let body;
    try {
      body = await request.json();
    } catch {
      return json({ ok: false, error: "JSON 요청만 허용됩니다." }, 400, origin);
    }
    if (!env.EXECUTION_TOKEN || tokenFrom(request, body) !== env.EXECUTION_TOKEN) {
      return json({ ok: false, error: "실행 토큰이 올바르지 않습니다." }, 401, origin);
    }
    const mode = body.mode === "backtest" ? "backtest" : "validate";
    const payloadError = validatePayload(body);
    if (payloadError) return json({ ok: false, error: payloadError }, 400, origin);
    const githubToken = env.GITHUB_TOKEN;
    if (!githubToken) return json({ ok: false, error: "GitHub dispatch token이 설정되지 않았습니다." }, 500, origin);
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
          request_guard: { max_strategies: 10, max_payload_bytes: 120000 }
        }
      })
    });
    if (!response.ok) {
      const text = await response.text();
      return json({ ok: false, error: "GitHub Actions 실행 요청에 실패했습니다.", github_status: response.status, github_response: text.slice(0, 500) }, 502, origin);
    }
    return json({ ok: true, github_event_type: eventType, result_path: "external_strategy_results/latest.json" }, 202, origin);
  }
};
