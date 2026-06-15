const ALLOWED_ORIGINS = new Set([
  "https://kdhyun32.github.io",
  "https://kdhyun32.github.io/alpha-research-ops-dashboard",
  "https://alpha-research-ops-dashboard.pages.dev"
]);

const MAX_STRATEGIES = 200;
const MAX_PAYLOAD_BYTES = 1500000;

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

function githubHeaders(token) {
  return {
    "accept": "application/vnd.github+json",
    "authorization": `Bearer ${token}`,
    "content-type": "application/json",
    "user-agent": "alpha-research-ops-dashboard-dispatch-worker",
    "x-github-api-version": "2022-11-28"
  };
}

function safePathPart(value) {
  return String(value || "").replace(/[^A-Za-z0-9._-]/g, "-").slice(0, 120);
}

function base64EncodeUtf8(text) {
  const bytes = new TextEncoder().encode(text);
  let binary = "";
  const chunkSize = 0x8000;
  for (let index = 0; index < bytes.length; index += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(index, index + chunkSize));
  }
  return btoa(binary);
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
    const requestId = body.request_id || `alpha-public-${Date.now()}-${crypto.randomUUID()}`;
    const strategyBatch = {
      ...body.strategy_batch,
      request_id: requestId,
      input_batch_hash: body.input_batch_hash || body.strategy_batch.input_batch_hash || "",
      input_strategy_sequence_hash: body.input_strategy_sequence_hash || body.strategy_batch.input_strategy_sequence_hash || ""
    };
    const requestPath = `external_strategy_requests/${safePathPart(requestId)}.json`;
    const requestContent = JSON.stringify(strategyBatch, null, 2) + "\n";
    const uploadResponse = await fetch(`https://api.github.com/repos/kdhyun32/alpha-research-ops-dashboard/contents/${requestPath}`, {
      method: "PUT",
      headers: githubHeaders(githubToken),
      body: JSON.stringify({
        message: `Store external strategy request ${requestId}`,
        content: base64EncodeUtf8(requestContent),
        branch: "main"
      })
    });

    if (!uploadResponse.ok) {
      const text = await uploadResponse.text();
      return json({
        ok: false,
        error: "GitHub request payload upload failed.",
        github_status: uploadResponse.status,
        github_response: text.slice(0, 500)
      }, 502, origin);
    }

    const response = await fetch("https://api.github.com/repos/kdhyun32/alpha-research-ops-dashboard/dispatches", {
      method: "POST",
      headers: githubHeaders(githubToken),
      body: JSON.stringify({
        event_type: eventType,
        client_payload: {
          request_id: requestId,
          strategy_batch_path: requestPath,
          input_batch_hash: strategyBatch.input_batch_hash,
          input_strategy_sequence_hash: strategyBatch.input_strategy_sequence_hash,
          strategy_count: strategyBatch.strategies.length,
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

    return json({
      ok: true,
      github_event_type: eventType,
      request_id: requestId,
      request_path: requestPath,
      result_path: "external_strategy_results/latest.json",
      result_index_path: "external_strategy_results/index.json"
    }, 202, origin);
  }
};
