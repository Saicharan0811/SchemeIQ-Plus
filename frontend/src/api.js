/**
 * src/api.js — SchemeIQ+ centralized API client
 *
 * All fetch calls go through this module. The API base URL is
 * configurable via the VITE_API_BASE_URL environment variable.
 *
 * INVARIANTS (mirrored from backend):
 * - Frontend NEVER decides eligibility.
 * - Frontend NEVER calculates XGBoost scores.
 * - Frontend NEVER fabricates RAG answers.
 * - Backend results are displayed exactly as received.
 */

const rawBase = import.meta.env.VITE_API_BASE_URL;
const API_BASE =
  typeof rawBase === "string" && rawBase.trim().length > 0
    ? rawBase.trim().replace(/\/+$/, "")
    : "http://127.0.0.1:5000";

async function apiRequest(path, options = {}) {
  const cleanPath = path.startsWith("/") ? path : `/${path}`;
  const url = `${API_BASE}${cleanPath}`;
  let response;
  try {
    response = await fetch(url, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
  } catch (netErr) {
    const err = new Error(
      `Unable to connect to SchemeIQ+ API server at ${API_BASE}. Please ensure the backend server is running.`
    );
    err.code = "BACKEND_UNAVAILABLE";
    err.details = [];
    throw err;
  }

  const data = await response.json().catch(() => ({
    error: `HTTP ${response.status} — non-JSON response`,
    code: "PARSE_ERROR",
  }));

  if (!response.ok) {
    const err = new Error(data.error || `HTTP ${response.status}`);
    err.code = data.code || "HTTP_ERROR";
    err.details = data.details || [];
    err.status = response.status;
    throw err;
  }

  return data;
}

/** GET /health */
export async function getHealth() {
  return apiRequest("/health");
}

/**
 * POST /api/recommend
 * @param {object} profile - UserProfile fields (no prohibited PII)
 * @param {object} options - { top_k, query, include_rag_explanation, enable_xgboost }
 */
export async function getRecommendations(profile, options = {}) {
  return apiRequest("/api/recommend", {
    method: "POST",
    body: JSON.stringify({
      profile,
      top_k: options.top_k ?? 5,
      query: options.query || undefined,
      include_rag_explanation: options.include_rag_explanation ?? false,
      enable_xgboost: options.enable_xgboost ?? false,
    }),
  });
}

/**
 * GET /api/schemes/<scheme_id>
 * @param {string} schemeId
 */
export async function getSchemeDetails(schemeId) {
  return apiRequest(`/api/schemes/${encodeURIComponent(schemeId)}`);
}

/**
 * POST /api/ask
 * @param {string} query
 * @param {number} top_k
 */
export async function askQuestion(query, top_k = 5) {
  return apiRequest("/api/ask", {
    method: "POST",
    body: JSON.stringify({ query, top_k }),
  });
}
