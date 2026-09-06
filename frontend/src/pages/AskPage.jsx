/**
 * pages/AskPage.jsx — Grounded RAG Question Answering
 *
 * Submits citizen queries to POST /api/ask and displays answers grounded
 * in official government documents with source citations.
 * Frontend never fabricates answers — displays backend results exactly.
 */
import { useState } from "react";
import { askQuestion } from "../api";
import LoadingSpinner from "../components/LoadingSpinner";
import ErrorAlert from "../components/ErrorAlert";

const GROUNDING_STATUS_CONFIG = {
  GROUNDED: {
    label: "Fully Grounded",
    icon: "✅",
    bg: "bg-green-50",
    text: "text-green-800",
    border: "border-green-200",
  },
  PARTIALLY_GROUNDED: {
    label: "Partially Grounded",
    icon: "🟡",
    bg: "bg-yellow-50",
    text: "text-yellow-800",
    border: "border-yellow-200",
  },
  NOT_GROUNDED: {
    label: "Not Grounded",
    icon: "⚠️",
    bg: "bg-red-50",
    text: "text-red-800",
    border: "border-red-200",
  },
  FALLBACK: {
    label: "Fallback Mode",
    icon: "ℹ️",
    bg: "bg-blue-50",
    text: "text-blue-800",
    border: "border-blue-200",
  },
};

const EXAMPLE_QUERIES = [
  "Who is eligible for Rythu Bharosa financial assistance?",
  "What is the income limit for Aasara pension?",
  "What documents are needed for PM Kisan?",
  "Can a widow apply for housing under PMAY?",
  "What is the benefit amount for Cheyutha scheme?",
];

export default function AskPage() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const trimmed = query.trim();
    if (trimmed.length < 5) {
      setError({ message: "Query must be at least 5 characters.", code: "QUERY_TOO_SHORT" });
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await askQuestion(trimmed, topK);
      setResult(data);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  };

  const gsCfg = result
    ? GROUNDING_STATUS_CONFIG[result.grounding_status] || {
        label: result.grounding_status,
        icon: "📄",
        bg: "bg-gray-50",
        text: "text-gray-700",
        border: "border-gray-200",
      }
    : null;

  return (
    <div className="max-w-3xl mx-auto px-4 py-10">
      <h1 className="text-3xl font-bold text-gray-900 mb-1">Ask SchemeIQ+</h1>
      <p className="text-gray-500 text-sm mb-8">
        Ask any question about government welfare schemes. Answers are generated strictly from
        verified official government documents — never fabricated.
      </p>

      {/* Query form */}
      <form onSubmit={handleSubmit} className="bg-white rounded-xl border border-gray-200 p-6 mb-6 shadow-sm">
        <label htmlFor="query" className="block text-sm font-medium text-gray-700 mb-2">
          Your Question
        </label>
        <textarea
          id="query"
          rows={3}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g. Who is eligible for Rythu Bharosa financial assistance?"
          className="w-full border border-gray-300 rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 resize-none"
        />
        <div className="flex items-center gap-4 mt-3">
          <div className="flex items-center gap-2 text-sm text-gray-600">
            <label htmlFor="topk_ask">Sources (top-k):</label>
            <input
              id="topk_ask"
              type="number"
              min={1}
              max={20}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="w-16 border border-gray-300 rounded px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500"
            />
          </div>
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="ml-auto bg-indigo-700 text-white font-semibold px-5 py-2 rounded-lg hover:bg-indigo-600 disabled:opacity-50 transition-colors"
          >
            {loading ? "Searching…" : "Ask"}
          </button>
        </div>

        {/* Example queries */}
        <div className="mt-4">
          <p className="text-xs text-gray-400 mb-2">Try an example:</p>
          <div className="flex flex-wrap gap-2">
            {EXAMPLE_QUERIES.map((q) => (
              <button
                key={q}
                type="button"
                onClick={() => setQuery(q)}
                className="text-xs px-2.5 py-1 rounded-full bg-indigo-50 text-indigo-700 hover:bg-indigo-100 border border-indigo-200 transition-colors"
              >
                {q.length > 50 ? q.slice(0, 50) + "…" : q}
              </button>
            ))}
          </div>
        </div>
      </form>

      {loading && <LoadingSpinner label="Searching official documents…" />}
      {error && <ErrorAlert error={error} onDismiss={() => setError(null)} />}

      {/* Results */}
      {result && !loading && (
        <div className="space-y-5">
          {/* Question echo */}
          <div className="bg-indigo-50 rounded-xl border border-indigo-100 p-4">
            <p className="text-xs font-semibold text-indigo-600 mb-1">Your Question</p>
            <p className="text-sm text-indigo-900 font-medium">{result.question}</p>
          </div>

          {/* Grounding badge */}
          {gsCfg && (
            <div className={`rounded-xl border px-4 py-3 flex items-center gap-2 ${gsCfg.bg} ${gsCfg.border}`}>
              <span className="text-lg">{gsCfg.icon}</span>
              <div>
                <p className={`text-sm font-semibold ${gsCfg.text}`}>
                  {gsCfg.label}
                  {result.detected_scheme_name
                    ? ` — ${result.detected_scheme_name}`
                    : ""}
                </p>
                {result.grounding_note && (
                  <p className={`text-xs mt-0.5 ${gsCfg.text} opacity-80`}>{result.grounding_note}</p>
                )}
              </div>
            </div>
          )}

          {/* Answer */}
          <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
            <h2 className="text-base font-bold text-gray-800 mb-3">📖 Answer</h2>
            <p className="text-sm text-gray-800 leading-relaxed whitespace-pre-wrap">
              {result.answer}
            </p>
          </div>

          {/* Sources */}
          {result.sources && result.sources.length > 0 && (
            <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
              <h2 className="text-base font-bold text-gray-800 mb-3">
                📚 Sources ({result.sources.length})
              </h2>
              <div className="space-y-3">
                {result.sources.map((src) => (
                  <div
                    key={src.source_id}
                    className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg border border-gray-100"
                  >
                    <div className="flex-shrink-0 w-6 h-6 rounded-full bg-indigo-100 text-indigo-700 text-xs font-bold flex items-center justify-center">
                      {src.source_id}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-semibold text-gray-800">
                        {src.document_title || src.scheme_name}
                        {src.scheme_id && (
                          <span className="ml-2 text-xs text-gray-400 font-mono">
                            {src.scheme_id}
                          </span>
                        )}
                      </p>
                      <p className="text-xs text-gray-500 mt-0.5 font-mono">{src.source_filename}</p>
                      {src.official_url && (
                        <a
                          href={src.official_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-xs text-indigo-600 hover:underline mt-0.5 block truncate"
                        >
                          {src.official_url} ↗
                        </a>
                      )}
                    </div>
                  </div>
                ))}
              </div>
              <p className="text-xs text-gray-400 mt-3">
                {result.context_chunks_used} context chunks used ·{" "}
                {result.generation_timestamp
                  ? new Date(result.generation_timestamp).toLocaleString()
                  : ""}
              </p>
            </div>
          )}

          {/* Fallback notice for no sources */}
          {(!result.sources || result.sources.length === 0) && (
            <div className="text-sm text-gray-400 bg-gray-50 rounded-lg p-4 border border-gray-100">
              ℹ️ No source citations available for this answer.
            </div>
          )}
        </div>
      )}

      {!result && !loading && !error && (
        <div className="flex flex-col items-center justify-center py-20 text-center text-gray-400">
          <div className="text-5xl mb-4">💬</div>
          <p className="font-medium text-gray-500">Ask a question about government schemes</p>
          <p className="text-sm mt-1">Answers are grounded in official documents.</p>
        </div>
      )}
    </div>
  );
}
