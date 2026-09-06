/**
 * pages/SchemeDetailPage.jsx — Official Scheme Information Lookup
 *
 * Fetches GET /api/schemes/<scheme_id> and displays verified metadata,
 * statutory rules, and official URLs from the backend's verified dataset.
 */
import { useState, useEffect } from "react";
import { useParams, Link } from "react-router-dom";
import { getSchemeDetails } from "../api";
import LoadingSpinner from "../components/LoadingSpinner";
import ErrorAlert from "../components/ErrorAlert";

const RULE_TYPE_LABELS = {
  hard_filter: { label: "Hard Filter", bg: "bg-red-100 text-red-700" },
  preference: { label: "Preference", bg: "bg-blue-100 text-blue-700" },
  domain_match: { label: "Domain", bg: "bg-purple-100 text-purple-700" },
};

function RuleRow({ rule }) {
  const rt = RULE_TYPE_LABELS[rule.rule_type] || { label: rule.rule_type, bg: "bg-gray-100 text-gray-700" };
  return (
    <tr className="border-t border-gray-100 hover:bg-gray-50 text-sm">
      <td className="py-2 px-3 font-mono text-xs text-gray-500">{rule.rule_id}</td>
      <td className="py-2 px-3 font-medium text-gray-800">{rule.field}</td>
      <td className="py-2 px-3 text-gray-600">{rule.operator}</td>
      <td className="py-2 px-3 text-gray-600">
        {rule.expected_value !== undefined && rule.expected_value !== null
          ? String(rule.expected_value)
          : "—"}
      </td>
      <td className="py-2 px-3">
        <span className={`px-2 py-0.5 rounded text-xs font-medium ${rt.bg}`}>{rt.label}</span>
      </td>
      <td className="py-2 px-3 text-center">
        {rule.required ? (
          <span className="text-red-500 font-bold">✓</span>
        ) : (
          <span className="text-gray-400">–</span>
        )}
      </td>
    </tr>
  );
}

export default function SchemeDetailPage() {
  const { schemeId } = useParams();
  const [scheme, setScheme] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!schemeId) return;
    setLoading(true);
    setError(null);
    getSchemeDetails(schemeId)
      .then(setScheme)
      .catch(setError)
      .finally(() => setLoading(false));
  }, [schemeId]);

  return (
    <div className="max-w-4xl mx-auto px-4 py-10">
      <Link to="/recommend" className="text-sm text-indigo-600 hover:underline mb-6 inline-block">
        ← Back to Recommendations
      </Link>

      {loading && <LoadingSpinner label="Loading scheme details…" />}
      {error && <ErrorAlert error={error} />}

      {scheme && !loading && (
        <div className="space-y-6">
          {/* Header */}
          <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
            <div className="flex flex-wrap items-start gap-3 mb-3">
              <div>
                <h1 className="text-2xl font-bold text-gray-900">{scheme.scheme_name}</h1>
                <p className="text-sm text-gray-500 mt-0.5">
                  <span className="font-mono text-gray-400">{scheme.scheme_id}</span>
                  {" · "}
                  {scheme.category}
                  {" · "}
                  <span
                    className={`font-semibold ${
                      scheme.scheme_scope === "State" ? "text-indigo-600" : "text-orange-600"
                    }`}
                  >
                    {scheme.scheme_scope} Scheme
                  </span>
                </p>
              </div>
            </div>

            <div className="grid sm:grid-cols-2 gap-4 mt-4">
              <InfoItem label="Source Authority" value={scheme.source_authority || "—"} />
              <InfoItem label="Source File" value={scheme.source_filename || "—"} mono />
              <div className="sm:col-span-2">
                <p className="text-xs text-gray-500 mb-1 font-medium">Official URL</p>
                {scheme.official_url ? (
                  <a
                    href={scheme.official_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-indigo-600 text-sm hover:underline break-all"
                  >
                    {scheme.official_url} ↗
                  </a>
                ) : (
                  <span className="text-gray-400 text-sm">—</span>
                )}
              </div>
            </div>
          </div>

          {/* Rules table */}
          {scheme.rules && scheme.rules.length > 0 && (
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
              <div className="px-6 py-4 border-b border-gray-100">
                <h2 className="text-base font-bold text-gray-800">
                  📋 Statutory Eligibility Rules ({scheme.rules.length})
                </h2>
                <p className="text-xs text-gray-400 mt-0.5">
                  From verified official government source dataset.
                </p>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50">
                    <tr className="text-left text-xs text-gray-500">
                      <th className="py-2 px-3 font-medium">Rule ID</th>
                      <th className="py-2 px-3 font-medium">Field</th>
                      <th className="py-2 px-3 font-medium">Operator</th>
                      <th className="py-2 px-3 font-medium">Expected</th>
                      <th className="py-2 px-3 font-medium">Type</th>
                      <th className="py-2 px-3 font-medium text-center">Required</th>
                    </tr>
                  </thead>
                  <tbody>
                    {scheme.rules.map((rule) => (
                      <RuleRow key={rule.rule_id} rule={rule} />
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Unstructured criteria */}
          {scheme.unstructured_criteria && scheme.unstructured_criteria.length > 0 && (
            <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
              <h2 className="text-base font-bold text-gray-800 mb-3">
                📄 Additional Criteria ({scheme.unstructured_criteria.length})
              </h2>
              <ul className="space-y-2">
                {scheme.unstructured_criteria.map((c, i) => (
                  <li key={i} className="text-sm text-gray-700 flex gap-2">
                    <span className="text-gray-400 mt-0.5">•</span>
                    <span>{c.description || c.text || JSON.stringify(c)}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function InfoItem({ label, value, mono = false }) {
  return (
    <div>
      <p className="text-xs text-gray-500 mb-0.5 font-medium">{label}</p>
      <p className={`text-sm text-gray-800 ${mono ? "font-mono" : ""}`}>{value}</p>
    </div>
  );
}
