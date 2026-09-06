/**
 * pages/RecommendPage.jsx — Citizen Profile Form + Recommendation Results
 *
 * Displays a form matching all UserProfile fields from the backend schema.
 * Submits to POST /api/recommend and shows results in-page.
 * All eligibility decisions come exclusively from the backend.
 */
import { useState } from "react";
import { Link } from "react-router-dom";
import { getRecommendations } from "../api";
import StatusBadge from "../components/StatusBadge";
import LoadingSpinner from "../components/LoadingSpinner";
import ErrorAlert from "../components/ErrorAlert";

// ─── Form field helpers ────────────────────────────────────────────────────────

function FieldGroup({ label, htmlFor, children, hint }) {
  return (
    <div>
      <label htmlFor={htmlFor} className="block text-sm font-medium text-gray-700 mb-1">
        {label}
      </label>
      {children}
      {hint && <p className="mt-1 text-xs text-gray-400">{hint}</p>}
    </div>
  );
}

function Select({ id, value, onChange, options, placeholder = "— select —" }) {
  return (
    <select
      id={id}
      value={value}
      onChange={onChange}
      className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 bg-white"
    >
      <option value="">{placeholder}</option>
      {options.map(([v, label]) => (
        <option key={v} value={v}>
          {label}
        </option>
      ))}
    </select>
  );
}

function NumberInput({ id, value, onChange, min, max, step = 1, placeholder }) {
  return (
    <input
      id={id}
      type="number"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
    />
  );
}

function Toggle({ id, label, checked, onChange }) {
  return (
    <label htmlFor={id} className="flex items-center gap-3 cursor-pointer select-none">
      <div className="relative">
        <input id={id} type="checkbox" className="sr-only" checked={checked} onChange={onChange} />
        <div
          className={`w-10 h-5 rounded-full transition-colors ${
            checked ? "bg-indigo-600" : "bg-gray-300"
          }`}
        />
        <div
          className={`absolute top-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform ${
            checked ? "translate-x-5" : "translate-x-0.5"
          }`}
        />
      </div>
      <span className="text-sm text-gray-700">{label}</span>
    </label>
  );
}

// ─── Initial form state ────────────────────────────────────────────────────────

const INITIAL_FORM = {
  age: "",
  gender: "",
  state: "Telangana",
  district: "",
  residence_type: "",
  category: "",
  annual_income: "",
  occupation: "",
  farmer_status: false,
  land_ownership_status: false,
  land_acres: "",
  student_status: false,
  education_level: "",
  business_owner_status: false,
  project_cost: "",
  loan_requirement: "",
  owns_pucca_house: false,
  is_bpl_or_white_ration_card: false,
  pays_income_tax: false,
  has_savings_bank_account: false,
  is_pregnant_or_newborn: false,
  disability_status: false,
  pensioner_status: false,
  pensioner_category: "",
  monthly_pension: "",
  marital_status: "",
  // API options
  top_k: 5,
  enable_xgboost: false,
  include_rag_explanation: false,
};

// ─── Helper: build clean profile payload ──────────────────────────────────────

function buildProfile(form) {
  const profile = {};
  const numFields = [
    "age",
    "annual_income",
    "land_acres",
    "project_cost",
    "loan_requirement",
    "monthly_pension",
  ];
  const boolFields = [
    "farmer_status",
    "land_ownership_status",
    "student_status",
    "business_owner_status",
    "owns_pucca_house",
    "is_bpl_or_white_ration_card",
    "pays_income_tax",
    "has_savings_bank_account",
    "is_pregnant_or_newborn",
    "disability_status",
    "pensioner_status",
  ];
  const strFields = [
    "gender",
    "state",
    "district",
    "residence_type",
    "category",
    "occupation",
    "education_level",
    "pensioner_category",
    "marital_status",
  ];

  strFields.forEach((f) => {
    if (form[f]) profile[f] = form[f];
  });
  numFields.forEach((f) => {
    if (form[f] !== "" && form[f] !== null) profile[f] = Number(form[f]);
  });
  boolFields.forEach((f) => {
    if (form[f]) profile[f] = true; // only send true booleans to keep payload minimal
  });

  return profile;
}

// ─── Recommendation card ───────────────────────────────────────────────────────

function RecommendationCard({ rec, rank }) {
  const [expanded, setExpanded] = useState(false);
  const hasRules =
    rec.matched_rules?.length || rec.failed_rules?.length || rec.unknown_rules?.length;
  const missing = rec.missing_information || [];
  const unstructured = rec.unstructured_criteria || [];
  const xgScore = rec.scoring_breakdown?.xgboost_score;

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm hover:shadow-md transition-shadow">
      {/* Card header */}
      <div className="p-5 flex items-start gap-4">
        <div className="flex-shrink-0 w-9 h-9 rounded-full bg-indigo-100 text-indigo-700 font-bold text-sm flex items-center justify-center">
          {rank}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex flex-wrap items-center gap-2 mb-1">
            <h3 className="font-bold text-gray-900 text-base">{rec.scheme_name}</h3>
            <StatusBadge status={rec.eligibility_status} />
            <span className="text-xs text-gray-400 font-mono">{rec.scheme_id}</span>
          </div>
          <p className="text-xs text-gray-500 mb-2">
            {rec.category} · {rec.scheme_scope} Scheme ·{" "}
            <span className="font-medium text-gray-600">{rec.confidence} confidence</span>
          </p>
          {rec.explanation_summary && (
            <p className="text-sm text-gray-700 leading-relaxed">{rec.explanation_summary}</p>
          )}
        </div>
        <div className="flex-shrink-0 text-right">
          <div className="text-lg font-bold text-indigo-700">{rec.score?.toFixed(1)}</div>
          <div className="text-xs text-gray-400">score</div>
          {xgScore !== undefined && (
            <div className="mt-1 text-xs text-amber-600 font-medium" title="Advisory XGBoost score">
              XGB: {Number(xgScore).toFixed(4)}
            </div>
          )}
        </div>
      </div>

      {/* Expandable details */}
      <div className="border-t border-gray-100 px-5 py-3 flex items-center justify-between gap-3 flex-wrap">
        <div className="flex flex-wrap gap-3 text-xs text-gray-500">
          {rec.matched_rules?.length > 0 && (
            <span className="text-green-600">✓ {rec.matched_rules.length} criteria met</span>
          )}
          {rec.failed_rules?.length > 0 && (
            <span className="text-red-500">✗ {rec.failed_rules.length} criteria failed</span>
          )}
          {rec.unknown_rules?.length > 0 && (
            <span className="text-yellow-600">? {rec.unknown_rules.length} unknown</span>
          )}
          {missing.length > 0 && (
            <span className="text-blue-500">ℹ {missing.length} info needed</span>
          )}
        </div>
        <div className="flex items-center gap-3">
          {rec.official_url && (
            <a
              href={rec.official_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-indigo-600 hover:underline"
            >
              Official Site ↗
            </a>
          )}
          <Link
            to={`/schemes/${rec.scheme_id}`}
            className="text-xs text-indigo-600 hover:underline"
          >
            View Details →
          </Link>
          {hasRules && (
            <button
              onClick={() => setExpanded((v) => !v)}
              className="text-xs text-gray-500 hover:text-gray-700"
            >
              {expanded ? "▲ Hide" : "▼ Rules"}
            </button>
          )}
        </div>
      </div>

      {/* Expanded rule breakdown */}
      {expanded && (
        <div className="border-t border-gray-100 px-5 py-4 space-y-4 bg-gray-50 rounded-b-xl">
          <RuleList title="✓ Matched Rules" rules={rec.matched_rules} color="green" />
          <RuleList title="✗ Failed Rules" rules={rec.failed_rules} color="red" />
          <RuleList title="? Unknown Rules" rules={rec.unknown_rules} color="yellow" />
          {missing.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-blue-700 mb-1">ℹ Missing Information</p>
              <ul className="text-xs text-gray-600 space-y-0.5">
                {missing.map((m, i) => (
                  <li key={i} className="flex gap-1.5">
                    <span>•</span>
                    <span>{m}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {unstructured.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-gray-600 mb-1">📋 Additional Criteria</p>
              <ul className="text-xs text-gray-500 space-y-0.5">
                {unstructured.map((c, i) => (
                  <li key={i} className="flex gap-1.5">
                    <span>•</span>
                    <span>{c.description || c.text || JSON.stringify(c)}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {rec.source_authority && (
            <p className="text-xs text-gray-400">Source: {rec.source_authority}</p>
          )}
        </div>
      )}
    </div>
  );
}

function RuleList({ title, rules, color }) {
  if (!rules || rules.length === 0) return null;
  const textColor = { green: "text-green-700", red: "text-red-700", yellow: "text-yellow-700" }[color];
  const bgColor = { green: "bg-green-50", red: "bg-red-50", yellow: "bg-yellow-50" }[color];

  return (
    <div>
      <p className={`text-xs font-semibold ${textColor} mb-1`}>{title}</p>
      <ul className={`rounded-md ${bgColor} px-3 py-2 space-y-1.5`}>
        {rules.map((rule) => (
          <li key={rule.rule_id} className="text-xs text-gray-700">
            <span className="font-medium">{rule.field}</span>
            {rule.reason ? ` — ${rule.reason}` : ""}
          </li>
        ))}
      </ul>
    </div>
  );
}

// ─── Main page component ───────────────────────────────────────────────────────

export default function RecommendPage() {
  const [form, setForm] = useState(INITIAL_FORM);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (field) => (e) => {
    const val =
      e.target.type === "checkbox" ? e.target.checked : e.target.value;
    setForm((f) => ({ ...f, [field]: val }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const profile = buildProfile(form);
      const data = await getRecommendations(profile, {
        top_k: Number(form.top_k),
        enable_xgboost: form.enable_xgboost,
        include_rag_explanation: form.include_rag_explanation,
      });
      setResult(data);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setForm(INITIAL_FORM);
    setResult(null);
    setError(null);
  };

  return (
    <div className="max-w-5xl mx-auto px-4 py-10">
      <h1 className="text-3xl font-bold text-gray-900 mb-1">Check Scheme Eligibility</h1>
      <p className="text-gray-500 text-sm mb-8">
        Enter your profile details below. All eligibility is determined by the backend's verified
        rule engine — this form never decides eligibility.
      </p>

      <div className="grid lg:grid-cols-5 gap-8">
        {/* ── Form ── */}
        <form
          onSubmit={handleSubmit}
          className="lg:col-span-2 space-y-6 bg-white rounded-xl border border-gray-200 p-6 self-start"
        >
          <h2 className="text-lg font-semibold text-gray-800">👤 Demographics</h2>

          <FieldGroup label="Age (years)" htmlFor="age">
            <NumberInput
              id="age"
              value={form.age}
              onChange={handleChange("age")}
              min={0}
              max={120}
              placeholder="e.g. 35"
            />
          </FieldGroup>

          <FieldGroup label="Gender" htmlFor="gender">
            <Select
              id="gender"
              value={form.gender}
              onChange={handleChange("gender")}
              options={[
                ["female", "Female"],
                ["male", "Male"],
                ["transgender", "Transgender"],
                ["other", "Other"],
              ]}
            />
          </FieldGroup>

          <FieldGroup label="State" htmlFor="state">
            <Select
              id="state"
              value={form.state}
              onChange={handleChange("state")}
              options={[
                ["Telangana", "Telangana"],
                ["Andhra Pradesh", "Andhra Pradesh"],
                ["other", "Other"],
              ]}
            />
          </FieldGroup>

          <FieldGroup label="District" htmlFor="district">
            <input
              id="district"
              type="text"
              value={form.district}
              onChange={handleChange("district")}
              placeholder="e.g. Nalgonda"
              className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </FieldGroup>

          <FieldGroup label="Residence Type" htmlFor="residence_type">
            <Select
              id="residence_type"
              value={form.residence_type}
              onChange={handleChange("residence_type")}
              options={[
                ["rural", "Rural"],
                ["urban", "Urban"],
              ]}
            />
          </FieldGroup>

          <FieldGroup label="Social Category" htmlFor="category">
            <Select
              id="category"
              value={form.category}
              onChange={handleChange("category")}
              options={[
                ["SC", "SC"],
                ["ST", "ST"],
                ["BC", "BC"],
                ["EBC", "EBC"],
                ["OBC", "OBC"],
                ["Minority", "Minority"],
                ["General", "General"],
              ]}
            />
          </FieldGroup>

          <FieldGroup label="Annual Family Income (₹)" htmlFor="annual_income">
            <NumberInput
              id="annual_income"
              value={form.annual_income}
              onChange={handleChange("annual_income")}
              min={0}
              step={1000}
              placeholder="e.g. 150000"
            />
          </FieldGroup>

          <FieldGroup label="Marital Status" htmlFor="marital_status">
            <Select
              id="marital_status"
              value={form.marital_status}
              onChange={handleChange("marital_status")}
              options={[
                ["single", "Single"],
                ["married", "Married"],
                ["widowed", "Widowed"],
                ["divorced", "Divorced"],
              ]}
            />
          </FieldGroup>

          {/* Occupation */}
          <hr className="border-gray-100" />
          <h2 className="text-lg font-semibold text-gray-800">💼 Occupation &amp; Assets</h2>

          <FieldGroup label="Occupation" htmlFor="occupation">
            <Select
              id="occupation"
              value={form.occupation}
              onChange={handleChange("occupation")}
              options={[
                ["farmer", "Farmer"],
                ["student", "Student"],
                ["business_owner", "Business Owner"],
                ["artisan", "Artisan"],
                ["weaver", "Weaver"],
                ["unemployed", "Unemployed"],
                ["employed", "Employed (Salaried)"],
                ["self_employed", "Self Employed"],
                ["retired", "Retired"],
                ["other", "Other"],
              ]}
            />
          </FieldGroup>

          <div className="space-y-3">
            <Toggle
              id="farmer_status"
              label="Actively engaged in farming"
              checked={form.farmer_status}
              onChange={handleChange("farmer_status")}
            />
            <Toggle
              id="land_ownership_status"
              label="Owns agricultural land"
              checked={form.land_ownership_status}
              onChange={handleChange("land_ownership_status")}
            />
          </div>

          {(form.farmer_status || form.land_ownership_status) && (
            <FieldGroup label="Land holding (acres)" htmlFor="land_acres">
              <NumberInput
                id="land_acres"
                value={form.land_acres}
                onChange={handleChange("land_acres")}
                min={0}
                step={0.1}
                placeholder="e.g. 2.5"
              />
            </FieldGroup>
          )}

          <Toggle
            id="student_status"
            label="Currently enrolled as a student"
            checked={form.student_status}
            onChange={handleChange("student_status")}
          />

          {form.student_status && (
            <FieldGroup label="Education Level" htmlFor="education_level">
              <Select
                id="education_level"
                value={form.education_level}
                onChange={handleChange("education_level")}
                options={[
                  ["below_8th", "Below 8th"],
                  ["8th_pass", "8th Pass"],
                  ["10th_pass", "10th Pass"],
                  ["12th_pass", "12th Pass"],
                  ["diploma", "Diploma"],
                  ["graduate", "Graduate"],
                  ["post_graduate", "Post-Graduate"],
                ]}
              />
            </FieldGroup>
          )}

          <Toggle
            id="business_owner_status"
            label="Owns / proposes a micro-enterprise"
            checked={form.business_owner_status}
            onChange={handleChange("business_owner_status")}
          />

          {form.business_owner_status && (
            <>
              <FieldGroup label="Project Cost (₹)" htmlFor="project_cost">
                <NumberInput
                  id="project_cost"
                  value={form.project_cost}
                  onChange={handleChange("project_cost")}
                  min={0}
                  step={1000}
                  placeholder="e.g. 200000"
                />
              </FieldGroup>
              <FieldGroup label="Loan Requirement (₹)" htmlFor="loan_requirement">
                <NumberInput
                  id="loan_requirement"
                  value={form.loan_requirement}
                  onChange={handleChange("loan_requirement")}
                  min={0}
                  step={1000}
                  placeholder="e.g. 150000"
                />
              </FieldGroup>
            </>
          )}

          {/* Housing & Benefits */}
          <hr className="border-gray-100" />
          <h2 className="text-lg font-semibold text-gray-800">🏠 Housing &amp; Benefits</h2>

          <div className="space-y-3">
            <Toggle
              id="owns_pucca_house"
              label="Owns a pucca house"
              checked={form.owns_pucca_house}
              onChange={handleChange("owns_pucca_house")}
            />
            <Toggle
              id="is_bpl_or_white_ration_card"
              label="Holds BPL / Food Security Card"
              checked={form.is_bpl_or_white_ration_card}
              onChange={handleChange("is_bpl_or_white_ration_card")}
            />
            <Toggle
              id="pays_income_tax"
              label="Active income tax payer"
              checked={form.pays_income_tax}
              onChange={handleChange("pays_income_tax")}
            />
            <Toggle
              id="has_savings_bank_account"
              label="Has savings bank account"
              checked={form.has_savings_bank_account}
              onChange={handleChange("has_savings_bank_account")}
            />
            <Toggle
              id="is_pregnant_or_newborn"
              label="Pregnant / mother with newborn"
              checked={form.is_pregnant_or_newborn}
              onChange={handleChange("is_pregnant_or_newborn")}
            />
            <Toggle
              id="disability_status"
              label="Certified disability"
              checked={form.disability_status}
              onChange={handleChange("disability_status")}
            />
            <Toggle
              id="pensioner_status"
              label="Drawing / seeking a pension"
              checked={form.pensioner_status}
              onChange={handleChange("pensioner_status")}
            />
          </div>

          {form.pensioner_status && (
            <>
              <FieldGroup label="Pension Category" htmlFor="pensioner_category">
                <Select
                  id="pensioner_category"
                  value={form.pensioner_category}
                  onChange={handleChange("pensioner_category")}
                  options={[
                    ["old_age", "Old Age"],
                    ["widow", "Widow"],
                    ["disabled", "Disabled"],
                    ["weaver", "Weaver"],
                    ["toddy_tapper", "Toddy Tapper"],
                    ["single_women", "Single Women"],
                    ["filaria", "Filaria"],
                  ]}
                />
              </FieldGroup>
              <FieldGroup label="Current Monthly Pension (₹)" htmlFor="monthly_pension">
                <NumberInput
                  id="monthly_pension"
                  value={form.monthly_pension}
                  onChange={handleChange("monthly_pension")}
                  min={0}
                  step={100}
                  placeholder="e.g. 2000"
                />
              </FieldGroup>
            </>
          )}

          {/* API Options */}
          <hr className="border-gray-100" />
          <h2 className="text-lg font-semibold text-gray-800">⚙️ Options</h2>

          <FieldGroup
            label="Number of results (top-k)"
            htmlFor="top_k"
            hint="1 – 14"
          >
            <NumberInput
              id="top_k"
              value={form.top_k}
              onChange={handleChange("top_k")}
              min={1}
              max={14}
              placeholder="5"
            />
          </FieldGroup>

          <div className="space-y-3">
            <Toggle
              id="enable_xgboost"
              label="Enable advisory XGBoost ranking"
              checked={form.enable_xgboost}
              onChange={handleChange("enable_xgboost")}
            />
            <Toggle
              id="include_rag_explanation"
              label="Include RAG context explanation"
              checked={form.include_rag_explanation}
              onChange={handleChange("include_rag_explanation")}
            />
          </div>

          {/* Privacy notice */}
          <p className="text-xs text-gray-400 leading-relaxed bg-gray-50 rounded p-2">
            🔒 Do NOT submit Aadhaar, PAN, or bank account numbers. Only anonymised demographic and
            occupational data is used for eligibility screening.
          </p>

          {/* Actions */}
          <div className="flex gap-3">
            <button
              type="submit"
              disabled={loading}
              className="flex-1 bg-indigo-700 text-white font-semibold py-2.5 rounded-lg hover:bg-indigo-600 disabled:opacity-50 transition-colors"
            >
              {loading ? "Evaluating…" : "Check Eligibility"}
            </button>
            <button
              type="button"
              onClick={handleReset}
              className="px-4 py-2.5 border border-gray-300 text-gray-600 rounded-lg hover:bg-gray-100 text-sm"
            >
              Reset
            </button>
          </div>
        </form>

        {/* ── Results Panel ── */}
        <div className="lg:col-span-3 space-y-5">
          {loading && <LoadingSpinner label="Evaluating eligibility…" />}

          {error && (
            <ErrorAlert error={error} onDismiss={() => setError(null)} />
          )}

          {result && !loading && (
            <>
              {/* Summary */}
              <div className="bg-white rounded-xl border border-gray-200 p-5">
                <h2 className="text-lg font-bold text-gray-800 mb-3">
                  📊 Evaluation Summary
                </h2>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mb-4">
                  {Object.entries(result.status_counts || {}).map(([status, count]) => (
                    <div
                      key={status}
                      className="text-center bg-gray-50 rounded-lg py-3 px-2 border border-gray-100"
                    >
                      <div className="text-2xl font-bold text-gray-800">{count}</div>
                      <div className="mt-1">
                        <StatusBadge status={status} />
                      </div>
                    </div>
                  ))}
                </div>
                <p className="text-xs text-gray-400">
                  {result.total_schemes_evaluated} schemes evaluated ·{" "}
                  {result.evaluation_timestamp
                    ? new Date(result.evaluation_timestamp).toLocaleString()
                    : ""}
                </p>
              </div>

              {/* Profile summary */}
              {result.user_profile_summary && (
                <div className="bg-indigo-50 rounded-xl border border-indigo-100 p-4">
                  <p className="text-xs font-semibold text-indigo-700 mb-2">Profile Summary</p>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(result.user_profile_summary)
                      .filter(([, v]) => v !== null && v !== undefined && v !== false)
                      .map(([k, v]) => (
                        <span
                          key={k}
                          className="px-2 py-0.5 bg-indigo-100 text-indigo-700 rounded text-xs"
                        >
                          {k.replace(/_/g, " ")}: {String(v)}
                        </span>
                      ))}
                  </div>
                </div>
              )}

              {/* Recommendation cards */}
              <h2 className="text-xl font-bold text-gray-800">
                🏆 Top {result.recommendations?.length || 0} Recommendations
              </h2>
              {(result.recommendations || []).length === 0 ? (
                <p className="text-gray-500 text-sm">No recommendations returned.</p>
              ) : (
                <div className="space-y-4">
                  {result.recommendations.map((rec, idx) => (
                    <RecommendationCard key={rec.scheme_id} rec={rec} rank={idx + 1} />
                  ))}
                </div>
              )}

              {/* Disclaimer */}
              {result.disclaimer && (
                <div className="text-xs text-gray-400 bg-gray-50 rounded-lg p-4 border border-gray-100 leading-relaxed">
                  ℹ️ {result.disclaimer}
                </div>
              )}
            </>
          )}

          {!result && !loading && !error && (
            <div className="flex flex-col items-center justify-center py-20 text-center text-gray-400">
              <div className="text-6xl mb-4">🔍</div>
              <p className="font-medium text-gray-500">Fill in your profile and click "Check Eligibility"</p>
              <p className="text-sm mt-1">Results will appear here.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
