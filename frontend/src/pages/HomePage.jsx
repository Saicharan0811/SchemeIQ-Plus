/**
 * pages/HomePage.jsx — Landing / Home page
 *
 * Displays SchemeIQ+ branding, project description, and navigation cards
 * to the Eligibility Checker and Ask SchemeIQ+ pages.
 */
import { Link } from "react-router-dom";

const FEATURE_CARDS = [
  {
    to: "/recommend",
    icon: "🔍",
    title: "Check Eligibility",
    description:
      "Enter your profile details to discover which Telangana and Central Government welfare schemes you qualify for. Get transparent, rule-based eligibility assessments with detailed explanations.",
    cta: "Check Now →",
    color: "indigo",
  },
  {
    to: "/ask",
    icon: "💬",
    title: "Ask SchemeIQ+",
    description:
      "Ask natural language questions about government schemes. Get answers grounded strictly in official government orders and notifications with full source citations.",
    cta: "Ask a Question →",
    color: "emerald",
  },
];

const SCHEME_HIGHLIGHTS = [
  { name: "Rythu Bharosa", category: "Agriculture", scope: "State" },
  { name: "PM Kisan", category: "Agriculture", scope: "Central" },
  { name: "Aasara Pension", category: "Social Security", scope: "State" },
  { name: "PMAY-G", category: "Housing", scope: "Central" },
  { name: "Cheyutha", category: "Entrepreneurship", scope: "State" },
  { name: "PM Matru Vandana Yojana", category: "Healthcare", scope: "Central" },
];

export default function HomePage() {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="bg-gradient-to-br from-indigo-800 via-indigo-700 to-indigo-600 text-white py-20 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <div className="text-6xl mb-4">🏛️</div>
          <h1 className="text-4xl md:text-5xl font-bold mb-4 tracking-tight">
            SchemeIQ<span className="text-yellow-300">+</span>
          </h1>
          <p className="text-xl md:text-2xl text-indigo-100 mb-3 font-light">
            Intelligent Welfare Scheme Navigator for Telangana &amp; Central India
          </p>
          <p className="text-base text-indigo-200 max-w-2xl mx-auto mb-8">
            Instantly discover government schemes you qualify for — powered by deterministic
            eligibility rules, verified official sources, and grounded AI explanations.
          </p>
          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <Link
              to="/recommend"
              className="px-6 py-3 bg-yellow-400 text-indigo-900 font-bold rounded-lg hover:bg-yellow-300 transition-colors shadow"
            >
              🔍 Check My Eligibility
            </Link>
            <Link
              to="/ask"
              className="px-6 py-3 bg-indigo-500 text-white font-semibold rounded-lg hover:bg-indigo-400 border border-indigo-300 transition-colors"
            >
              💬 Ask a Question
            </Link>
          </div>
        </div>
      </section>

      {/* Feature Cards */}
      <section className="py-14 px-4 bg-white">
        <div className="max-w-5xl mx-auto">
          <h2 className="text-2xl font-bold text-center text-gray-800 mb-10">
            What can SchemeIQ+ do for you?
          </h2>
          <div className="grid md:grid-cols-2 gap-6">
            {FEATURE_CARDS.map(({ to, icon, title, description, cta, color }) => (
              <Link
                key={to}
                to={to}
                className={`group block rounded-xl border-2 border-gray-100 hover:border-${color}-300 bg-white hover:shadow-lg transition-all p-6`}
              >
                <div className="text-4xl mb-3">{icon}</div>
                <h3 className={`text-xl font-bold text-gray-800 mb-2 group-hover:text-${color}-700`}>
                  {title}
                </h3>
                <p className="text-gray-600 text-sm mb-4 leading-relaxed">{description}</p>
                <span className={`text-${color}-600 font-semibold text-sm`}>{cta}</span>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* Schemes Covered */}
      <section className="py-12 px-4 bg-gray-50">
        <div className="max-w-5xl mx-auto">
          <h2 className="text-2xl font-bold text-center text-gray-800 mb-3">
            Schemes Covered
          </h2>
          <p className="text-center text-gray-500 text-sm mb-8">
            14 verified official schemes across Agriculture, Social Security, Healthcare, Housing &amp; more.
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {SCHEME_HIGHLIGHTS.map(({ name, category, scope }) => (
              <div
                key={name}
                className="bg-white rounded-lg border border-gray-200 p-4 text-center shadow-sm"
              >
                <p className="font-semibold text-gray-800 text-sm">{name}</p>
                <p className="text-xs text-gray-500 mt-1">{category}</p>
                <span
                  className={`mt-2 inline-block px-2 py-0.5 rounded text-xs font-medium ${
                    scope === "State"
                      ? "bg-indigo-100 text-indigo-700"
                      : "bg-orange-100 text-orange-700"
                  }`}
                >
                  {scope}
                </span>
              </div>
            ))}
            <div className="bg-indigo-50 rounded-lg border border-indigo-200 p-4 text-center shadow-sm flex flex-col justify-center">
              <p className="font-semibold text-indigo-700 text-sm">+ 8 more schemes</p>
              <Link to="/recommend" className="text-xs text-indigo-500 mt-1 hover:underline">
                Check eligibility →
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Trust / Disclaimer */}
      <section className="py-10 px-4 bg-white border-t border-gray-100">
        <div className="max-w-3xl mx-auto text-center">
          <h3 className="font-semibold text-gray-700 mb-2">🔒 Privacy &amp; Integrity</h3>
          <p className="text-sm text-gray-500 leading-relaxed">
            SchemeIQ+ does <strong>not</strong> collect or store Aadhaar numbers, PAN numbers, bank
            details, or any sensitive identifiers. All eligibility is assessed in-memory against verified
            official rule datasets. Results are advisory only — final benefit sanctioning rests with the
            competent government authority.
          </p>
        </div>
      </section>
    </div>
  );
}
