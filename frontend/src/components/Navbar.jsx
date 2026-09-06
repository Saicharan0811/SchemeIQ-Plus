/**
 * components/Navbar.jsx
 * Top navigation bar with SchemeIQ+ branding and navigation links.
 */
import { Link, useLocation } from "react-router-dom";

const NAV_LINKS = [
  { to: "/", label: "Home" },
  { to: "/recommend", label: "Check Eligibility" },
  { to: "/ask", label: "Ask SchemeIQ+" },
];

export default function Navbar() {
  const { pathname } = useLocation();

  return (
    <nav className="bg-indigo-800 text-white shadow-md">
      <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
        {/* Branding */}
        <Link to="/" className="flex items-center gap-2 font-bold text-xl tracking-tight">
          <span className="text-2xl">🏛️</span>
          <span>SchemeIQ<span className="text-yellow-300">+</span></span>
        </Link>

        {/* Nav links */}
        <div className="flex items-center gap-1">
          {NAV_LINKS.map(({ to, label }) => (
            <Link
              key={to}
              to={to}
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                pathname === to
                  ? "bg-indigo-600 text-white"
                  : "text-indigo-200 hover:bg-indigo-700 hover:text-white"
              }`}
            >
              {label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
}
