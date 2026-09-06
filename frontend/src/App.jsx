/**
 * App.jsx — Root application with client-side routing
 */
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import HomePage from "./pages/HomePage";
import RecommendPage from "./pages/RecommendPage";
import SchemeDetailPage from "./pages/SchemeDetailPage";
import AskPage from "./pages/AskPage";

function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] text-center px-4">
      <div className="text-6xl mb-4">🔍</div>
      <h1 className="text-2xl font-bold text-gray-800 mb-2">Page Not Found</h1>
      <p className="text-gray-500 mb-6">The page you're looking for doesn't exist.</p>
      <a
        href="/"
        className="px-4 py-2 bg-indigo-700 text-white rounded-lg hover:bg-indigo-600 text-sm font-medium"
      >
        Go Home
      </a>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen flex flex-col">
        <Navbar />
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/recommend" element={<RecommendPage />} />
            <Route path="/schemes/:schemeId" element={<SchemeDetailPage />} />
            <Route path="/ask" element={<AskPage />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </main>

        {/* Footer */}
        <footer className="bg-gray-800 text-gray-400 text-xs py-4 text-center">
          <p>
            SchemeIQ+ · Welfare Scheme Navigator for Telangana &amp; Central India ·{" "}
            <span className="text-gray-500">Results are advisory only.</span>
          </p>
        </footer>
      </div>
    </BrowserRouter>
  );
}
