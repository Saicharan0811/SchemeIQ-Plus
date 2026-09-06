/**
 * components/LoadingSpinner.jsx
 * Centered spinning indicator with optional label.
 */
export default function LoadingSpinner({ label = "Loading…" }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 gap-3">
      <div className="w-10 h-10 border-4 border-indigo-300 border-t-indigo-700 rounded-full animate-spin" />
      <p className="text-sm text-gray-500">{label}</p>
    </div>
  );
}
