/**
 * components/ErrorAlert.jsx
 * Renders a dismissible error alert box.
 */
export default function ErrorAlert({ error, onDismiss }) {
  if (!error) return null;

  const message = typeof error === "string" ? error : error.message || "An error occurred.";
  const code = error.code;

  return (
    <div
      role="alert"
      className="flex items-start gap-3 p-4 rounded-lg border border-red-300 bg-red-50 text-red-800"
    >
      <span className="text-lg mt-0.5">⚠️</span>
      <div className="flex-1 min-w-0">
        <p className="font-semibold">Error{code ? ` (${code})` : ""}</p>
        <p className="text-sm mt-0.5 break-words">{message}</p>
        {error.details && error.details.length > 0 && (
          <ul className="mt-2 text-xs list-disc list-inside space-y-0.5">
            {error.details.map((d, i) => (
              <li key={i}>
                <strong>{d.field}:</strong> {d.message}
              </li>
            ))}
          </ul>
        )}
      </div>
      {onDismiss && (
        <button
          onClick={onDismiss}
          aria-label="Dismiss error"
          className="text-red-500 hover:text-red-700 font-bold text-lg leading-none"
        >
          ×
        </button>
      )}
    </div>
  );
}
