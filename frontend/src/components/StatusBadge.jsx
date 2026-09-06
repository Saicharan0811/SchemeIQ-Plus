/**
 * components/StatusBadge.jsx
 * Renders a colour-coded eligibility status badge.
 * Colours are static Tailwind classes so the JIT compiler can detect them.
 */

const STATUS_CONFIG = {
  ELIGIBLE: {
    label: "Eligible",
    bg: "bg-green-100",
    text: "text-green-800",
    border: "border-green-300",
    dot: "bg-green-500",
  },
  POTENTIALLY_ELIGIBLE: {
    label: "Potentially Eligible",
    bg: "bg-yellow-100",
    text: "text-yellow-800",
    border: "border-yellow-300",
    dot: "bg-yellow-500",
  },
  INSUFFICIENT_INFORMATION: {
    label: "Insufficient Info",
    bg: "bg-blue-100",
    text: "text-blue-800",
    border: "border-blue-300",
    dot: "bg-blue-500",
  },
  NOT_APPLICABLE: {
    label: "Not Applicable",
    bg: "bg-gray-100",
    text: "text-gray-600",
    border: "border-gray-300",
    dot: "bg-gray-400",
  },
  NOT_ELIGIBLE: {
    label: "Not Eligible",
    bg: "bg-red-100",
    text: "text-red-800",
    border: "border-red-300",
    dot: "bg-red-500",
  },
};

export default function StatusBadge({ status }) {
  const cfg = STATUS_CONFIG[status] || {
    label: status,
    bg: "bg-gray-100",
    text: "text-gray-700",
    border: "border-gray-300",
    dot: "bg-gray-400",
  };

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold border ${cfg.bg} ${cfg.text} ${cfg.border}`}
    >
      <span className={`w-2 h-2 rounded-full ${cfg.dot}`} />
      {cfg.label}
    </span>
  );
}
