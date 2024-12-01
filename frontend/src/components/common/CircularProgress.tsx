export default function CircularProgress({
  percentage,
}: {
  percentage: number;
}) {
  const circleWidth = 40;
  const radius = 15;
  const dashArray = radius * Math.PI * 2;
  const dashOffset = dashArray - (dashArray * percentage) / 100;

  return (
    <div className="relative w-10 h-10">
      <svg
        width={circleWidth}
        height={circleWidth}
        viewBox={`0 0 ${circleWidth} ${circleWidth}`}
      >
        {/* Background circle */}
        <circle
          cx={circleWidth / 2}
          cy={circleWidth / 2}
          r={radius}
          className="fill-none stroke-gray-200"
          strokeWidth="4"
        />
        {/* Progress circle */}
        <circle
          cx={circleWidth / 2}
          cy={circleWidth / 2}
          r={radius}
          className="fill-none stroke-green-500"
          strokeWidth="4"
          strokeLinecap="round"
          style={{
            strokeDasharray: dashArray,
            strokeDashoffset: dashOffset,
            transform: "rotate(-90deg)",
            transformOrigin: "center",
          }}
        />
      </svg>
      {/* Percentage text */}
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-xs font-semibold">{percentage}%</span>
      </div>
    </div>
  );
}
