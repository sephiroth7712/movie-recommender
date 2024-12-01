export default function MovieLoadingCard() {
  return (
    <div
      role="status"
      className="max-w-sm p-6 border border-gray-200 rounded-lg shadow animate-pulse dark:border-gray-800"
    >
      <div className="h-5 bg-gray-200 rounded-full dark:bg-gray-700 w-48 mb-4"></div>
      <div className="h-2 bg-gray-200 rounded-full dark:bg-gray-700 mb-2.5"></div>
      <div className="h-2 bg-gray-200 rounded-full dark:bg-gray-700 mb-2.5"></div>
      <div className="h-2 bg-gray-200 rounded-full dark:bg-gray-700 mb-3"></div>
      <div className="flex flex-wrap gap-2 mb-3">
        {[...Array(4)].map((_) => (
          <span
            className={`w-10 h-5 bg-gray-200 font-medium me-2 px-2.5 py-0.5 rounded`}
          ></span>
        ))}
      </div>
      <span className="sr-only">Loading...</span>
    </div>
  );
}
