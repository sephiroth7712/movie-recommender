import { genreColors } from "../../config/genres";
import { GenrePrediction } from "../../services/genrePrediction.service";

interface GenrePredictionsProps {
  predictions: GenrePrediction[];
}

export default function GenrePredictions({
  predictions,
}: GenrePredictionsProps) {
  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
        Predicted Genres
      </h3>
      <div className="space-y-2">
        {predictions.map((prediction) => (
          <div
            key={prediction.genre}
            className="flex items-center justify-between p-3 bg-white border border-gray-200 rounded-lg dark:bg-gray-800 dark:border-gray-700"
          >
            <span
              className={`${genreColors[prediction.genre]} text-white text-sm font-medium px-3 py-1 rounded`}
            >
              {prediction.genre}
            </span>
            <div className="flex items-center gap-2">
              <div className="w-48 bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
                <div
                  className="bg-blue-600 h-2.5 rounded-full"
                  style={{ width: `${prediction.confidence * 100}%` }}
                ></div>
              </div>
              <span className="text-sm text-gray-500 dark:text-gray-400">
                {(prediction.confidence * 100).toFixed(2)}%
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
