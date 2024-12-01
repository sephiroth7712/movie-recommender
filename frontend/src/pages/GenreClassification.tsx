// src/pages/MovieGenreClassification.tsx

import { useState } from "react";
import {
  genreClassificationService,
  GenrePredictionResponse,
} from "../services/genrePrediction.service";
import GenrePredictions from "../components/features/GenrePredictions";

export default function MovieGenreClassification() {
  const [plot, setPlot] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [predictions, setPredictions] =
    useState<GenrePredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!plot.trim()) {
      setError("Please enter a plot summary");
      return;
    }

    try {
      setIsLoading(true);
      setError(null);
      const result = await genreClassificationService.predictGenres({
        plot_summary: plot,
      });
      setPredictions(result);
    } catch (err) {
      setError("An error occurred while predicting genres. Please try again.");
      console.error("Error predicting genres:", err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-gray-800">
          Movie Genre Classification
        </h2>
        <p className="text-gray-600 dark:text-gray-400">
          Enter a movie plot summary to predict its genres
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4 mb-8">
        <div>
          <label
            htmlFor="plot"
            className="block mb-2 text-sm font-medium text-gray-900"
          >
            Plot Summary
          </label>
          <textarea
            id="plot"
            rows={6}
            className="block p-2.5 w-full text-sm text-gray-900 bg-white rounded-lg border border-gray-300 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-800 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
            placeholder="Enter the movie plot summary here..."
            value={plot}
            onChange={(e) => setPlot(e.target.value)}
          />
        </div>

        {error && (
          <div
            className="p-4 text-sm text-red-800 rounded-lg bg-red-50 dark:bg-gray-800 dark:text-red-400"
            role="alert"
          >
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={isLoading}
          className="text-white bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:ring-blue-300 font-medium rounded-lg text-sm px-5 py-2.5 text-center dark:bg-blue-600 dark:hover:bg-blue-700 dark:focus:ring-blue-800 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? (
            <>
              <span className="inline-block animate-spin mr-2">⌛</span>
              Predicting...
            </>
          ) : (
            "Predict Genres"
          )}
        </button>
      </form>

      {predictions && !isLoading && (
        <div className="space-y-6">
          <div className="p-6 bg-white border border-gray-200 rounded-lg shadow dark:bg-gray-800 dark:border-gray-700">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Analyzed Plot Summary
            </h3>
            <p className="text-gray-700 dark:text-gray-400">
              {predictions.plot_summary}
            </p>
          </div>

          <GenrePredictions predictions={predictions.predictions} />
        </div>
      )}
    </div>
  );
}
