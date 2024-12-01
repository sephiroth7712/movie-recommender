import { useState, useMemo, useCallback } from "react";
import {
  Combobox,
  ComboboxInput,
  ComboboxOptions,
  ComboboxOption,
} from "@headlessui/react";
import debounce from "lodash/debounce";
import { IoIosSearch } from "react-icons/io";
import { movieService } from "../../services/movie.service";
import { Movie } from "../../types/movie";
import { useNavigate } from "react-router-dom";

const SearchBar = () => {
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Movie[]>([]);

  const searchMovies = useCallback(async (searchTerm: string) => {
    const searchResults = await movieService.searchMovie({ name: searchTerm });
    setResults(searchResults.movies);
  }, []);

  const debouncedSearch = useMemo(() => {
    return debounce(searchMovies, 1000);
  }, [searchMovies]);

  const handleSearch = (value: string) => {
    setQuery(value);
    debouncedSearch(value);
  };

  const handleSelection = (value: string) => {
    navigate(`/movie/${value}`);
  };

  return (
    <div className="w-full mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 m">
      <Combobox
        as="div"
        className="relative"
        value={query}
        onClose={() => setQuery("")}
        onChange={handleSelection}
      >
        <div className="relative">
          <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3">
            <IoIosSearch />
          </div>
          <ComboboxInput
            className="block w-full p-4 ps-10 text-sm text-gray-900 border border-gray-300 rounded-lg bg-gray-50 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
            onChange={(event) => handleSearch(event.target.value)}
            placeholder="Search movies..."
          />
        </div>
        <div className="w-full p-6">
          <ComboboxOptions
            anchor="bottom"
            className="w-[var(--input-width)] border border-gray-300 rounded-lg bg-gray-50 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
          >
            {results.map((movie) => (
              <ComboboxOption
                key={movie.movie_id}
                value={movie.movie_id}
                className="data-[focus]:bg-blue-100 text-gray-900 p-4 ps-10 text-sm"
              >
                {movie.title}
              </ComboboxOption>
            ))}
          </ComboboxOptions>
        </div>
      </Combobox>
    </div>
  );
};

export default SearchBar;
