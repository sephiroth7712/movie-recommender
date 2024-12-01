import { useState, useEffect } from "react";

interface StarRatingProps {
  initialRating?: number;
  onChange: (rating: number) => void;
  readonly?: boolean;
}

export default function StarRating({
  initialRating = 0,
  onChange,
  readonly = false,
}: StarRatingProps) {
  const [rating, setRating] = useState(initialRating);
  const [hover, setHover] = useState(0);

  useEffect(() => {
    setRating(initialRating);
  }, [initialRating]);

  const handleClick = (value: number) => {
    if (readonly) return;
    setRating(value);
    onChange(value);
  };

  return (
    <div className="flex items-center">
      {[1, 2, 3, 4, 5].map((value) => (
        <svg
          key={value}
          className={`w-4 h-4 ms-1 cursor-pointer transition-colors ${
            readonly ? "cursor-default" : "cursor-pointer"
          } ${
            (hover || rating) >= value
              ? "fill-yellow-400 text-yellow-400"
              : "text-gray-300"
          }`}
          aria-hidden="true"
          xmlns="http://www.w3.org/2000/svg"
          fill="currentColor"
          viewBox="0 0 22 20"
          onMouseEnter={() => !readonly && setHover(value)}
          onMouseLeave={() => !readonly && setHover(0)}
          onClick={() => handleClick(value)}
        >
          <path d="M20.924 7.625a1.523 1.523 0 0 0-1.238-1.044l-5.051-.734-2.259-4.577a1.534 1.534 0 0 0-2.752 0L7.365 5.847l-5.051.734A1.535 1.535 0 0 0 1.463 9.2l3.656 3.563-.863 5.031a1.532 1.532 0 0 0 2.226 1.616L11 17.033l4.518 2.375a1.534 1.534 0 0 0 2.226-1.617l-.863-5.03L20.537 9.2a1.523 1.523 0 0 0 .387-1.575Z" />
        </svg>
      ))}
    </div>
  );
}
