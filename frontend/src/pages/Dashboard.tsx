import SearchBar from "../components/features/SearchBar";
import Recommendation from "../components/layout/Recommendations";
import { useEffect, useState } from "react";
import {
  recommendationService,
  UserRecommendationsRequest,
  RecommendationsResponse,
} from "../services/recommendation.service";
import { useAuth } from "../hooks/useAuth";

export default function Dashboard() {
  const { user } = useAuth();
  const [isCBFLoading, setIsCBFLoading] = useState(true);
  const [isCoBFLoading, setIsCoBFLoading] = useState(true);
  const [cbfResponse, setCBFResponse] = useState<RecommendationsResponse>();
  const [cobfResponse, setCoBFResponse] = useState<RecommendationsResponse>();

  const fetchRecommendations = async (type: "cbf" | "cobf") => {
    try {
      if (user) {
        setIsCBFLoading(true);
        const data: UserRecommendationsRequest = {
          type,
          userId: user.id,
          n_recommendations: 4,
        };
        const results =
          await recommendationService.getUserRecommendations(data);
        if (type === "cbf") {
          setCBFResponse(results);
        } else {
          setCoBFResponse(results);
        }
      }
    } catch (error) {
      console.log(error);
    }
    if (type === "cbf") {
      setIsCBFLoading(false);
    } else {
      setIsCoBFLoading(false);
    }
  };

  useEffect(() => {
    fetchRecommendations("cbf");
    fetchRecommendations("cobf");
  }, []);

  return (
    <>
      <SearchBar />
      <div className="">
        <Recommendation
          type="cbf"
          isLoading={isCBFLoading}
          movies={cbfResponse}
        />
        <Recommendation
          type="cobf"
          isLoading={isCoBFLoading}
          movies={cobfResponse}
        />
      </div>
    </>
  );
}
