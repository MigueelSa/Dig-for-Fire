import { fetchRecommendations } from "./fetcher.js";
import { renderRecommendations } from "./render.js";

async function getRecommendations() {
    const data = await fetchRecommendations();
    if (data && data.recommendations) {
        renderRecommendations(data.recommendations);
    }
}

document.getElementById("get-recommendations-btn").addEventListener("click", getRecommendations);