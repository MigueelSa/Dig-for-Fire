import { fetchRecommendations } from "./fetcher.js";
import { renderRecommendations } from "./render.js";

async function getRecommendations() {

    const button = document.getElementById("get-recommendations-btn");
    const loader = document.querySelector(".loader");

    button.disabled = true;
    loader.style.display = "inline-grid";

    const data = await fetchRecommendations();
    if (data && data.recommendations) {
        renderRecommendations(data.recommendations);
    }

    loader.style.display = "none";
    button.disabled = false;
}

document.getElementById("get-recommendations-btn").addEventListener("click", getRecommendations);