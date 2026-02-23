import { fetchRecommendations } from "./fetcher.js";
import { renderAlbums } from "./render.js";

export async function getRecommendations() {
  const button = document.getElementById("get-recommendations-btn");
  const loader = document.querySelector(".loader");

  button.disabled = true;
  loader.style.display = "inline-grid";

  const data = await fetchRecommendations();
  if (data && data.library) {
    renderAlbums(data.library);
  }

  loader.style.display = "none";
  button.disabled = false;
}
