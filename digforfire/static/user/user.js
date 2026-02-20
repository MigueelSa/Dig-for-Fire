import { fetchLibrary, fetchHistory } from "./fetcher.js";
import { renderAlbums } from "./render.js ";

export async function getLibrary(containerId) {
  const data = await fetchLibrary();
  if (data && data.library) {
    renderAlbums(data.library, containerId);
  }
}

export async function getHistory(containerId) {
  const data = await fetchHistory();
  if (data && data.library) {
    renderAlbums(data.library, containerId);
  }
}
