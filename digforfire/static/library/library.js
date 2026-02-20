import { fetchLibrary } from "./fetcher.js";
import { renderAlbums } from "./render.js ";

export async function getLibrary() {
  const data = await fetchLibrary();
  if (data && data.library) {
    renderAlbums(data.library);
  }
}
