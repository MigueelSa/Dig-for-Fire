import { fetchLibrary, fetchHistory } from "./fetcher.js";
import { renderAlbums } from "./render.js";

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

export async function enrichLibrary(file) {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/enrich-library", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const text = await response.text();
      return `Error: ${text}`;
    }

    return "Enrichment started successfully!";
  } catch (err) {
    console.error(err);
    return "Error uploading library.";
  }
}

export async function importLibrary() {
  var fileInput = document.getElementById("library-file");
  if (!fileInput.files.length) {
    alert("Select a file first!");
    return;
  }

  var file = fileInput.files[0];

  try {
    var statusText = await enrichLibrary(file);
    document.getElementById("import-status").innerText = statusText;
  } catch (err) {
    console.error(err);
    document.getElementById("import-status").innerText =
      "Error uploading library.";
  }
}
