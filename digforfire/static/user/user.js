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
  console.log("History data inside getHistory:", data);

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

    const data = await response.json();

    if (data.task_id) {
      pollProgress(data.task_id);
    }

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

export async function pollProgress(taskId) {
  const progressBar = document.getElementById("progress-bar");
  const statusText = document.getElementById("import-status");

  const interval = setInterval(async () => {
    const res = await fetch(`/progress/${taskId}`);
    if (!res.ok) return;

    const data = await res.json();
    progressBar.value = data.ratio * 100;

    let etaText = "";
    if (data.eta >= 0) {
      const minutes = Math.floor(data.eta / 60);
      const seconds = Math.floor(data.eta % 60);
      etaText = ` – ETA: ${minutes}m ${seconds}s`;
    }

    statusText.innerText = `Progress: ${progressBar.value.toFixed(1)}%${etaText}`;

    if (data.ratio >= 1) clearInterval(interval);
  }, 1000);
}
