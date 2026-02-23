import { pollProgress } from "/static/user/user.js";

export async function saveSpotifyCredentials(form) {
  const formData = new FormData(form);

  const res = await fetch("/user/spotify/save-credentials", {
    method: "POST",
    body: JSON.stringify({
      client_id: formData.get("client_id"),
      client_secret: formData.get("client_secret"),
      redirect_uri: formData.get("redirect_uri"),
    }),
    headers: { "Content-Type": "application/json" },
  });
}

export async function checkSpotifyCredentials() {
  const res = await fetch("/user/spotify/credentials-check");
  const credData = await res.json();
  if (!credData.ok) {
    window.location.href = "/user/spotify/setup";
  } else {
    const loader = document.querySelector(".loader");
    loader.style.display = "inline-grid";
    const importation = await fetch("/user/spotify/import-library");
    loader.style.display = "none";
  }
}
