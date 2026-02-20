export function renderAlbums(albums) {
  const resultsList = document.getElementById("results");
  resultsList.innerHTML = "";
  albums.forEach((album) =>
    resultsList.appendChild(createAlbumListItem(album)),
  );
}

export function createAlbumListItem(album) {
  const li = document.createElement("li");

  if (album.cover_url) {
    const img = document.createElement("img");
    img.src = album.cover_url;
    img.alt = album.title + " cover";
    img.onerror = () => (img.style.display = "none");
    li.appendChild(img);
  }

  const info = document.createElement("div");
  info.innerHTML = `
                            <strong>${album.title}</strong><br>
                            Artist: ${album.artist.join(", ")}<br>
                            Genres: ${album.genres.join(", ")}<br>
                            Tags: ${album.tags.join(", ")}<br>
                            Score: ${album.score.toFixed(5)}<br>
                            <br>
                        `;

  li.appendChild(info);
  return li;
}
