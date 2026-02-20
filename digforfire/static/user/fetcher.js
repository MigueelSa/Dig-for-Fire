export async function fetchLibrary() {
  try {
    const response = await fetch("/library");
    const result = await response.json();
    return result;
  } catch (error) {
    console.error("Error fetching library:", error);
    return null;
  }
}

export async function fetchHistory() {
  try {
    const response = await fetch("/history");
    const result = await response.json();
    return result;
  } catch (error) {
    console.error("Error fetching history:", error);
    return null;
  }
}
