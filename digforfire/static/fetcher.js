export async function fetchRecommendations() {
    try {
        const response = await fetch("/recommend");
        // response.body is a JSON string from the server
        return await response.json();
        // JSON string → JS object (or array)
    } catch (error) {
        console.error("Error fetching recommendations:", error);
        return null;
    }
}