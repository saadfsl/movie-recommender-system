const initialPage = document.getElementById("initial-page");
const recommendationsPage = document.getElementById("recommendations-page");
const recommendationsList = document.getElementById("recommendations-list");
const moodInput = document.getElementById("mood");

let selectedAction = null;

function showInitialPage() {
  initialPage.style.display = "block";
  recommendationsPage.style.display = "none";
}

function showRecommendationsPage() {
  initialPage.style.display = "none";
  recommendationsPage.style.display = "block";
}

document.getElementById("submit-mood").addEventListener("click", () => {
  const mood = moodInput.value.trim();
  if (!mood) {
    alert("Please enter a mood.");
    return;
  }

  fetch("http://localhost:5000/get-recommendations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: mood }),
  })
    .then((response) => response.json())
    .then((movies) => {
      displayRecommendations(movies);
      showRecommendationsPage();
    })
    .catch((error) => {
      console.error("Error:", error);
      alert("Failed to fetch recommendations.");
    });
});

function displayRecommendations(movies) {
  recommendationsList.innerHTML = "";
  if (movies.length === 0) {
    recommendationsList.innerHTML = "<li>No movies found.</li>";
    return;
  }

  movies.forEach((movie) => {
    const li = document.createElement("li");
    li.innerHTML = `<strong>${movie.title}</strong> (${movie.release_date})<br>${movie.overview}`;
    recommendationsList.appendChild(li);
  });

  document.getElementById("feedback-yes").style.display = "inline-block";
  document.getElementById("feedback-no").style.display = "inline-block";
  document.getElementById("more-recs").style.display = "none";
}

document.getElementById("feedback-yes").addEventListener("click", () => {
  sendFeedback("yes");
});
document.getElementById("feedback-no").addEventListener("click", () => {
  sendFeedback("no");
});

function sendFeedback(feedback) {
  fetch("http://localhost:5000/update-agent", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ feedback }),
  })
    .then(() => {
      document.getElementById("more-recs").style.display = "block";
    })
    .catch((error) => {
      console.error("Error:", error);
      alert("Failed to send feedback.");
    });
}

document.getElementById("get-more-recs").addEventListener("click", () => {
  fetch("http://localhost:5000/get-recommendations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  })
    .then((response) => response.json())
    .then((movies) => {
      displayRecommendations(movies);
    })
    .catch((error) => {
      console.error("Error:", error);
      alert("Failed to fetch more recommendations.");
    });
});

document.getElementById("reset").addEventListener("click", () => {
  fetch("http://localhost:5000/reset-agent", {
    method: "POST",
  })
    .then(() => {
      showInitialPage();
    })
    .catch((error) => {
      console.error("Error:", error);
      alert("Failed to reset.");
    });
});

showInitialPage();
