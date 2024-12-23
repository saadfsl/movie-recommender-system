const initialPage = document.getElementById("initial-page");
const recommendationsPage = document.getElementById("recommendations-page");
const recommendationsList = document.getElementById("recommendations-list");
const moodInput = document.getElementById("mood");
const modelChoice = document.getElementById("model-choice");

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
  const selectedModel = modelChoice.value;
  if (!mood) {
    alert("Please enter a mood.");
    return;
  }

  fetch("http://localhost:5000/get-recommendations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text: mood,
      use_chatgpt: selectedModel === "chatgpt",
    }),
  })
    .then((response) => response.json())
    .then((response) => {
      if (response.recommendations) {
        displayRecommendations(response.recommendations);
      } else if (Array.isArray(response)) {
        displayRecommendations(response);
      } else {
        alert("Failed to fetch recommendations. Please try again.");
      }
      showRecommendationsPage();
    })
    .catch((error) => {
      console.error("Error:", error);
      alert("Failed to fetch recommendations.");
    });
});

function displayRecommendations(recommendations) {
  recommendationsList.innerHTML = "";

  if (typeof recommendations === "string") {
    const li = document.createElement("li");
    li.textContent = recommendations;
    recommendationsList.appendChild(li);
  } else if (Array.isArray(recommendations) && recommendations.length === 0) {
    recommendationsList.innerHTML = "<li>No movies found.</li>";
  } else {
    recommendations.forEach((movie) => {
      const li = document.createElement("li");
      li.innerHTML = `<strong>${movie.title}</strong> (${movie.release_date})<br>${movie.overview}`;
      recommendationsList.appendChild(li);
    });
  }

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
    body: JSON.stringify({
      use_chatgpt: modelChoice.value === "chatgpt",
    }),
  })
    .then((response) => response.json())
    .then((response) => {
      if (response.recommendations) {
        displayRecommendations(response.recommendations);
      } else if (Array.isArray(response)) {
        displayRecommendations(response);
      } else {
        alert("Failed to fetch recommendations. Please try again.");
      }
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
