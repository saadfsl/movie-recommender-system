from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.recommender import load_model, predict_sentiment, map_to_genre, get_movie_recommendations
from utils.q_agent import QAgent
import itertools
import os

app = Flask(__name__)
CORS(app)

app.config["TMDB_API_KEY"] = os.getenv("TMDB_API_KEY") # set your TMDB API key as an environment variable
if not app.config["TMDB_API_KEY"]:
  raise ValueError("TMDB_API_KEY environment variable not set")

model_path = "../models/sentiment_model"
model, tokenizer = load_model(model_path)

agent = None
current_page = 1
selected_action = None
previous_action = None

SENTIMENT_KEYS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

@app.route("/get-recommendations", methods=["POST"])
def get_recommendations_api():
  data = request.json
  
  global agent, current_page, selected_action, previous_action
  if not agent:
    if "text" not in data or not isinstance(data["text"], str):
      return jsonify({"error": "Invalid input. 'text' field is required."}), 400
    text = data["text"]
    sentiment = predict_sentiment(text, model, tokenizer, SENTIMENT_KEYS)
    print("Sentiment: ", [(key, value) for key, value in sentiment.items() if value >= 0.05])
    genres = map_to_genre(sentiment, threshold=0.05)
    print("Initial Genres: ", genres)
    genre_combinations = []
    for i in range(1, len(genres) + 1):
      genre_combinations.extend(list(itertools.combinations(genres, i)))
    print("Possible Genre Combinations: ", genre_combinations)
    agent = QAgent(
      genre_combinations,
      [
        ("2015-01-01", "2024-12-31"),
        ("2000-01-01", "2015-12-31"),
        ("1900-01-01", "2000-12-31")
      ],
      [(7.5, 10), (5, 7.5), (0, 5)],
      [(10000, float("inf")), (1000, 10000), (100, 1000), (0, 100)]
    )

  selected_action = agent.get_action()
  if selected_action != previous_action:
    current_page = 1
    previous_action = selected_action
  print(f"Selected Action: {selected_action}")

  recs = get_movie_recommendations(selected_action, app.config["TMDB_API_KEY"], current_page)
  if not recs:
    current_page = 1  
    recs = get_movie_recommendations(selected_action, app.config["TMDB_API_KEY"], current_page)
  else:
    current_page += 1

  return jsonify(recs)

@app.route("/update-agent", methods=["POST"])
def update_agent():
  data = request.json

  global agent, selected_action
  if not agent or not selected_action:
    return jsonify({"message": "Agent or selected action not initialized"})

  feedback = data["feedback"]
  if feedback == "yes":
    reward = 1
  else:
    reward = -1
  agent.update_q_value(selected_action, reward)
  return jsonify({"message": "Agent has been updated"})

@app.route("/reset-agent", methods=["POST"])
def reset_agent():
  global agent, current_page, selected_action, previous_action
  agent = None
  current_page = 1
  selected_action = None
  previous_action = None
  return jsonify({"message": "Agent has been reset"})

if __name__ == "__main__":
  app.run(debug=True)
