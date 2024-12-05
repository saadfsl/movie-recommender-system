import random
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import requests
import numpy as np
import itertools

class QAgent:
  def __init__(self, genres, release_years, rating_ranges, popularities):
    self.actions = [
      (genre_combo, release_year, rating_range, popularity)
      for genre_combo in genres
      for release_year in release_years
      for rating_range in rating_ranges
      for popularity in popularities
    ]
    self.q_table = {action: 0 for action in self.actions}
    self.learning_rate = 0.1 # alpha
    self.discount_factor = 0.9 # gamma
    self.exploration_rate = 0.2 # epsilon

  def get_action(self):
    if random.random() < self.exploration_rate:
      return random.choice(self.actions)
    else:
      return max(self.q_table, key=self.q_table.get)
    
  def update_q_value(self, action, reward):
    old = self.q_table[action]
    # immediate reward only (short-term)
    self.q_table[action] = old + self.learning_rate * (reward - old)

def load_model(model_path):
  model = TFBertForSequenceClassification.from_pretrained(model_path)
  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
  return model, tokenizer

def predict_sentiment(text, model, tokenizer, sentiment_keys):
  tokens = tokenizer(
    text,
    max_length=128,
    padding="max_length",
    truncation=True,
    return_tensors="tf"
  )
  input_ids = tokens["input_ids"]
  attention_mask = tokens["attention_mask"]
  outputs = model(input_ids, attention_mask=attention_mask)
  logits = outputs.logits
  probabilities = tf.nn.sigmoid(logits)[0].numpy()
  sentiment = {key: probabilities[i] for i, key in enumerate(sentiment_keys)}
  return sentiment

SENTIMENT_TO_GENRE = {
  "admiration": [14], # Fantasy
  "amusement": [35], # Comedy
  "anger": [53], # Thriller
  "annoyance": [10749], # Romance
  "approval": [18], # Drama
  "caring": [10751], # Family
  "confusion": [12], # Adventure
  "curiosity": [9648], # Mystery
  "desire": [10749], # Romance
  "disappointment": [18], # Drama
  "disapproval": [18], # Drama
  "disgust": [27], # Horror
  "embarrassment": [35], # Comedy
  "excitement": [28], # Action
  "fear": [27], # Horror
  "gratitude": [18], # Drama
  "grief": [18], # Drama
  "joy": [35], # Comedy
  "love": [10749], # Romance
  "nervousness": [53], # Thriller
  "optimism": [14], # Fantasy
  "pride": [14], # Fantasy
  "realization": [18], # Drama
  "relief": [18], # Drama
  "remorse": [18], # Drama
  "sadness": [18], # Drama
  "surprise": [9648], # Mystery
  "neutral": [18] # Drama
}

def map_to_genre(sentiment, threshold=0.01):
  genres = set()
  for key, value in sentiment.items():
    if value >= threshold:
      genres.update(SENTIMENT_TO_GENRE[key])
  return list(genres)

def get_movie_recommendations(action, api_key, current_page, num_movies=5):
  genre_combo, release_year, rating_range, popularity = action

  release_year_filter = {"primary_release_date.gte": release_year[0], "primary_release_date.lte": release_year[1]}
  rating_filter = {"vote_average.gte": rating_range[0], "vote_average.lte": rating_range[1]}

  url = "https://api.themoviedb.org/3/discover/movie"
  params = {
    "api_key": api_key,
    "language": "en-US",
    "sort_by": "popularity.desc",
    "with_genres": ",".join(str(genre) for genre in genre_combo),
    "vote_count.gte": popularity[0],
    **release_year_filter,
    **rating_filter,
    "page": current_page,
  }

  response = requests.get(url, params=params)

  if response.status_code == 200:
    movies = response.json()["results"]
    random.shuffle(movies)
    return movies[:num_movies]
  else:
    print("Failed to fetch movie recommendations: ", response.status_code)
    return []

if __name__ == "__main__":
  model_path = "models/sentiment_model"
  model, tokenizer = load_model(model_path)
  sentiment_keys = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]

  tmdb_api_key = "REMOVED" # replace with your own API key
  text = input("Describe your current mood: ")

  sentiment = predict_sentiment(text, model, tokenizer, sentiment_keys)
  genres = map_to_genre(sentiment)
  print("Initial Genres: ", genres)

  release_years = [
    ("1900-01-01", "2000-12-31"),
    ("2000-01-01", "2015-12-31"),
    ("2015-01-01", "2024-12-31")
  ]
  rating_ranges = [(0, 5), (5, 7.5), (7.5, 10)]
  popularities = [(0, 100), (100, 1000), (1000, 10000), (10000, float("inf"))]

  genre_combinations = []
  for i in range(1, len(genres) + 1):
    genre_combinations.extend(list(itertools.combinations(genres, i)))
  print("Possible Genre Combinations: ", genre_combinations)

  # RL agent
  agent = QAgent(genre_combinations, release_years, rating_ranges, popularities)

  movie_pool = []
  current_page = 1
  previous_action = None

  session_stats = {
    "genres": {}, "release_years": {}, "ratings": {}, "popularity": {}
  }

  while True:
    selected_action = agent.get_action()
    print(f"Selected Action: {selected_action}")
    print()

    if selected_action != previous_action:
      movie_pool = []
      current_page = 1

    recs = get_movie_recommendations(selected_action, tmdb_api_key, current_page)
    current_page += 1

    print("Recommended Movies:")
    for rec in recs:
      print(f"{rec['title']} ({rec['release_date'][:4]})")
      print(f"Rating: {rec['vote_average']}")
      print(rec["overview"])
      print()

    # get feedback
    feedback = input("Did you like the recommendations? (yes/no): ").strip().lower()
    if feedback == "yes":
      reward = 1
    else:
      reward = -1

    # update q-value
    agent.update_q_value(selected_action, reward)

    # update session stats
    if feedback == "yes":
      for i, attr in enumerate(["genres", "release_years", "ratings", "popularity"]):
        value = selected_action[i]
        if isinstance(value, list):
          for v in value:
            session_stats[attr][v] = session_stats[attr].get(v, 0) + 1
        else:
          session_stats[attr][value] = session_stats[attr].get(value, 0) + 1

    cont = input("Do you want another recommendation? (yes/no): ").strip().lower()
    if cont != "yes":
      break

  # print summary
  print("\nSession Summary:")
  for attr, values in session_stats.items():
    if values:
      top_value = max(values, key=values.get)
      print(f"Top {attr}: {top_value} ({values[top_value]} selections)")
