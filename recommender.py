import random
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import requests
import numpy as np
import itertools

class QAgent:
  def __init__(self, genres):
    self.genres = genres
    self.q_table = {tuple(action): 0.0 for action in genres}
    self.learning_rate = 0.1 # alpha
    self.discount_factor = 0.9 # gamma
    self.exploration_rate = 0.2 # epsilon

  def get_action(self):
    if random.random() < self.exploration_rate:
      return random.choice(self.genres)
    else:
      return max(self.q_table, key=self.q_table.get)
    
  def update_q_value(self, genre, reward):
    old = self.q_table[genre]
    # immediate reward only (short-term)
    self.q_table[genre] = old + self.learning_rate * (reward - old)

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

def get_movie_recommendations(genres, api_key, pool, current_page, num_movies=5):
  if len(pool) < num_movies:
    url = "https://api.themoviedb.org/3/discover/movie"
    params = {
      "api_key": api_key,
      "language": "en-US",
      "sort_by": "popularity.desc",
      "with_genres": ",".join(str(genre) for genre in genres),
      "page": current_page
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
      movies = response.json()["results"]
      random.shuffle(movies)
      pool.extend(movies)
    else:
      print("Failed to fetch movie recommendations: ", response.status_code)

  # return next batch of movies and remove them from the pool
  next_batch = pool[:num_movies]
  del pool[:num_movies]
  return next_batch
    
if __name__ == "__main__":
  model_path = "models/sentiment_model"
  model, tokenizer = load_model(model_path)
  sentiment_keys = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]

  tmdb_api_key = "REMOVED" # replace with your own API key
  text = input("Describe your current mood: ")

  sentiment = predict_sentiment(text, model, tokenizer, sentiment_keys)
  genres = map_to_genre(sentiment)
  print("Initial Genres: ", genres)

  genre_combinations = []
  for i in range(1, len(genres) + 1):
    genre_combinations.extend(list(itertools.combinations(genres, i)))
  print("Possible Genre Combinations: ", genre_combinations)

  # RL agent
  agent = QAgent(genre_combinations)

  movie_pool = []
  current_page = 1

  while True:
    selected_combo = agent.get_action()
    print(f"Selected Genre: {selected_combo}")

    recs = get_movie_recommendations(selected_combo, tmdb_api_key, movie_pool, current_page)
    current_page += 1
    print("Recommended Movies:")
    for i, rec in enumerate(recs):
      print(f"{i + 1}. {rec['title']} ({rec['release_date'][:4]})")
      print(rec["overview"])
      print()

    # get feedback
    feedback = input("Did you like the recommendations? (yes/no): ").strip().lower()
    if feedback == "yes":
      reward = 1
    else:
      reward = -1

    # update q-value
    agent.update_q_value(selected_combo, reward)

    cont = input("Do you want to see more movies from the same genres? (yes/no): ").strip().lower()
    if cont != "yes":
      # reset movie pool and current page
      movie_pool = []
      current_page = 1
      cont = input("Do you want a new recommendation? (yes/no): ").strip().lower()
      if cont != "yes":
        break


