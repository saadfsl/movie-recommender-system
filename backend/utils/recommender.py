import random
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import requests
import os

def load_model(model_path):
  current_dir = os.path.dirname(__file__)
  model_path = os.path.join(current_dir, model_path)
  model = TFBertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
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

def map_to_genre(sentiment, sentiment_to_genre, threshold=0.01):
  genres = set()
  for key, value in sentiment.items():
    if value >= threshold:
      genres.update(sentiment_to_genre[key])
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
