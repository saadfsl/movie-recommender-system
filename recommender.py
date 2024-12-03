import random
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import requests

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

def get_movie_recommendations(genres, api_key, num_movies=5):
  url = "https://api.themoviedb.org/3/discover/movie"
  params = {
    "api_key": api_key,
    "language": "en-US",
    "sort_by": "popularity.desc",
    "with_genres": ",".join(str(genre) for genre in genres)
  }

  response = requests.get(url, params=params)

  if response.status_code == 200:
    movies = response.json()["results"][:20]
    random.shuffle(movies)
    selected_movies = movies[:num_movies]
    return selected_movies
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
  print
  # print as a readable table
  print("{:<15} | {}".format("Sentiment", "Probability"))
  print("-" * 30)
  for key, value in sentiment.items():
    if value >= 0.01:
      print("{:<15} | {:.2f}".format(key, value))
  print()

  genres = map_to_genre(sentiment)
  print("Genres: ", genres)
  print()

  recs = get_movie_recommendations(genres, tmdb_api_key)
  print("Recommended Movies:")
  for rec in recs:
    print(f"{rec['title']} ({rec['release_date'][:4]})")
    print(rec["overview"])
    print()


