import itertools
import os
from backend.utils.q_agent import QAgent
from backend.utils.recommender import load_model, predict_sentiment, map_to_genre, get_movie_recommendations

if __name__ == "__main__":
  model_path = "../models/sentiment_model"
  model, tokenizer = load_model(model_path)
  sentiment_keys = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]

  tmdb_api_key = os.getenv("TMDB_API_KEY") # set your TMDB API key as an environment variable
  text = input("Describe your current mood: ")

  sentiment = predict_sentiment(text, model, tokenizer, sentiment_keys)
  print("Sentiment: ", [(key, value) for key, value in sentiment.items() if value >= 0.05])
  genres = map_to_genre(sentiment, threshold=0.05)
  print("Initial Genres: ", genres)

  release_years = [
    ("2015-01-01", "2024-12-31"),
    ("2000-01-01", "2015-12-31"),
    ("1900-01-01", "2000-12-31")
  ]
  rating_ranges = [(7.5, 10), (5, 7.5), (0, 5)]
  popularities = [(10000, float("inf")), (1000, 10000), (100, 1000), (0, 100)]

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
      