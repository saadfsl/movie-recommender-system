import random

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