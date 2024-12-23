# Movie Recommender System

This project is a movie recommender system that supports two modes: trained model-based recommendations and ChatGPT-assisted recommendations. Users can enter their mood to get tailored movie recommendations, provide feedback, and get additional recommendations.

## Features
- Two Recommendation Modes:
  - Trained Model: Uses a sentiment analysis model.
  - ChatGPT: Uses OpenAI's GPT-3.5 Turbo.
- Feedback System: User feedback updates the recommendation logic dynamically:
  - Trained Model: Uses reinforcement Q-learning to update actions.
  - ChatGPT: Feedback is added to chat context for future recommendations.
- Basic frontend for user interaction.

## Setup and Usage
1. Navigate to the project directory:
   ```bash
   cd path/to/movie-recommender-system
   ```
2. Create a virtual environment:
    ```bash
    python3 -m venv .venv
    ```
3. Activate the virtual environment:
    - Linux/Mac:
      ```bash
      source .venv/bin/activate
      ```
    - Windows:
      ```bash
      .venv\Scripts\activate
      ```
4. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
5. Set up the environment variables:
    - Linux/Mac:
      ```bash
      export OPENAI_API_KEY=<your-api-key>
      export TMDB_API_KEY=<your-api-key>
      ```
    - Windows (PowerShell):
      ```bash
      $env:OPENAI_API_KEY="<your-api-key>"
      $env:TMDB_API_KEY="<your-api-key>"
      ```
6. Run the Flask Server:
    ```bash
    python backend/app.py
    ```
7. Open the frontend/index.html file in a browser to interact with the app.
      

