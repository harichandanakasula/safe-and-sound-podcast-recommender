# Safe-and-Sound Podcast Recommender 🎧

A full-stack podcast recommendation system using **Spotify data**, **machine-learning ranking**, **safety filtering**, and a **React frontend** served by a **Flask REST API**.

This project demonstrates:
- Real-time ingestion from Spotify Web API  
- Content-based recommendation with embeddings + TF-IDF fallback  
- Safety scoring and filtering  
- Multi-signal ranking (similarity, recency, popularity, safety)  
- Diversity re-ranking using MMR  
- Full backend + frontend integration  

---

## Tech Stack

### Backend
- Python 3.12  
- Flask  
- Spotify Web API (spotipy)  
- scikit-learn  
- numpy, pandas  

### Frontend
- React (Vite)  
- JavaScript / JSX  

---


---

## Prerequisites

Install:

- Python 3.12  
- Node.js 18+  

Create a Spotify Developer account and get:

- `SPOTIFY_CLIENT_ID`  
- `SPOTIFY_CLIENT_SECRET`  

---

## Backend Setup

### 1. Create Virtual Environment

From repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate

2. Install Backend Dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

3. Create Environment Variables

Create a file named .env in the root folder:

SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret

Running the Backend
Start API Server
python api.py


Backend runs at:

http://127.0.0.1:5000

Test Backend Health

Open in browser:

http://127.0.0.1:5000/health


Expected output:

{ "status": "ok" }

Backend API Endpoints
Method	Endpoint	Description
GET	/health	Health check
POST	/users	Create new user
GET	/users/<user_id>	Get user info
PUT	/users/<user_id>/interests	Update user interests
GET	/recommendations/<user_id>	Get recommendations
POST	/feedback/<user_id>	Submit feedback
GET	/episodes	List all episodes
GET	/episodes/<episode_id>	Episode details
GET	/search?q=...	Search episodes
POST	/evaluate	Evaluate recommender
GET	/stats	System statistics
Running the Recommender Demo (Terminal)

This runs the full ML pipeline and prints recommendations.

python recommender.py


This will:

Load episodes from Spotify

Compute embeddings

Apply safety scoring

Rank and diversify episodes

Print top recommendations

Frontend Setup

Open a new terminal.

1. Go to Frontend Folder
cd frontend

2. Install Frontend Dependencies
npm install

3. Configure API URL

Create a file .env inside frontend/:

VITE_API_BASE_URL=http://127.0.0.1:5000

4. Start Frontend
npm run dev


Frontend runs at:

http://localhost:5173

Running Full System Together
Terminal 1 — Backend
cd safe_sound_recs
.\.venv\Scripts\activate
python api.py

Terminal 2 — Frontend
cd safe_sound_recs\podcast-ui
npm run dev
