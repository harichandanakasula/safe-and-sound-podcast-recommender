# Safe-and-Sound Podcast Recommender 🎧

A production-style podcast recommendation system that integrates **Spotify data**, prioritizes **safety**, **fairness/diversity**, and **explainability**, and exposes recommendations through a **Flask REST API** with a **React frontend UI**.

This project demonstrates:
- Real-time ingestion from the **Spotify Web API**
- Content-based recommendation with embedding + TF-IDF fallback
- Safety scoring and filtering
- Multi-signal ranking (similarity, recency, popularity, safety)
- Diversity re-ranking using MMR
- A full backend + frontend system

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
- Fetch-based REST API integration

---

## Repository Structure

safe_sound_recs/
api.py
recommender.py
requirements.txt
.env.example
data/
models/
podcast-ui/
package.json
src/
vite.config.*

yaml
Copy code

---

## Prerequisites

- Python 3.12
- Node.js 18 or newer
- Spotify Developer Account

Create a Spotify app and obtain:
- SPOTIFY_CLIENT_ID
- SPOTIFY_CLIENT_SECRET

---

## Backend Setup

### 1. Create Virtual Environment

From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
2. Install Dependencies
powershell
Copy code
python -m pip install --upgrade pip
pip install -r requirements.txt
3. Create Environment Variables
Create a file named .env in the repository root:

env
Copy code
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
Running the Backend
Start API Server
powershell
Copy code
python api.py
Backend runs at:

http://127.0.0.1:5000

Test Health Endpoint
Open in browser:

arduino
Copy code
http://127.0.0.1:5000/health
Response:

json
Copy code
{ "status": "ok" }
Backend API Endpoints
Method	Endpoint	Description
GET	/health	Health check
POST	/users	Create new user
GET	/users/<user_id>	Get user info
PUT	/users/<user_id>/interests	Update interests
GET	/recommendations/<user_id>	Get recommendations
POST	/feedback/<user_id>	Submit feedback
GET	/episodes	List episodes
GET	/episodes/<episode_id>	Episode details
GET	/search?q=...	Search episodes
POST	/evaluate	Evaluate recommender
GET	/stats	System statistics

Running the Recommender Demo (CLI)
powershell
Copy code
python recommender.py
This:

Loads Spotify episodes

Builds embeddings

Computes safety scores

Runs ranking and diversity re-ranking

Prints top recommendations in terminal

Frontend Setup
Open a new terminal while backend is running.

1. Go to Frontend Folder
powershell
Copy code
cd podcast-ui
2. Install Frontend Dependencies
powershell
Copy code
npm install
3. Configure API URL
Create .env inside podcast-ui/:

env
Copy code
VITE_API_BASE_URL=http://127.0.0.1:5000
4. Start Frontend
powershell
Copy code
npm run dev
Frontend runs at:

http://localhost:5173

Running Full System Together
Terminal 1 — Backend
powershell
Copy code
cd safe_sound_recs
.\.venv\Scripts\activate
python api.py
Terminal 2 — Frontend
powershell
Copy code
cd safe_sound_recs\podcast-ui
npm install
npm run dev
Open:

http://localhost:5173

Recommendation Pipeline
Spotify episode ingestion

Text embedding (SentenceTransformer or TF-IDF fallback)

Safety scoring (toxicity + misinformation heuristic)

Candidate generation (cosine similarity)

Ranking using weighted signals

Diversity re-ranking using MMR

Explanation generation

Safety & Fairness
Safety scores computed per episode

Unsafe episodes filtered before ranking

Diversity enforced using Maximal Marginal Relevance

Small creators boosted by popularity normalization

Data Persistence
Users stored in: data/users.pkl

Episodes loaded dynamically from Spotify

Embeddings computed at runtime

Git Ignore (Required)
Your .gitignore must contain:

bash
Copy code
.venv/
.env
.cache/
__pycache__/
data/users.pkl
podcast-ui/node_modules/
podcast-ui/dist/
Viewing the Project
Backend API: http://127.0.0.1:5000/health

Frontend UI: http://localhost:5173

License
MIT License

markdown
Copy code

---

Hari — **this is now a real full-stack ML system**:

- Spotify API ✔  
- Recommender ✔  
- Safety + fairness ✔  
- Flask API ✔  
- React UI ✔  

This is **excellent portfolio material** for Spotify, ML roles, and backend + applied ML interviews.

If you want next, I can help you with:
- Final GitHub repo cleanup  
- Screenshots / demo instructions  
- Resume bullet points  
- How to describe this to Spotify recruiters  
- Deploying this on Render / Fly.io / AWS
