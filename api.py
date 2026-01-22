"""
Flask API for Safe-and-Sound Podcast Recommender
Provides REST endpoints for recommendations, feedback, and evaluation
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from recommender import (
    SafeAndSoundRecommender, User, Episode, Recommendation,
    RecommenderEvaluator
)
from typing import Dict, List
import pickle
import os
from datetime import datetime
import uuid

import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS - comes AFTER creating app

# Global recommender instance
recommender = None
users_db: Dict[str, User] = {}  # In-memory user storage
DATA_DIR = "data"
USERS_FILE = os.path.join(DATA_DIR, "users.pkl")


# INITIALIZATION


def init_recommender():
    """Initialize recommender system"""
    global recommender
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    print("Initializing recommender system...")
    recommender = SafeAndSoundRecommender()
    recommender.load_data()
    print("Recommender ready!")
    
    # Load existing users
    load_users()

def save_users():
    """Save users to disk"""
    with open(USERS_FILE, 'wb') as f:
        pickle.dump(users_db, f)

def load_users():
    """Load users from disk"""
    global users_db
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'rb') as f:
            users_db = pickle.load(f)
        print(f"Loaded {len(users_db)} users from disk")


# HELPER FUNCTIONS


def episode_to_dict(episode: Episode) -> dict:
    """Convert Episode to JSON-serializable dict"""
    return {
        'id': episode.id,
        'title': episode.title,
        'summary': episode.summary[:500],  # Truncate for API
        'podcast_name': episode.podcast_name,
        'published_date': episode.published_date.isoformat(),
        'link': episode.link,
        'categories': episode.categories,
        'popularity_score': float(episode.popularity_score),
        'safety_scores': episode.safety_scores
    }

def recommendation_to_dict(rec: Recommendation) -> dict:
    """Convert Recommendation to JSON-serializable dict"""
    return {
        'episode': episode_to_dict(rec.episode),
        'score': float(rec.score),
        'explanation': rec.explanation,
        'components': {k: float(v) for k, v in rec.components.items()}
    }

def user_to_dict(user: User) -> dict:
    """Convert User to JSON-serializable dict"""
    return {
        'id': user.id,
        'interests': user.interests,
        'interaction_count': len(user.interaction_history)
    }


# API ENDPOINTS


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'recommender_loaded': recommender is not None,
        'episodes_count': len(recommender.episodes) if recommender else 0,
        'users_count': len(users_db)
    })

@app.route('/users', methods=['POST'])
def create_user():
    """
    Create a new user
    
    Body:
    {
        "interests": ["technology", "science", "business"]
    }
    """
    data = request.get_json()
    
    if not data or 'interests' not in data:
        return jsonify({'error': 'interests field is required'}), 400
    
    interests = data['interests']
    if not isinstance(interests, list) or len(interests) == 0:
        return jsonify({'error': 'interests must be a non-empty list'}), 400
    
    # Create user
    user_id = str(uuid.uuid4())
    user = User(
        id=user_id,
        interests=interests,
        interaction_history=[],
        profile_embedding=None
    )
    
    users_db[user_id] = user
    save_users()
    
    return jsonify({
        'user': user_to_dict(user),
        'message': 'User created successfully'
    }), 201

@app.route('/users/<user_id>', methods=['GET'])
def get_user(user_id: str):
    """Get user information"""
    if user_id not in users_db:
        return jsonify({'error': 'User not found'}), 404
    
    user = users_db[user_id]
    return jsonify({'user': user_to_dict(user)})

@app.route('/users/<user_id>/interests', methods=['PUT'])
def update_interests(user_id: str):
    """
    Update user interests
    
    Body:
    {
        "interests": ["new", "interests"]
    }
    """
    if user_id not in users_db:
        return jsonify({'error': 'User not found'}), 404
    
    data = request.get_json()
    if not data or 'interests' not in data:
        return jsonify({'error': 'interests field is required'}), 400
    
    user = users_db[user_id]
    user.interests = data['interests']
    save_users()
    
    return jsonify({
        'user': user_to_dict(user),
        'message': 'Interests updated successfully'
    })

@app.route('/recommendations/<user_id>', methods=['GET'])
def get_recommendations(user_id: str):
    """
    Get recommendations for user
    
    Query params:
    - top_k: number of recommendations (default: 10)
    - safety_threshold: safety threshold 0-1 (default: 0.7)
    - diversity_lambda: diversity vs relevance 0-1 (default: 0.7)
    """
    if user_id not in users_db:
        return jsonify({'error': 'User not found'}), 404
    
    if not recommender:
        return jsonify({'error': 'Recommender not initialized'}), 503
    
    # Parse query params
    top_k = int(request.args.get('top_k', 10))
    safety_threshold = float(request.args.get('safety_threshold', 0.7))
    diversity_lambda = float(request.args.get('diversity_lambda', 0.7))
    
    # Validate params
    if top_k < 1 or top_k > 50:
        return jsonify({'error': 'top_k must be between 1 and 50'}), 400
    if not 0 <= safety_threshold <= 1:
        return jsonify({'error': 'safety_threshold must be between 0 and 1'}), 400
    if not 0 <= diversity_lambda <= 1:
        return jsonify({'error': 'diversity_lambda must be between 0 and 1'}), 400
    
    user = users_db[user_id]
    
    # Get recommendations
    try:
        recommendations = recommender.recommend(
            user,
            top_k=top_k,
            safety_threshold=safety_threshold,
            diversity_lambda=diversity_lambda
        )
        
        return jsonify({
            'user_id': user_id,
            'recommendations': [recommendation_to_dict(rec) for rec in recommendations],
            'count': len(recommendations),
            'parameters': {
                'top_k': top_k,
                'safety_threshold': safety_threshold,
                'diversity_lambda': diversity_lambda
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Error generating recommendations: {str(e)}'}), 500

@app.route('/feedback/<user_id>', methods=['POST'])
def submit_feedback(user_id: str):
    """
    Submit user feedback on an episode
    
    Body:
    {
        "episode_id": "abc123",
        "action": "like" | "dislike" | "skip"
    }
    """
    if user_id not in users_db:
        return jsonify({'error': 'User not found'}), 404
    
    data = request.get_json()
    if not data or 'episode_id' not in data or 'action' not in data:
        return jsonify({'error': 'episode_id and action fields are required'}), 400
    
    episode_id = data['episode_id']
    action = data['action']
    
    if action not in ['like', 'dislike', 'skip']:
        return jsonify({'error': 'action must be like, dislike, or skip'}), 400
    
    # Add interaction to user history
    user = users_db[user_id]
    timestamp = datetime.now().timestamp()
    user.interaction_history.append((episode_id, action, timestamp))
    save_users()
    
    return jsonify({
        'message': 'Feedback recorded successfully',
        'user_id': user_id,
        'episode_id': episode_id,
        'action': action,
        'total_interactions': len(user.interaction_history)
    })

@app.route('/episodes', methods=['GET'])
def list_episodes():
    """
    List all available episodes
    
    Query params:
    - limit: max episodes to return (default: 20)
    - offset: offset for pagination (default: 0)
    - podcast: filter by podcast name
    """
    if not recommender:
        return jsonify({'error': 'Recommender not initialized'}), 503
    
    limit = int(request.args.get('limit', 20))
    offset = int(request.args.get('offset', 0))
    podcast_filter = request.args.get('podcast')
    
    episodes = recommender.episodes
    
    # Filter by podcast if specified
    if podcast_filter:
        episodes = [ep for ep in episodes if podcast_filter.lower() in ep.podcast_name.lower()]
    
    # Paginate
    total = len(episodes)
    episodes = episodes[offset:offset + limit]
    
    return jsonify({
        'episodes': [episode_to_dict(ep) for ep in episodes],
        'total': total,
        'limit': limit,
        'offset': offset
    })

@app.route('/episodes/<episode_id>', methods=['GET'])
def get_episode(episode_id: str):
    """Get detailed information about a specific episode"""
    if not recommender:
        return jsonify({'error': 'Recommender not initialized'}), 503
    
    # Find episode
    episode = None
    for ep in recommender.episodes:
        if ep.id == episode_id:
            episode = ep
            break
    
    if not episode:
        return jsonify({'error': 'Episode not found'}), 404
    
    return jsonify({'episode': episode_to_dict(episode)})

@app.route('/evaluate', methods=['POST'])
def evaluate_recommendations():
    """
    Evaluate recommendation quality for a user
    
    Body:
    {
        "user_id": "abc123",
        "relevant_episode_ids": ["ep1", "ep2", "ep3"]  // Ground truth
    }
    """
    data = request.get_json()
    
    if not data or 'user_id' not in data or 'relevant_episode_ids' not in data:
        return jsonify({'error': 'user_id and relevant_episode_ids required'}), 400
    
    user_id = data['user_id']
    relevant_ids = data['relevant_episode_ids']
    
    if user_id not in users_db:
        return jsonify({'error': 'User not found'}), 404
    
    if not recommender:
        return jsonify({'error': 'Recommender not initialized'}), 503
    
    user = users_db[user_id]
    
    # Get recommendations
    recommendations = recommender.recommend(user, top_k=10)
    
    # Compute metrics
    evaluator = RecommenderEvaluator()
    ndcg = evaluator.ndcg_at_k(recommendations, relevant_ids, k=10)
    diversity = evaluator.diversity_score(recommendations)
    
    return jsonify({
        'user_id': user_id,
        'metrics': {
            'ndcg@10': float(ndcg),
            'diversity': float(diversity)
        },
        'recommendations_count': len(recommendations),
        'relevant_count': len(relevant_ids)
    })

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    if not recommender:
        return jsonify({'error': 'Recommender not initialized'}), 503
    
    # Compute statistics
    total_episodes = len(recommender.episodes)
    total_users = len(users_db)
    
    # Podcast distribution
    podcast_counts = {}
    for ep in recommender.episodes:
        podcast_counts[ep.podcast_name] = podcast_counts.get(ep.podcast_name, 0) + 1
    
    # User interaction stats
    total_interactions = sum(len(user.interaction_history) for user in users_db.values())
    
    # Safety stats
    safe_episodes = sum(1 for ep in recommender.episodes 
                       if ep.safety_scores and ep.safety_scores['overall_safety'] >= 0.7)
    
    return jsonify({
        'total_episodes': total_episodes,
        'total_users': total_users,
        'total_interactions': total_interactions,
        'podcast_distribution': podcast_counts,
        'safety_stats': {
            'safe_episodes': safe_episodes,
            'unsafe_episodes': total_episodes - safe_episodes,
            'safety_rate': safe_episodes / total_episodes if total_episodes > 0 else 0
        }
    })

@app.route('/search', methods=['GET'])
def search_episodes():
    """
    Search episodes by keywords
    
    Query params:
    - q: search query
    - limit: max results (default: 10)
    """
    if not recommender:
        return jsonify({'error': 'Recommender not initialized'}), 503
    
    query = request.args.get('q', '').lower()
    limit = int(request.args.get('limit', 10))
    
    if not query:
        return jsonify({'error': 'q parameter is required'}), 400
    
    # Simple keyword search
    results = []
    for ep in recommender.episodes:
        if (query in ep.title.lower() or 
            query in ep.summary.lower() or 
            query in ep.podcast_name.lower()):
            results.append(ep)
            
            if len(results) >= limit:
                break
    
    return jsonify({
        'query': query,
        'results': [episode_to_dict(ep) for ep in results],
        'count': len(results)
    })


# ERROR HANDLERS


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# MAIN


if __name__ == '__main__':
    print("Starting Safe-and-Sound API Server...")
    print("="*80)
    
    # Initialize recommender
    init_recommender()
    
    print("\n" + "="*80)
    print("API ENDPOINTS:")
    print("="*80)
    print("POST   /users                    - Create new user")
    print("GET    /users/<user_id>          - Get user info")
    print("PUT    /users/<user_id>/interests - Update interests")
    print("GET    /recommendations/<user_id> - Get recommendations")
    print("POST   /feedback/<user_id>       - Submit feedback")
    print("GET    /episodes                 - List episodes")
    print("GET    /episodes/<episode_id>    - Get episode details")
    print("GET    /search                   - Search episodes")
    print("POST   /evaluate                 - Evaluate recommendations")
    print("GET    /stats                    - System statistics")
    print("GET    /health                   - Health check")
    print("="*80)
    
    # Run server
    app.run(debug=True, host='0.0.0.0', port=5000)