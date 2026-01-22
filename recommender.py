"""
Safe-and-Sound Podcast Recommender - Spotify Integration
A production-grade recommendation system with safety, fairness, and explainability
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import pickle
import hashlib
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Episode:
    """Represents a podcast episode"""
    id: str
    title: str
    summary: str
    podcast_name: str
    published_date: datetime
    link: str
    categories: List[str]
    embedding: Optional[np.ndarray] = None
    popularity_score: float = 0.0
    safety_scores: Optional[Dict[str, float]] = None
    
    def __hash__(self):
        return hash(self.id)

@dataclass
class User:
    """Represents a user with preferences and history"""
    id: str
    interests: List[str]
    interaction_history: List[Tuple[str, str, float]]  # (episode_id, action, timestamp)
    profile_embedding: Optional[np.ndarray] = None
    
@dataclass
class Recommendation:
    """Represents a recommendation with explanation"""
    episode: Episode
    score: float
    explanation: str
    components: Dict[str, float]  # Score breakdown


# ============================================================================
# SPOTIFY DATA INGESTION
# ============================================================================

class SpotifyPodcastLoader:
    """Loads podcast episodes from Spotify API"""
    
    # Popular Spotify podcast show IDs
    SPOTIFY_SHOWS = [
        '4rOoJ6Egrf8K2IrywzwOMk',  # The Joe Rogan Experience
        '2MAi0BvDc6GTFvKFPXnkCL',  # Lex Fridman Podcast
        '5CnDmMUG0S5bSSw612fs8C',  # Huberman Lab
        '4yAEKfvoAPzqWYVinfOJ4Q',  # The Tim Ferriss Show
        '0ofXAdFIQQRsCYj9754UFx',  # TED Talks Daily
        '1VXcH8QHkjRcTCEd88U3ti',  # Stuff You Should Know
        '6kAsbP8pxwaU2kPibKTuHE',  # Crime Junkie
        '2t7lZWs320KyXjqWz7PBvQ',  # Call Her Daddy
        '1OLcQdw2PFDPG1jo3s0wbp',  # SmartLess
        '4V3K3zyD0k789eaSWFXzhc',  # Huberman Lab (backup)
    ]
    
    def __init__(self, client_id: str, client_secret: str):
        """Initialize Spotify client"""
        auth_manager = SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret
        )
        self.sp = spotipy.Spotify(auth_manager=auth_manager)
    
    def load_episodes(self, episodes_per_show: int = 10) -> List[Episode]:
        """Load episodes from Spotify shows"""
        all_episodes = []
        
        for show_id in self.SPOTIFY_SHOWS:
            try:
                # Get show info
                show = self.sp.show(show_id)
                show_name = show['name']
                show_popularity = show.get('popularity', 50) / 100.0
                
                # Get episodes
                results = self.sp.show_episodes(show_id, limit=episodes_per_show)
                episodes = results['items']
                
                for ep in episodes:
                    try:
                        # Parse release date
                        release_date_str = ep.get('release_date', '')
                        if len(release_date_str) == 10:  # YYYY-MM-DD
                            published_date = datetime.strptime(release_date_str, '%Y-%m-%d')
                        elif len(release_date_str) == 7:  # YYYY-MM
                            published_date = datetime.strptime(release_date_str + '-01', '%Y-%m-%d')
                        else:  # YYYY or unknown
                            published_date = datetime.now()
                        
                        # Create episode object
                        episode = Episode(
                            id=ep['id'],
                            title=ep['name'],
                            summary=ep.get('description', ''),
                            podcast_name=show_name,
                            published_date=published_date,
                            link=ep['external_urls']['spotify'],
                            categories=[],
                            popularity_score=show_popularity
                        )
                        
                        all_episodes.append(episode)
                        
                    except Exception as e:
                        print(f"Error processing episode: {e}")
                        continue
                
                print(f"Loaded {len(episodes)} episodes from {show_name}")
                
            except Exception as e:
                print(f"Error loading show {show_id}: {e}")
                continue
        
        return all_episodes


class PodcastDataLoader:
    """Updated loader that uses Spotify API"""
    
    # ADD YOUR SPOTIFY CREDENTIALS HERE
    SPOTIFY_CLIENT_ID = "a22758c057ba46c58d92710c1c4aafbc"
    SPOTIFY_CLIENT_SECRET = "0d712dfb4dcc42e995923fdc043f1db3"
    
    @staticmethod
    def load_episodes() -> List[Episode]:
        """Load episodes from Spotify"""
        print("Loading podcasts from Spotify API...")
        
        loader = SpotifyPodcastLoader(
            client_id=PodcastDataLoader.SPOTIFY_CLIENT_ID,
            client_secret=PodcastDataLoader.SPOTIFY_CLIENT_SECRET
        )
        
        episodes = loader.load_episodes(episodes_per_show=10)
        print(f"Total episodes loaded from Spotify: {len(episodes)}")
        
        return episodes


# ============================================================================
# EMBEDDING & CONTENT-BASED
# ============================================================================

class ContentBasedEngine:
    """Content-based filtering using embeddings"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def compute_embeddings(self, episodes: List[Episode]) -> List[Episode]:
        """Compute embeddings for all episodes"""
        texts = [f"{ep.title}. {ep.summary}" for ep in episodes]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        for ep, emb in zip(episodes, embeddings):
            ep.embedding = emb
            
        return episodes
    
    def compute_user_profile(self, user: User, episodes: List[Episode]) -> np.ndarray:
        """Compute user profile from interests and history"""
        # Start with interest-based profile
        interest_text = " ".join(user.interests)
        interest_embedding = self.model.encode([interest_text])[0]
        
        # If user has history, incorporate it
        if user.interaction_history:
            episode_dict = {ep.id: ep for ep in episodes}
            history_embeddings = []
            weights = []
            
            for ep_id, action, timestamp in user.interaction_history:
                if ep_id in episode_dict:
                    ep = episode_dict[ep_id]
                    if ep.embedding is not None:
                        history_embeddings.append(ep.embedding)
                        # Weight: like=1.0, skip=-0.5, dislike=-1.0
                        weight_map = {"like": 1.0, "skip": -0.5, "dislike": -1.0}
                        weights.append(weight_map.get(action, 0.5))
            
            if history_embeddings:
                history_embeddings = np.array(history_embeddings)
                weights = np.array(weights).reshape(-1, 1)
                weighted_history = (history_embeddings * weights).mean(axis=0)
                
                # Combine interest and history (70% history, 30% interests)
                profile = 0.7 * weighted_history + 0.3 * interest_embedding
                return profile / np.linalg.norm(profile)
        
        return interest_embedding
    
    def get_candidates(self, user_profile: np.ndarray, episodes: List[Episode], 
                      top_k: int = 100) -> List[Tuple[Episode, float]]:
        """Get top-k candidate episodes based on similarity"""
        embeddings = np.array([ep.embedding for ep in episodes])
        similarities = cosine_similarity([user_profile], embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        candidates = [(episodes[i], similarities[i]) for i in top_indices]
        return candidates


# ============================================================================
# COLLABORATIVE FILTERING (SIMPLIFIED)
# ============================================================================

class CollaborativeFilteringEngine:
    """Simplified collaborative filtering using item-item similarity"""
    
    def __init__(self):
        self.item_similarity_matrix = None
        self.episode_id_to_idx = {}
        
    def build_similarity_matrix(self, episodes: List[Episode]):
        """Build item-item similarity matrix from embeddings"""
        embeddings = np.array([ep.embedding for ep in episodes])
        self.item_similarity_matrix = cosine_similarity(embeddings)
        self.episode_id_to_idx = {ep.id: i for i, ep in enumerate(episodes)}
    
    def get_collaborative_score(self, episode_id: str, user_history: List[Tuple[str, str, float]]) -> float:
        """Get collaborative score based on user's liked episodes"""
        if episode_id not in self.episode_id_to_idx:
            return 0.0
            
        ep_idx = self.episode_id_to_idx[episode_id]
        
        # Find episodes user liked
        liked_episodes = [ep_id for ep_id, action, _ in user_history if action == "like"]
        
        if not liked_episodes:
            return 0.0
        
        # Average similarity to liked episodes
        scores = []
        for liked_id in liked_episodes:
            if liked_id in self.episode_id_to_idx:
                liked_idx = self.episode_id_to_idx[liked_id]
                scores.append(self.item_similarity_matrix[ep_idx, liked_idx])
        
        return np.mean(scores) if scores else 0.0


# ============================================================================
# SAFETY FILTERING
# ============================================================================

class SafetyFilter:
    """Multi-dimensional safety filtering"""
    
    def __init__(self):
        # In production, you'd use actual models like Detoxify
        # For now, using keyword-based heuristics
        self.toxic_keywords = ['hate', 'violent', 'explicit', 'nsfw', 'offensive']
        
    def compute_safety_scores(self, episode: Episode) -> Dict[str, float]:
        """Compute safety scores across dimensions"""
        text = f"{episode.title} {episode.summary}".lower()
        
        # Toxicity score (0 = safe, 1 = toxic)
        toxicity = sum(1 for kw in self.toxic_keywords if kw in text) / len(self.toxic_keywords)
        
        # Misinformation risk (simplified - check for sensational language)
        sensational_words = ['shocking', 'unbelievable', 'secret', 'they dont want you to know']
        misinformation_risk = sum(1 for word in sensational_words if word in text) / len(sensational_words)
        
        scores = {
            'toxicity': min(toxicity, 1.0),
            'misinformation_risk': min(misinformation_risk, 1.0),
            'overall_safety': 1.0 - max(toxicity, misinformation_risk)
        }
        
        episode.safety_scores = scores
        return scores
    
    def filter_unsafe(self, episodes: List[Episode], threshold: float = 0.7) -> List[Episode]:
        """Filter out unsafe episodes"""
        safe_episodes = []
        for ep in episodes:
            if ep.safety_scores is None:
                self.compute_safety_scores(ep)
            
            if ep.safety_scores['overall_safety'] >= threshold:
                safe_episodes.append(ep)
                
        return safe_episodes


# ============================================================================
# RANKING MODEL (FEATURE-BASED)
# ============================================================================

class RankingModel:
    """ML-based ranking using LightGBM (simplified version using weighted features)"""
    
    def __init__(self):
        self.weights = {
            'content_similarity': 0.4,
            'collaborative_score': 0.3,
            'recency': 0.15,
            'popularity': 0.1,
            'safety': 0.05
        }
    
    def compute_features(self, episode: Episode, content_score: float, 
                        collab_score: float) -> Dict[str, float]:
        """Compute ranking features"""
        # Recency score (decay over time)
        days_old = (datetime.now() - episode.published_date).days
        recency_score = np.exp(-days_old / 30)  # 30-day half-life
        
        features = {
            'content_similarity': content_score,
            'collaborative_score': collab_score,
            'recency': recency_score,
            'popularity': episode.popularity_score,
            'safety': episode.safety_scores.get('overall_safety', 1.0) if episode.safety_scores else 1.0
        }
        
        return features
    
    def rank(self, episode: Episode, content_score: float, collab_score: float) -> float:
        """Compute final ranking score"""
        features = self.compute_features(episode, content_score, collab_score)
        
        # Weighted sum
        score = sum(features[k] * self.weights[k] for k in self.weights)
        return score


# ============================================================================
# FAIRNESS & DIVERSITY (MMR)
# ============================================================================

class FairnessReranker:
    """Maximal Marginal Relevance for diversity"""
    
    @staticmethod
    def mmr_rerank(candidates: List[Tuple[Episode, float]], 
                   lambda_param: float = 0.7, top_k: int = 10) -> List[Tuple[Episode, float]]:
        """
        Re-rank using MMR to balance relevance and diversity
        lambda_param: 1.0 = only relevance, 0.0 = only diversity
        """
        selected = []
        remaining = candidates.copy()
        
        while len(selected) < top_k and remaining:
            if not selected:
                # First item: highest relevance
                best_idx = 0
                selected.append(remaining.pop(best_idx))
            else:
                # Compute MMR score for each remaining item
                mmr_scores = []
                selected_embeddings = np.array([ep.embedding for ep, _ in selected])
                
                for ep, rel_score in remaining:
                    # Max similarity to already selected items
                    similarities = cosine_similarity([ep.embedding], selected_embeddings)[0]
                    max_sim = similarities.max()
                    
                    # MMR = λ * relevance - (1-λ) * max_similarity
                    mmr_score = lambda_param * rel_score - (1 - lambda_param) * max_sim
                    mmr_scores.append(mmr_score)
                
                # Select item with highest MMR score
                best_idx = np.argmax(mmr_scores)
                selected.append(remaining.pop(best_idx))
        
        return selected
    
    @staticmethod
    def boost_small_creators(episodes: List[Episode], boost_factor: float = 1.2) -> List[Episode]:
        """Boost episodes from less popular podcasts"""
        # Calculate average popularity per podcast
        podcast_popularity = {}
        for ep in episodes:
            if ep.podcast_name not in podcast_popularity:
                podcast_popularity[ep.podcast_name] = []
            podcast_popularity[ep.podcast_name].append(ep.popularity_score)
        
        avg_popularity = {name: np.mean(scores) for name, scores in podcast_popularity.items()}
        median_popularity = np.median(list(avg_popularity.values()))
        
        # Boost small creators
        for ep in episodes:
            if avg_popularity[ep.podcast_name] < median_popularity:
                ep.popularity_score *= boost_factor
        
        return episodes


# ============================================================================
# MAIN RECOMMENDATION ENGINE
# ============================================================================

class SafeAndSoundRecommender:
    """Main recommendation system orchestrating all components"""
    
    def __init__(self):
        self.content_engine = ContentBasedEngine()
        self.collab_engine = CollaborativeFilteringEngine()
        self.safety_filter = SafetyFilter()
        self.ranking_model = RankingModel()
        self.fairness_reranker = FairnessReranker()
        
        self.episodes: List[Episode] = []
        
    def load_data(self):
        """Load and prepare all data"""
        print("Loading podcast episodes from Spotify...")
        self.episodes = PodcastDataLoader.load_episodes()
        
        print(f"Loaded {len(self.episodes)} episodes")
        
        print("Computing embeddings...")
        self.episodes = self.content_engine.compute_embeddings(self.episodes)
        
        print("Computing safety scores...")
        for ep in self.episodes:
            self.safety_filter.compute_safety_scores(ep)
        
        print("Building collaborative filtering matrix...")
        self.collab_engine.build_similarity_matrix(self.episodes)
        
        print("Data preparation complete!")
    
    def recommend(self, user: User, top_k: int = 10, 
                 safety_threshold: float = 0.7,
                 diversity_lambda: float = 0.7) -> List[Recommendation]:
        """
        Generate recommendations for user
        
        Pipeline:
        1. Candidate Generation (content-based, top 100)
        2. Safety Filtering
        3. Ranking (ML model combining multiple signals)
        4. Re-ranking (MMR for diversity)
        5. Explanation generation
        """
        
        # Stage 1: Candidate Generation
        user_profile = self.content_engine.compute_user_profile(user, self.episodes)
        candidates = self.content_engine.get_candidates(user_profile, self.episodes, top_k=100)
        
        # Stage 2: Safety Filtering
        safe_candidates = [(ep, score) for ep, score in candidates 
                          if ep.safety_scores['overall_safety'] >= safety_threshold]
        
        # Stage 3: Ranking
        ranked_candidates = []
        for ep, content_score in safe_candidates:
            collab_score = self.collab_engine.get_collaborative_score(ep.id, user.interaction_history)
            final_score = self.ranking_model.rank(ep, content_score, collab_score)
            ranked_candidates.append((ep, final_score))
        
        ranked_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Stage 4: Re-ranking for Diversity
        diverse_recommendations = self.fairness_reranker.mmr_rerank(
            ranked_candidates, lambda_param=diversity_lambda, top_k=top_k
        )
        
        # Stage 5: Generate Explanations
        recommendations = []
        for ep, score in diverse_recommendations:
            explanation = self._generate_explanation(ep, user)
            
            # Get score components
            content_score = cosine_similarity([user_profile], [ep.embedding])[0][0]
            collab_score = self.collab_engine.get_collaborative_score(ep.id, user.interaction_history)
            features = self.ranking_model.compute_features(ep, content_score, collab_score)
            
            rec = Recommendation(
                episode=ep,
                score=score,
                explanation=explanation,
                components=features
            )
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_explanation(self, episode: Episode, user: User) -> str:
        """Generate human-readable explanation"""
        explanations = []
        
        # Interest matching
        matching_interests = [interest for interest in user.interests 
                            if interest.lower() in episode.summary.lower() or 
                               interest.lower() in episode.title.lower()]
        if matching_interests:
            explanations.append(f"Matches your interest in {', '.join(matching_interests)}")
        
        # Similar to liked episodes
        liked_episodes = [ep_id for ep_id, action, _ in user.interaction_history if action == "like"]
        if liked_episodes:
            explanations.append("Similar to episodes you've liked")
        
        # Safety
        if episode.safety_scores['overall_safety'] > 0.95:
            explanations.append("High-quality, safe content")
        
        # Recency
        days_old = (datetime.now() - episode.published_date).days
        if days_old < 7:
            explanations.append("Recently published")
        
        return " • ".join(explanations) if explanations else "Recommended for you"


# ============================================================================
# EVALUATION METRICS
# ============================================================================

class RecommenderEvaluator:
    """Evaluate recommendation quality"""
    
    @staticmethod
    def ndcg_at_k(recommendations: List[Recommendation], relevant_ids: List[str], k: int = 10) -> float:
        """Normalized Discounted Cumulative Gain"""
        dcg = 0.0
        for i, rec in enumerate(recommendations[:k]):
            if rec.episode.id in relevant_ids:
                dcg += 1.0 / np.log2(i + 2)  # +2 because log2(1)=0
        
        # Ideal DCG
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_ids), k)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def diversity_score(recommendations: List[Recommendation]) -> float:
        """Measure intra-list diversity (average pairwise distance)"""
        if len(recommendations) < 2:
            return 0.0
        
        embeddings = np.array([rec.episode.embedding for rec in recommendations])
        similarities = cosine_similarity(embeddings)
        
        # Average pairwise distance (1 - similarity)
        n = len(recommendations)
        total_distance = 0.0
        count = 0
        
        for i in range(n):
            for j in range(i+1, n):
                total_distance += (1 - similarities[i, j])
                count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    @staticmethod
    def catalog_coverage(recommendations: List[List[Recommendation]], total_episodes: int) -> float:
        """Percentage of catalog covered by recommendations"""
        recommended_ids = set()
        for rec_list in recommendations:
            for rec in rec_list:
                recommended_ids.add(rec.episode.id)
        
        return len(recommended_ids) / total_episodes


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Initialize recommender
    recommender = SafeAndSoundRecommender()
    recommender.load_data()
    
    # Create a user
    user = User(
        id="user_123",
        interests=["technology", "artificial intelligence", "startups"],
        interaction_history=[],  # Empty for new user
        profile_embedding=None
    )
    
    # Get recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR USER")
    print("="*80)
    
    recommendations = recommender.recommend(
        user, 
        top_k=10, 
        safety_threshold=0.7,
        diversity_lambda=0.7
    )
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec.episode.title}")
        print(f"   Podcast: {rec.episode.podcast_name}")
        print(f"   Score: {rec.score:.3f}")
        print(f"   Why: {rec.explanation}")
        print(f"   Safety: {rec.episode.safety_scores['overall_safety']:.2f}")
        print(f"   Spotify: {rec.episode.link}")
    
    # Evaluate
    print("\n" + "="*80)
    print("EVALUATION METRICS")
    print("="*80)
    
    evaluator = RecommenderEvaluator()
    diversity = evaluator.diversity_score(recommendations)
    print(f"Diversity Score: {diversity:.3f}")
    print(f"Total Episodes in Catalog: {len(recommender.episodes)}")