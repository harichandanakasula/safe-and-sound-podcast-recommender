"""
Safe-and-Sound Podcast Recommender - Spotify Integration
A production-grade recommendation system with safety, fairness, and explainability.

Notes:
- Uses SentenceTransformers embeddings when available.
- On Windows environments where torch fails to load (WinError 1114), it falls back to TF-IDF embeddings.
- Requires Spotify credentials via environment variables or a .env file:
  SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
"""

from __future__ import annotations

import os
import pickle
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from dotenv import load_dotenv


# =============================================================================
# DATA MODELS
# =============================================================================

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


# =============================================================================
# SPOTIFY DATA INGESTION
# =============================================================================

class SpotifyPodcastLoader:
    """Loads podcast episodes from Spotify API"""

    # Show IDs MUST be valid. Removed two that were returning 404 in your logs.
    SPOTIFY_SHOWS = [
        "4rOoJ6Egrf8K2IrywzwOMk",  # The Joe Rogan Experience
        "2MAi0BvDc6GTFvKFPXnkCL",  # Lex Fridman Podcast
        "0ofXAdFIQQRsCYj9754UFx",  # TED Talks Daily
        "1VXcH8QHkjRcTCEd88U3ti",  # Stuff You Should Know
        "6kAsbP8pxwaU2kPibKTuHE",  # Crime Junkie (may vary by region)
        "1OLcQdw2PFDPG1jo3s0wbp",  # SmartLess (may vary by region)
        "2mTUnDkuKUkhiueKcVWoP0",  # Fest & Flauschig (example from your logs; keep if it works)
        # Removed from your failing logs:
        # "4yAEKfvoAPzqWYVinfOJ4Q"  # 404 in your logs
        # "2t7lZWs320KyXjqWz7PBvQ"  # 404 in your logs
    ]

    def __init__(self, client_id: str, client_secret: str):
        """Initialize Spotify client"""
        auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        self.sp = spotipy.Spotify(auth_manager=auth_manager)

    def load_episodes(self, episodes_per_show: int = 10) -> List[Episode]:
        """Load episodes from Spotify shows"""
        all_episodes: List[Episode] = []

        for show_id in self.SPOTIFY_SHOWS:
            try:
                show = self.sp.show(show_id,market="US")
                show_name = (show.get("name") or "").strip() or "Unknown Show"
                show_popularity = float(show.get("popularity", 50)) / 100.0

                results = self.sp.show_episodes(show_id, limit=episodes_per_show,market="US")
                items = (results or {}).get("items") or []

                loaded_for_show = 0
                for ep in items:
                    # Sometimes Spotify returns None entries or missing keys; guard hard.
                    if not ep:
                        continue

                    try:
                        release_date_str = (ep.get("release_date") or "").strip()
                        if len(release_date_str) == 10:      # YYYY-MM-DD
                            published_date = datetime.strptime(release_date_str, "%Y-%m-%d")
                        elif len(release_date_str) == 7:     # YYYY-MM
                            published_date = datetime.strptime(release_date_str + "-01", "%Y-%m-%d")
                        elif len(release_date_str) == 4:     # YYYY
                            published_date = datetime.strptime(release_date_str + "-01-01", "%Y-%m-%d")
                        else:
                            published_date = datetime.now()

                        urls = ep.get("external_urls") or {}
                        link = (urls.get("spotify") or "").strip()

                        episode = Episode(
                            id=(ep.get("id") or "").strip(),
                            title=(ep.get("name") or "").strip(),
                            summary=(ep.get("description") or "").strip(),
                            podcast_name=show_name,
                            published_date=published_date,
                            link=link,
                            categories=[],
                            popularity_score=show_popularity,
                        )

                        # Skip malformed
                        if not episode.id or not episode.title:
                            continue

                        all_episodes.append(episode)
                        loaded_for_show += 1

                    except Exception as e:
                        print(f"Error processing episode: {e}")
                        continue

                print(f"Loaded {loaded_for_show} episodes from {show_name}")

            except Exception as e:
                print(f"Error loading show {show_id}: {e}")
                continue

        # Deduplicate by episode id
        dedup = {}
        for ep in all_episodes:
            dedup[ep.id] = ep
        return list(dedup.values())


class PodcastDataLoader:
    """Loader that uses Spotify API + .env"""

    @staticmethod
    def load_episodes() -> List[Episode]:
        """Load episodes from Spotify"""
        load_dotenv()  # loads .env if present

        client_id = os.getenv("SPOTIFY_CLIENT_ID")
        client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

        if not client_id or not client_secret:
            raise ValueError(
                "Missing Spotify credentials. Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in your environment/.env."
            )

        print("Loading podcasts from Spotify API...")
        loader = SpotifyPodcastLoader(client_id=client_id, client_secret=client_secret)
        episodes = loader.load_episodes(episodes_per_show=10)
        print(f"Total episodes loaded from Spotify: {len(episodes)}")
        return episodes


# =============================================================================
# EMBEDDING & CONTENT-BASED (SentenceTransformer with TF-IDF fallback)
# =============================================================================

class ContentBasedEngine:
    """Content-based filtering using embeddings (SentenceTransformer) with TF-IDF fallback."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.backend = None  # "st" or "tfidf"
        self.model = None
        self.vectorizer = None
        self.tfidf_matrix = None

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.backend = "st"
        except Exception as e:
            self.backend = "tfidf"
            print("[ContentBasedEngine] WARNING: SentenceTransformer/torch failed to load. Using TF-IDF fallback.")
            print(f"[ContentBasedEngine] Reason: {e}")

    def _texts(self, episodes: List[Episode]) -> List[str]:
        return [f"{ep.title}. {ep.summary}".strip() for ep in episodes]

    def compute_embeddings(self, episodes: List[Episode]) -> List[Episode]:
        """Compute embeddings for all episodes."""
        texts = self._texts(episodes)

        if self.backend == "st":
            embeddings = self.model.encode(texts, show_progress_bar=True)
            for ep, emb in zip(episodes, embeddings):
                ep.embedding = np.asarray(emb, dtype=np.float32)
            return episodes

        # TF-IDF fallback
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

        # store dense vectors on each episode (small catalog, ok)
        dense = self.tfidf_matrix.toarray().astype(np.float32)
        for ep, vec in zip(episodes, dense):
            ep.embedding = vec
        return episodes

    def compute_user_profile(self, user: User, episodes: List[Episode]) -> np.ndarray:
        """Compute user profile from interests and history"""
        interest_text = " ".join(user.interests).strip() or "general"
        if self.backend == "st":
            interest_embedding = np.asarray(self.model.encode([interest_text])[0], dtype=np.float32)
        else:
            # TF-IDF: transform interest text into same space
            if self.vectorizer is None:
                # in case compute_embeddings wasn't called
                from sklearn.feature_extraction.text import TfidfVectorizer
                self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
                self.vectorizer.fit(self._texts(episodes))
            interest_embedding = self.vectorizer.transform([interest_text]).toarray()[0].astype(np.float32)

        # Incorporate history
        if user.interaction_history:
            episode_dict = {ep.id: ep for ep in episodes}
            history_embeddings = []
            weights = []

            for ep_id, action, _ in user.interaction_history:
                ep = episode_dict.get(ep_id)
                if ep is None or ep.embedding is None:
                    continue
                history_embeddings.append(ep.embedding)

                weight_map = {"like": 1.0, "skip": -0.5, "dislike": -1.0}
                weights.append(weight_map.get(action, 0.5))

            if history_embeddings:
                history_embeddings = np.array(history_embeddings, dtype=np.float32)
                weights = np.array(weights, dtype=np.float32).reshape(-1, 1)
                weighted_history = (history_embeddings * weights).mean(axis=0)

                profile = 0.7 * weighted_history + 0.3 * interest_embedding
                norm = np.linalg.norm(profile) or 1.0
                return profile / norm

        norm = np.linalg.norm(interest_embedding) or 1.0
        return interest_embedding / norm

    def get_candidates(self, user_profile: np.ndarray, episodes: List[Episode], top_k: int = 100) -> List[Tuple[Episode, float]]:
        """Get top-k candidate episodes based on similarity"""
        embeddings = np.array([ep.embedding for ep in episodes], dtype=np.float32)
        similarities = cosine_similarity([user_profile], embeddings)[0]
        top_k = min(top_k, len(episodes))
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(episodes[i], float(similarities[i])) for i in top_indices]


# =============================================================================
# COLLABORATIVE FILTERING (SIMPLIFIED)
# =============================================================================

class CollaborativeFilteringEngine:
    """Simplified collaborative filtering using item-item similarity"""

    def __init__(self):
        self.item_similarity_matrix = None
        self.episode_id_to_idx = {}

    def build_similarity_matrix(self, episodes: List[Episode]):
        embeddings = np.array([ep.embedding for ep in episodes], dtype=np.float32)
        self.item_similarity_matrix = cosine_similarity(embeddings)
        self.episode_id_to_idx = {ep.id: i for i, ep in enumerate(episodes)}

    def get_collaborative_score(self, episode_id: str, user_history: List[Tuple[str, str, float]]) -> float:
        if self.item_similarity_matrix is None:
            return 0.0
        if episode_id not in self.episode_id_to_idx:
            return 0.0

        ep_idx = self.episode_id_to_idx[episode_id]
        liked_episodes = [ep_id for ep_id, action, _ in user_history if action == "like"]
        if not liked_episodes:
            return 0.0

        scores = []
        for liked_id in liked_episodes:
            liked_idx = self.episode_id_to_idx.get(liked_id)
            if liked_idx is not None:
                scores.append(self.item_similarity_matrix[ep_idx, liked_idx])

        return float(np.mean(scores)) if scores else 0.0


# =============================================================================
# SAFETY FILTERING
# =============================================================================

class SafetyFilter:
    """Multi-dimensional safety filtering (simple heuristics)"""

    def __init__(self):
        self.toxic_keywords = ["hate", "violent", "explicit", "nsfw", "offensive"]

    def compute_safety_scores(self, episode: Episode) -> Dict[str, float]:
        text = f"{episode.title} {episode.summary}".lower()

        toxicity = sum(1 for kw in self.toxic_keywords if kw in text) / max(len(self.toxic_keywords), 1)

        sensational_words = ["shocking", "unbelievable", "secret", "they dont want you to know", "they don't want you to know"]
        misinformation_risk = sum(1 for word in sensational_words if word in text) / max(len(sensational_words), 1)

        scores = {
            "toxicity": float(min(toxicity, 1.0)),
            "misinformation_risk": float(min(misinformation_risk, 1.0)),
            "overall_safety": float(1.0 - max(toxicity, misinformation_risk)),
        }
        episode.safety_scores = scores
        return scores

    def filter_unsafe(self, episodes: List[Episode], threshold: float = 0.7) -> List[Episode]:
        safe_episodes = []
        for ep in episodes:
            if ep.safety_scores is None:
                self.compute_safety_scores(ep)
            if (ep.safety_scores or {}).get("overall_safety", 1.0) >= threshold:
                safe_episodes.append(ep)
        return safe_episodes


# =============================================================================
# RANKING MODEL
# =============================================================================

class RankingModel:
    """Feature-based ranking"""

    def __init__(self):
        self.weights = {
            "content_similarity": 0.4,
            "collaborative_score": 0.3,
            "recency": 0.15,
            "popularity": 0.1,
            "safety": 0.05,
        }

    def compute_features(self, episode: Episode, content_score: float, collab_score: float) -> Dict[str, float]:
        days_old = (datetime.now() - episode.published_date).days
        recency_score = float(np.exp(-days_old / 30.0))  # 30-day decay

        safety = 1.0
        if episode.safety_scores:
            safety = float(episode.safety_scores.get("overall_safety", 1.0))

        return {
            "content_similarity": float(content_score),
            "collaborative_score": float(collab_score),
            "recency": recency_score,
            "popularity": float(episode.popularity_score),
            "safety": safety,
        }

    def rank(self, episode: Episode, content_score: float, collab_score: float) -> float:
        feats = self.compute_features(episode, content_score, collab_score)
        return float(sum(feats[k] * self.weights[k] for k in self.weights))


# =============================================================================
# FAIRNESS & DIVERSITY (MMR)
# =============================================================================

class FairnessReranker:
    """Maximal Marginal Relevance for diversity"""

    @staticmethod
    def mmr_rerank(
        candidates: List[Tuple[Episode, float]],
        lambda_param: float = 0.7,
        top_k: int = 10,
    ) -> List[Tuple[Episode, float]]:
        selected: List[Tuple[Episode, float]] = []
        remaining = candidates.copy()

        while len(selected) < top_k and remaining:
            if not selected:
                selected.append(remaining.pop(0))
                continue

            selected_embeddings = np.array([ep.embedding for ep, _ in selected], dtype=np.float32)
            mmr_scores = []

            for ep, rel_score in remaining:
                similarities = cosine_similarity([ep.embedding], selected_embeddings)[0]
                max_sim = float(similarities.max()) if len(similarities) else 0.0
                mmr_score = lambda_param * float(rel_score) - (1.0 - lambda_param) * max_sim
                mmr_scores.append(mmr_score)

            best_idx = int(np.argmax(mmr_scores))
            selected.append(remaining.pop(best_idx))

        return selected

    @staticmethod
    def boost_small_creators(episodes: List[Episode], boost_factor: float = 1.2) -> List[Episode]:
        podcast_popularity: Dict[str, List[float]] = {}
        for ep in episodes:
            podcast_popularity.setdefault(ep.podcast_name, []).append(ep.popularity_score)

        avg_popularity = {name: float(np.mean(scores)) for name, scores in podcast_popularity.items()}
        median_popularity = float(np.median(list(avg_popularity.values()))) if avg_popularity else 0.0

        for ep in episodes:
            if avg_popularity.get(ep.podcast_name, 0.0) < median_popularity:
                ep.popularity_score *= boost_factor

        return episodes


# =============================================================================
# MAIN RECOMMENDATION ENGINE
# =============================================================================

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

    def recommend(
        self,
        user: User,
        top_k: int = 10,
        safety_threshold: float = 0.7,
        diversity_lambda: float = 0.7,
    ) -> List[Recommendation]:

        user_profile = self.content_engine.compute_user_profile(user, self.episodes)
        candidates = self.content_engine.get_candidates(user_profile, self.episodes, top_k=100)

        safe_candidates = [(ep, score) for ep, score in candidates
                           if (ep.safety_scores or {}).get("overall_safety", 1.0) >= safety_threshold]

        ranked_candidates: List[Tuple[Episode, float]] = []
        for ep, content_score in safe_candidates:
            collab_score = self.collab_engine.get_collaborative_score(ep.id, user.interaction_history)
            final_score = self.ranking_model.rank(ep, content_score, collab_score)
            ranked_candidates.append((ep, final_score))

        ranked_candidates.sort(key=lambda x: x[1], reverse=True)

        diverse = self.fairness_reranker.mmr_rerank(
            ranked_candidates, lambda_param=diversity_lambda, top_k=top_k
        )

        recs: List[Recommendation] = []
        for ep, score in diverse:
            explanation = self._generate_explanation(ep, user)
            content_score = float(cosine_similarity([user_profile], [ep.embedding])[0][0])
            collab_score = self.collab_engine.get_collaborative_score(ep.id, user.interaction_history)
            features = self.ranking_model.compute_features(ep, content_score, collab_score)

            recs.append(
                Recommendation(
                    episode=ep,
                    score=float(score),
                    explanation=explanation,
                    components=features,
                )
            )

        return recs

    def _generate_explanation(self, episode: Episode, user: User) -> str:
        explanations = []

        matching_interests = [
            interest for interest in user.interests
            if interest.lower() in (episode.summary or "").lower() or interest.lower() in (episode.title or "").lower()
        ]
        if matching_interests:
            explanations.append(f"Matches your interest in {', '.join(matching_interests)}")

        liked_episodes = [ep_id for ep_id, action, _ in user.interaction_history if action == "like"]
        if liked_episodes:
            explanations.append("Similar to episodes you've liked")

        if (episode.safety_scores or {}).get("overall_safety", 0.0) > 0.95:
            explanations.append("High-quality, safe content")

        days_old = (datetime.now() - episode.published_date).days
        if days_old < 7:
            explanations.append("Recently published")

        return " • ".join(explanations) if explanations else "Recommended for you"


# =============================================================================
# EVALUATION METRICS
# =============================================================================

class RecommenderEvaluator:
    """Evaluate recommendation quality"""

    @staticmethod
    def ndcg_at_k(recommendations: List[Recommendation], relevant_ids: List[str], k: int = 10) -> float:
        dcg = 0.0
        for i, rec in enumerate(recommendations[:k]):
            if rec.episode.id in relevant_ids:
                dcg += 1.0 / np.log2(i + 2)

        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_ids), k)))
        return float(dcg / idcg) if idcg > 0 else 0.0

    @staticmethod
    def diversity_score(recommendations: List[Recommendation]) -> float:
        if len(recommendations) < 2:
            return 0.0

        embeddings = np.array([rec.episode.embedding for rec in recommendations], dtype=np.float32)
        similarities = cosine_similarity(embeddings)

        n = len(recommendations)
        total_distance = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_distance += (1.0 - float(similarities[i, j]))
                count += 1

        return float(total_distance / count) if count > 0 else 0.0

    @staticmethod
    def catalog_coverage(recommendations: List[List[Recommendation]], total_episodes: int) -> float:
        recommended_ids = set()
        for rec_list in recommendations:
            for rec in rec_list:
                recommended_ids.add(rec.episode.id)
        return float(len(recommended_ids) / max(total_episodes, 1))


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    recommender = SafeAndSoundRecommender()
    recommender.load_data()

    user = User(
        id="user_123",
        interests=["technology", "artificial intelligence", "startups"],
        interaction_history=[],
        profile_embedding=None,
    )

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR USER")
    print("=" * 80)

    recommendations = recommender.recommend(
        user,
        top_k=10,
        safety_threshold=0.7,
        diversity_lambda=0.7,
    )

    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec.episode.title}")
        print(f"   Podcast: {rec.episode.podcast_name}")
        print(f"   Score: {rec.score:.3f}")
        print(f"   Why: {rec.explanation}")
        print(f"   Safety: {(rec.episode.safety_scores or {}).get('overall_safety', 1.0):.2f}")
        print(f"   Spotify: {rec.episode.link}")

    print("\n" + "=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)

    evaluator = RecommenderEvaluator()
    diversity = evaluator.diversity_score(recommendations)
    print(f"Diversity Score: {diversity:.3f}")
    print(f"Total Episodes in Catalog: {len(recommender.episodes)}")
