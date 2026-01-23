import React, { useState, useEffect } from 'react';
import { ThumbsUp, ThumbsDown, SkipForward, Loader2, Settings, TrendingUp, Shield, Sparkles, RefreshCw, Music, BarChart3, ChevronDown, ChevronUp } from 'lucide-react';

const API_BASE = 'http://localhost:5000';

export default function PodcastRecommender() {
  const [userId, setUserId] = useState(null);
  const [interests, setInterests] = useState(['technology', 'ai']);
  const [newInterest, setNewInterest] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [safetyThreshold, setSafetyThreshold] = useState(0.7);
  const [diversityLambda, setDiversityLambda] = useState(0.7);
  const [stats, setStats] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  const [interactions, setInteractions] = useState(0);
  const [expandedEpisode, setExpandedEpisode] = useState(null);
  const [feedbackLoading, setFeedbackLoading] = useState({});

  useEffect(() => {
    createUser();
    fetchStats();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const createUser = async () => {
    try {
      const response = await fetch(`${API_BASE}/users`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ interests })
      });
      const data = await response.json();
      setUserId(data.user.id);
      fetchRecommendations(data.user.id);
    } catch (error) {
      console.error('Error creating user:', error);
    }
  };

  const fetchRecommendations = async (id = userId) => {
    if (!id) return;
    setLoading(true);
    try {
      const response = await fetch(
        `${API_BASE}/recommendations/${id}?safety_threshold=${safetyThreshold}&diversity_lambda=${diversityLambda}`
      );
      const data = await response.json();
      setRecommendations(data.recommendations);
    } catch (error) {
      console.error('Error fetching recommendations:', error);
    }
    setLoading(false);
  };

  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_BASE}/stats`);
      const data = await response.json();
      setStats(data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  const submitFeedback = async (episodeId, action) => {
    // Optimistic UI update - mark as loading
    setFeedbackLoading(prev => ({ ...prev, [episodeId]: action }));
    
    try {
      await fetch(`${API_BASE}/feedback/${userId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ episode_id: episodeId, action })
      });
      
      setInteractions(prev => prev + 1);
      
      // Fetch new recommendations in background (no loading spinner)
      setTimeout(() => {
        fetchRecommendations();
        fetchStats();
      }, 500);
      
    } catch (error) {
      console.error('Error submitting feedback:', error);
    } finally {
      // Remove loading state after 1 second
      setTimeout(() => {
        setFeedbackLoading(prev => {
          const updated = { ...prev };
          delete updated[episodeId];
          return updated;
        });
      }, 1000);
    }
  };

  const addInterest = () => {
    if (newInterest && !interests.includes(newInterest.toLowerCase())) {
      const updated = [...interests, newInterest.toLowerCase()];
      setInterests(updated);
      updateUserInterests(updated);
      setNewInterest('');
    }
  };

  const removeInterest = (interest) => {
    const updated = interests.filter(i => i !== interest);
    setInterests(updated);
    updateUserInterests(updated);
  };

  const updateUserInterests = async (updatedInterests) => {
    try {
      await fetch(`${API_BASE}/users/${userId}/interests`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ interests: updatedInterests })
      });
      fetchRecommendations();
    } catch (error) {
      console.error('Error updating interests:', error);
    }
  };

  const getScoreColor = (score) => {
    if (score >= 0.7) return 'from-green-500 to-emerald-500';
    if (score >= 0.5) return 'from-yellow-500 to-orange-500';
    return 'from-gray-400 to-gray-500';
  };

  const toggleDetails = (episodeId) => {
    setExpandedEpisode(expandedEpisode === episodeId ? null : episodeId);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-900">
      {/* Animated background */}
      <div className="fixed inset-0 opacity-20">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl animate-pulse"></div>
        <div className="absolute top-0 right-1/4 w-96 h-96 bg-pink-500 rounded-full mix-blend-multiply filter blur-3xl animate-pulse" style={{animationDelay: '700ms'}}></div>
        <div className="absolute bottom-0 left-1/3 w-96 h-96 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl animate-pulse" style={{animationDelay: '1000ms'}}></div>
      </div>

      {/* Header */}
      <div className="relative backdrop-blur-md bg-white/10 border-b border-white/20 shadow-2xl">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl blur-lg opacity-75 animate-pulse"></div>
                <div className="relative w-14 h-14 bg-gradient-to-br from-purple-500 via-pink-500 to-purple-600 rounded-2xl flex items-center justify-center transform hover:scale-110 transition-transform">
                  <Music className="w-8 h-8 text-white" />
                </div>
              </div>
              <div>
                <h1 className="text-3xl font-bold text-white">Safe & Sound</h1>
                <p className="text-sm text-purple-200">Spotify Podcast Intelligence</p>
              </div>
            </div>
            
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="group relative p-3 bg-white/10 backdrop-blur-sm rounded-xl border border-white/20 hover:bg-white/20 transition-all hover:scale-105"
            >
              <Settings className={`w-6 h-6 text-white transition-transform ${showSettings ? 'rotate-90' : ''}`} />
            </button>
          </div>
        </div>
      </div>

      <div className="relative max-w-7xl mx-auto px-6 py-8">
        {/* Stats Grid */}
        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            {[
              { icon: TrendingUp, label: 'Episodes', value: stats.total_episodes, gradient: 'from-blue-500 to-cyan-500' },
              { icon: Shield, label: 'Safety', value: `${(stats.safety_stats.safety_rate * 100).toFixed(0)}%`, gradient: 'from-green-500 to-emerald-500' },
              { icon: Sparkles, label: 'Likes', value: interactions, gradient: 'from-purple-500 to-pink-500' },
              { icon: BarChart3, label: 'Status', value: interactions > 0 ? 'Learning' : 'Ready', gradient: 'from-orange-500 to-red-500' }
            ].map((stat, i) => (
              <div key={i} className="group relative">
                <div className="absolute inset-0 bg-gradient-to-r opacity-0 group-hover:opacity-75 rounded-2xl blur-xl transition-opacity"></div>
                <div className="relative backdrop-blur-xl bg-white/10 rounded-2xl p-6 border border-white/20 hover:border-white/40 transition-all hover:transform hover:scale-105 hover:shadow-2xl">
                  <div className="flex items-center justify-between mb-3">
                    <stat.icon className="w-8 h-8 text-white" />
                    <div className={`px-3 py-1 bg-gradient-to-r ${stat.gradient} rounded-full`}>
                      <span className="text-xs font-bold text-white">{stat.label}</span>
                    </div>
                  </div>
                  <div className="text-4xl font-bold text-white">{stat.value}</div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Settings Panel */}
        {showSettings && (
          <div className="backdrop-blur-xl bg-white/10 rounded-3xl p-8 border border-white/20 mb-8 shadow-2xl">
            <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
              <Settings className="w-6 h-6" />
              Advanced Settings
            </h3>
            
            <div className="space-y-8">
              <div>
                <div className="flex justify-between items-center mb-3">
                  <label className="text-lg font-semibold text-white flex items-center gap-2">
                    <Shield className="w-5 h-5" />
                    Safety Threshold
                  </label>
                  <div className="px-4 py-2 bg-gradient-to-r from-green-500 to-emerald-500 rounded-full">
                    <span className="text-lg font-bold text-white">{safetyThreshold.toFixed(2)}</span>
                  </div>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={safetyThreshold}
                  onChange={(e) => setSafetyThreshold(parseFloat(e.target.value))}
                  className="w-full h-3 bg-white/20 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-6 [&::-webkit-slider-thumb]:h-6 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-gradient-to-r [&::-webkit-slider-thumb]:from-green-500 [&::-webkit-slider-thumb]:to-emerald-500 [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:shadow-lg hover:[&::-webkit-slider-thumb]:scale-110 [&::-webkit-slider-thumb]:transition-transform"
                />
                <p className="text-sm text-purple-200 mt-2">Higher = stricter content filtering</p>
              </div>

              <div>
                <div className="flex justify-between items-center mb-3">
                  <label className="text-lg font-semibold text-white flex items-center gap-2">
                    <Sparkles className="w-5 h-5" />
                    Diversity Balance
                  </label>
                  <div className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full">
                    <span className="text-lg font-bold text-white">{diversityLambda.toFixed(2)}</span>
                  </div>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={diversityLambda}
                  onChange={(e) => setDiversityLambda(parseFloat(e.target.value))}
                  className="w-full h-3 bg-white/20 rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-6 [&::-webkit-slider-thumb]:h-6 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-gradient-to-r [&::-webkit-slider-thumb]:from-purple-500 [&::-webkit-slider-thumb]:to-pink-500 [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:shadow-lg hover:[&::-webkit-slider-thumb]:scale-110 [&::-webkit-slider-thumb]:transition-transform"
                />
                <p className="text-sm text-purple-200 mt-2">Higher = more relevance, lower = more variety</p>
              </div>

              <button
                onClick={() => fetchRecommendations()}
                className="w-full bg-gradient-to-r from-purple-500 to-pink-500 text-white font-bold py-4 px-6 rounded-2xl hover:from-purple-600 hover:to-pink-600 transition-all transform hover:scale-105 shadow-xl flex items-center justify-center gap-2"
              >
                <RefreshCw className="w-5 h-5" />
                Apply Settings
              </button>
            </div>
          </div>
        )}

        {/* Interests Section */}
        <div className="backdrop-blur-xl bg-white/10 rounded-3xl p-8 border border-white/20 mb-8 shadow-2xl">
          <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
            <Sparkles className="w-6 h-6" />
            Your Interests
          </h3>
          
          <div className="flex flex-wrap gap-3 mb-6">
            {interests.map((interest) => (
              <span
                key={interest}
                className="group relative px-5 py-2.5 bg-gradient-to-r from-purple-500/80 to-pink-500/80 backdrop-blur-sm rounded-full text-white font-semibold flex items-center gap-2 hover:from-purple-600 hover:to-pink-600 transition-all transform hover:scale-105 shadow-lg"
              >
                <span className="capitalize">{interest}</span>
                <button
                  onClick={() => removeInterest(interest)}
                  className="w-6 h-6 bg-white/20 rounded-full flex items-center justify-center hover:bg-white/30 transition-all"
                >
                  <span className="text-lg leading-none">×</span>
                </button>
              </span>
            ))}
          </div>

          <div className="flex gap-3">
            <input
              type="text"
              value={newInterest}
              onChange={(e) => setNewInterest(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && addInterest()}
              placeholder="Add new interest..."
              className="flex-1 px-6 py-4 bg-white/10 backdrop-blur-sm border border-white/20 rounded-2xl text-white placeholder-purple-200 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all"
            />
            <button
              onClick={addInterest}
              className="px-8 py-4 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-bold rounded-2xl hover:from-purple-600 hover:to-pink-600 transition-all transform hover:scale-105 shadow-xl"
            >
              Add
            </button>
          </div>
        </div>

        {/* Recommendations Header */}
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-3xl font-bold text-white flex items-center gap-3">
            <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl flex items-center justify-center">
              <Music className="w-7 h-7 text-white" />
            </div>
            Your Recommendations
          </h3>
          <button
            onClick={() => fetchRecommendations()}
            disabled={loading}
            className="flex items-center gap-2 px-6 py-3 bg-white/10 backdrop-blur-sm border border-white/20 rounded-2xl text-white font-semibold hover:bg-white/20 transition-all disabled:opacity-50 transform hover:scale-105"
          >
            {loading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <RefreshCw className="w-5 h-5" />
            )}
            Refresh
          </button>
        </div>

        {/* Recommendations */}
        {loading ? (
          <div className="flex flex-col items-center justify-center py-20">
            <Loader2 className="w-16 h-16 animate-spin text-purple-400 mb-4" />
            <p className="text-white text-lg">Finding perfect podcasts...</p>
          </div>
        ) : (
          <div className="space-y-6">
            {recommendations.map((rec, index) => (
              <div
                key={rec.episode.id}
                className="group relative backdrop-blur-xl bg-white/10 rounded-3xl p-8 border border-white/20 hover:border-white/40 transition-all hover:shadow-2xl"
              >
                <div className="flex items-start gap-6">
                  {/* Rank Badge */}
                  <div className={`flex-shrink-0 w-16 h-16 bg-gradient-to-br ${getScoreColor(rec.score)} rounded-2xl flex items-center justify-center shadow-lg`}>
                    <span className="text-2xl font-bold text-white">#{index + 1}</span>
                  </div>

                  <div className="flex-1">
                    {/* Header with Score */}
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <div className={`px-4 py-2 bg-gradient-to-r ${getScoreColor(rec.score)} rounded-full shadow-lg`}>
                          <span className="text-lg font-bold text-white">{rec.score.toFixed(3)}</span>
                        </div>
                        <div className="px-4 py-2 bg-white/10 backdrop-blur-sm rounded-full">
                          <span className="text-sm font-semibold text-purple-200">{rec.episode.podcast_name}</span>
                        </div>
                      </div>
                      
                      {/* Show Details Button */}
                      <button
                        onClick={() => toggleDetails(rec.episode.id)}
                        className="flex items-center gap-1 px-3 py-1.5 bg-white/10 backdrop-blur-sm rounded-lg text-purple-200 text-sm hover:bg-white/20 transition-all"
                      >
                        {expandedEpisode === rec.episode.id ? (
                          <>Details <ChevronUp className="w-4 h-4" /></>
                        ) : (
                          <>Details <ChevronDown className="w-4 h-4" /></>
                        )}
                      </button>
                    </div>

                    {/* Title */}
                    <h4 className="text-2xl font-bold text-white mb-3 group-hover:text-purple-200 transition-colors">
                      {rec.episode.title}
                    </h4>

                    {/* Summary */}
                    <p className="text-purple-100 mb-4 leading-relaxed line-clamp-2">
                      {rec.episode.summary}
                    </p>

                    {/* Explanation */}
                    <div className="mb-4">
                      <div className="inline-block px-4 py-2 bg-gradient-to-r from-purple-500/30 to-pink-500/30 backdrop-blur-sm rounded-xl border border-purple-400/30">
                        <span className="text-sm text-white font-medium">{rec.explanation}</span>
                      </div>
                    </div>

                    {/* Score Components (Collapsible) */}
                    {expandedEpisode === rec.episode.id && (
                      <div className="grid grid-cols-5 gap-3 mb-6 animate-slideDown">
                        {[
                          { label: 'Content', value: rec.components.content_similarity, color: 'from-blue-500 to-cyan-500' },
                          { label: 'Collab', value: rec.components.collaborative_score, color: 'from-purple-500 to-pink-500' },
                          { label: 'Recency', value: rec.components.recency, color: 'from-orange-500 to-red-500' },
                          { label: 'Popular', value: rec.components.popularity, color: 'from-yellow-500 to-orange-500' },
                          { label: 'Safety', value: rec.components.safety, color: 'from-green-500 to-emerald-500' }
                        ].map((component, i) => (
                          <div key={i} className="backdrop-blur-sm bg-white/10 rounded-xl p-3 border border-white/20 text-center">
                            <div className="text-xs text-purple-200 mb-1">{component.label}</div>
                            <div className={`text-lg font-bold bg-gradient-to-r ${component.color} bg-clip-text text-transparent`}>
                              {component.value.toFixed(2)}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Action Buttons */}
                    <div className="flex gap-3">
                      <button
                        onClick={() => submitFeedback(rec.episode.id, 'like')}
                        disabled={feedbackLoading[rec.episode.id]}
                        className="flex items-center gap-2 px-5 py-3 bg-gradient-to-r from-green-500 to-emerald-500 text-white font-bold rounded-xl hover:from-green-600 hover:to-emerald-600 transition-all transform hover:scale-105 shadow-lg disabled:opacity-50"
                      >
                        {feedbackLoading[rec.episode.id] === 'like' ? (
                          <Loader2 className="w-5 h-5 animate-spin" />
                        ) : (
                          <ThumbsUp className="w-5 h-5" />
                        )}
                        Like
                      </button>
                      <button
                        onClick={() => submitFeedback(rec.episode.id, 'dislike')}
                        disabled={feedbackLoading[rec.episode.id]}
                        className="flex items-center gap-2 px-5 py-3 bg-gradient-to-r from-red-500 to-pink-500 text-white font-bold rounded-xl hover:from-red-600 hover:to-pink-600 transition-all transform hover:scale-105 shadow-lg disabled:opacity-50"
                      >
                        {feedbackLoading[rec.episode.id] === 'dislike' ? (
                          <Loader2 className="w-5 h-5 animate-spin" />
                        ) : (
                          <ThumbsDown className="w-5 h-5" />
                        )}
                        Dislike
                      </button>
                      <button
                        onClick={() => submitFeedback(rec.episode.id, 'skip')}
                        disabled={feedbackLoading[rec.episode.id]}
                        className="flex items-center gap-2 px-5 py-3 bg-white/10 backdrop-blur-sm text-white font-bold rounded-xl hover:bg-white/20 transition-all transform hover:scale-105 border border-white/20 disabled:opacity-50"
                      >
                        {feedbackLoading[rec.episode.id] === 'skip' ? (
                          <Loader2 className="w-5 h-5 animate-spin" />
                        ) : (
                          <SkipForward className="w-5 h-5" />
                        )}
                        Skip
                      </button>
                      <a
                        href={rec.episode.link}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="ml-auto flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 text-white font-bold rounded-xl hover:from-purple-600 hover:to-pink-600 transition-all transform hover:scale-105 shadow-lg"
                      >
                        <Music className="w-5 h-5" />
                        Listen
                      </a>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <style jsx>{`
        @keyframes slideDown {
          from {
            opacity: 0;
            transform: translateY(-10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-slideDown {
          animation: slideDown 0.2s ease-out;
        }
        .line-clamp-2 {
          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }
      `}</style>
    </div>
  );
}