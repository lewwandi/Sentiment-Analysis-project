import React, { useState, useEffect } from 'react';
import { MessageCircle, Send, BarChart3, AlertCircle, CheckCircle, XCircle, Minus, TrendingUp, Brain, Sparkles, History, Trash2 } from 'lucide-react';

const SentimentAnalysisApp = () => {
  const [comment, setComment] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState({ positive: 0, negative: 0, neutral: 0 });
  const [isTyping, setIsTyping] = useState(false);

  const API_BASE_URL = 'http://localhost:3001';

  useEffect(() => {
    const timer = setTimeout(() => setIsTyping(false), 500);
    return () => clearTimeout(timer);
  }, [comment]);

  const handleInputChange = (e) => {
    setComment(e.target.value);
    setIsTyping(true);
    setError('');
  };

  const analyzeSentiment = async () => {
    if (!comment.trim()) {
      setError('Please enter a comment to analyze');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ comment: comment.trim() }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error);
      }

      setResult(data);
      
      // Update stats
      if (data.total_counts) {
        setStats(data.total_counts);
      }
      
      // Add to history
      const newEntry = {
        id: Date.now(),
        comment: data.comment,
        sentiment: data.sentiment,
        timestamp: new Date().toLocaleTimeString(),
        confidence: data.confidence_scores
      };
      
      setHistory(prev => [newEntry, ...prev.slice(0, 9)]); // Keep last 10 entries
      
    } catch (err) {
      console.error('Error analyzing sentiment:', err);
      setError(err.message || 'Failed to analyze sentiment. Please make sure the Flask server is running on http://localhost:3001');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      analyzeSentiment();
    }
  };

  const clearAll = () => {
    setComment('');
    setResult(null);
    setError('');
  };

  const clearHistory = () => {
    setHistory([]);
  };

  const getSentimentIcon = (sentiment) => {
    switch (sentiment) {
      case 'positive':
        return <CheckCircle className="w-5 h-5 text-emerald-500" />;
      case 'negative':
        return <XCircle className="w-5 h-5 text-rose-500" />;
      case 'neutral':
        return <Minus className="w-5 h-5 text-slate-500" />;
      default:
        return <MessageCircle className="w-5 h-5 text-slate-400" />;
    }
  };

  const getSentimentColor = (sentiment) => {
    switch (sentiment) {
      case 'positive':
        return 'bg-gradient-to-r from-emerald-50 to-green-50 border-emerald-200 text-emerald-800';
      case 'negative':
        return 'bg-gradient-to-r from-rose-50 to-red-50 border-rose-200 text-rose-800';
      case 'neutral':
        return 'bg-gradient-to-r from-slate-50 to-gray-50 border-slate-200 text-slate-800';
      default:
        return 'bg-gradient-to-r from-slate-50 to-gray-50 border-slate-200 text-slate-800';
    }
  };

  const getSentimentGradient = (sentiment) => {
    switch (sentiment) {
      case 'positive':
        return 'bg-gradient-to-r from-emerald-500 to-green-500';
      case 'negative':
        return 'bg-gradient-to-r from-rose-500 to-red-500';
      case 'neutral':
        return 'bg-gradient-to-r from-slate-500 to-gray-500';
      default:
        return 'bg-gradient-to-r from-slate-500 to-gray-500';
    }
  };

  const formatConfidence = (score) => {
    if (!score) return '0.0%';
    return (score * 100).toFixed(1) + '%';
  };

  const ConfidenceBar = ({ label, value, isHighest }) => {
    const percentage = (value * 100).toFixed(1);
    const barColor = label === 'positive' ? 'bg-emerald-500' : 
                    label === 'negative' ? 'bg-rose-500' : 'bg-slate-500';
    
    return (
      <div className="space-y-2">
        <div className="flex justify-between items-center">
          <span className={`text-sm font-medium capitalize ${isHighest ? 'text-slate-800' : 'text-slate-600'}`}>
            {label}
          </span>
          <span className={`text-sm font-bold ${isHighest ? 'text-slate-800' : 'text-slate-600'}`}>
            {percentage}%
          </span>
        </div>
        <div className="w-full bg-slate-200 rounded-full h-2 overflow-hidden">
          <div 
            className={`h-full ${barColor} transition-all duration-700 ease-out ${isHighest ? 'animate-pulse' : ''}`}
            style={{ width: `${percentage}%` }}
          />
        </div>
      </div>
    );
  };

  const StatCard = ({ label, value, icon, color }) => (
    <div className={`bg-white rounded-xl p-4 shadow-sm border border-slate-200 hover:shadow-md transition-all duration-300`}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-slate-600 text-sm font-medium">{label}</p>
          <p className="text-2xl font-bold text-slate-800">{value}</p>
        </div>
        <div className={`p-3 rounded-lg ${color}`}>
          {icon}
        </div>
      </div>
    </div>
  );

  const totalAnalyses = stats.positive + stats.negative + stats.neutral;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {/* Animated background elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-300 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-yellow-300 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob animation-delay-2000"></div>
        <div className="absolute top-40 left-40 w-80 h-80 bg-pink-300 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob animation-delay-4000"></div>
      </div>

      <div className="relative z-10 p-4 max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12 pt-8">
          <div className="flex items-center justify-center mb-6">
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-full blur-lg opacity-30 animate-pulse"></div>
              <div className="relative bg-gradient-to-r from-indigo-600 to-purple-600 p-4 rounded-full">
                <Brain className="w-12 h-12 text-white" />
              </div>
            </div>
          </div>
          <h1 className="text-6xl font-bold bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 bg-clip-text text-transparent mb-4">
            AI Sentiment Analysis
          </h1>
          <p className="text-slate-600 text-xl max-w-2xl mx-auto leading-relaxed">
            Discover the emotional tone of your text.
          </p>
          
          {/* Stats Overview */}
          {totalAnalyses > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-8 max-w-4xl mx-auto">
              <StatCard 
                label="Total Analyses" 
                value={totalAnalyses} 
                icon={<TrendingUp className="w-6 h-6 text-indigo-600" />}
                color="bg-indigo-100"
              />
              <StatCard 
                label="Positive" 
                value={stats.positive} 
                icon={<CheckCircle className="w-6 h-6 text-emerald-600" />}
                color="bg-emerald-100"
              />
              <StatCard 
                label="Negative" 
                value={stats.negative} 
                icon={<XCircle className="w-6 h-6 text-rose-600" />}
                color="bg-rose-100"
              />
              <StatCard 
                label="Neutral" 
                value={stats.neutral} 
                icon={<Minus className="w-6 h-6 text-slate-600" />}
                color="bg-slate-100"
              />
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
          {/* Input Section */}
          <div className="xl:col-span-2 space-y-6">
            <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-white/20 p-8">
              <div className="flex items-center mb-6">
                <div className="bg-gradient-to-r from-indigo-500 to-purple-500 p-2 rounded-lg mr-4">
                  <Send className="w-6 h-6 text-white" />
                </div>
                <h2 className="text-2xl font-bold text-slate-800">Analyze Your Text</h2>
              </div>
              
              <div className="space-y-6">
                <div className="relative">
                  <textarea
                    value={comment}
                    onChange={handleInputChange}
                    onKeyPress={handleKeyPress}
                    placeholder="Share your thoughts, feedback, or any text you'd like to analyze..."
                    className="w-full h-40 p-6 border-2 border-slate-200 rounded-xl focus:ring-4 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all duration-300 resize-none text-slate-700 text-lg leading-relaxed placeholder-slate-400 bg-white/50 backdrop-blur-sm"
                    disabled={loading}
                  />
                  {isTyping && (
                    <div className="absolute bottom-4 right-4 flex items-center text-slate-400">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce animation-delay-100"></div>
                        <div className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce animation-delay-200"></div>
                      </div>
                    </div>
                  )}
                </div>
                
                <div className="flex space-x-4">
                  <button
                    onClick={analyzeSentiment}
                    disabled={loading || !comment.trim()}
                    className="flex-1 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 disabled:from-slate-400 disabled:to-slate-500 text-white font-semibold py-4 px-8 rounded-xl transition-all duration-300 flex items-center justify-center shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 disabled:transform-none"
                  >
                    {loading ? (
                      <>
                        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-3"></div>
                        <span>Analyzing Magic...</span>
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-5 h-5 mr-3" />
                        <span>Analyze Sentiment</span>
                      </>
                    )}
                  </button>
                  
                  <button
                    onClick={clearAll}
                    className="px-8 py-4 border-2 border-slate-300 text-slate-700 rounded-xl hover:bg-slate-50 hover:border-slate-400 transition-all duration-300 font-semibold"
                  >
                    Clear
                  </button>
                </div>
              </div>

              {error && (
                <div className="mt-6 p-4 bg-gradient-to-r from-rose-50 to-red-50 border-2 border-rose-200 rounded-xl flex items-center animate-shake">
                  <AlertCircle className="w-6 h-6 text-rose-500 mr-3 flex-shrink-0" />
                  <span className="text-rose-700 font-medium">{error}</span>
                </div>
              )}
            </div>

            {/* Results Section */}
            {result && (
              <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-white/20 p-8 animate-fadeIn">
                <div className="flex items-center mb-6">
                  <div className="bg-gradient-to-r from-emerald-500 to-green-500 p-2 rounded-lg mr-4">
                    <BarChart3 className="w-6 h-6 text-white" />
                  </div>
                  <h3 className="text-2xl font-bold text-slate-800">Analysis Results</h3>
                </div>
                
                <div className="space-y-6">
                  <div className="p-6 bg-gradient-to-r from-slate-50 to-gray-50 rounded-xl border border-slate-200">
                    <p className="text-slate-700 font-semibold mb-3 text-lg">Analyzed Text:</p>
                    <p className="text-slate-600 italic text-lg leading-relaxed">"{result.comment}"</p>
                  </div>
                  
                  <div className={`p-6 rounded-xl border-2 ${getSentimentColor(result.sentiment)} shadow-lg`}>
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center">
                        {getSentimentIcon(result.sentiment)}
                        <span className="ml-3 font-bold text-2xl capitalize">
                          {result.sentiment}
                        </span>
                      </div>
                      <div className={`px-4 py-2 rounded-full text-white font-semibold ${getSentimentGradient(result.sentiment)}`}>
                        Primary Match
                      </div>
                    </div>
                  </div>

                  {/* Confidence Scores */}
                  {result.confidence_scores && (
                    <div className="p-6 bg-white rounded-xl border border-slate-200 shadow-sm">
                      <h4 className="text-lg font-semibold text-slate-800 mb-4 flex items-center">
                        <TrendingUp className="w-5 h-5 mr-2 text-indigo-600" />
                        Confidence Breakdown
                      </h4>
                      <div className="space-y-4">
                        {Object.entries(result.confidence_scores)
                          .sort(([,a], [,b]) => b - a)
                          .map(([sentiment, score], index) => (
                            <ConfidenceBar 
                              key={sentiment}
                              label={sentiment}
                              value={score}
                              isHighest={index === 0}
                            />
                          ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* History Section */}
          <div className="xl:col-span-1">
            <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-white/20 p-6 sticky top-4">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center">
                  <div className="bg-gradient-to-r from-purple-500 to-pink-500 p-2 rounded-lg mr-3">
                    <History className="w-5 h-5 text-white" />
                  </div>
                  <h3 className="text-xl font-bold text-slate-800">Recent Analysis</h3>
                </div>
                {history.length > 0 && (
                  <button
                    onClick={clearHistory}
                    className="p-2 text-slate-400 hover:text-rose-500 hover:bg-rose-50 rounded-lg transition-all duration-200"
                    title="Clear History"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                )}
              </div>
              
              {history.length === 0 ? (
                <div className="text-center py-12">
                  <div className="bg-gradient-to-r from-slate-100 to-gray-100 rounded-full p-6 w-20 h-20 mx-auto mb-4 flex items-center justify-center">
                    <MessageCircle className="w-8 h-8 text-slate-400" />
                  </div>
                  <p className="text-slate-500 text-lg font-medium mb-2">No analysis yet</p>
                  <p className="text-slate-400 text-sm">Start by analyzing your first comment!</p>
                </div>
              ) : (
                <div className="space-y-3 max-h-96 overflow-y-auto custom-scrollbar">
                  {history.map((entry, index) => (
                    <div 
                      key={entry.id} 
                      className="p-4 border border-slate-200 rounded-xl hover:bg-slate-50 transition-all duration-200 hover:shadow-md animate-slideIn"
                      style={{ animationDelay: `${index * 100}ms` }}
                    >
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center">
                          {getSentimentIcon(entry.sentiment)}
                          <span className="ml-2 font-semibold text-sm capitalize text-slate-800">
                            {entry.sentiment}
                          </span>
                        </div>
                        <span className="text-xs text-slate-500 bg-slate-100 px-2 py-1 rounded-full">
                          {entry.timestamp}
                        </span>
                      </div>
                      <p className="text-sm text-slate-600 line-clamp-2 mb-2">
                        "{entry.comment}"
                      </p>
                      {entry.confidence && (
                        <div className="flex items-center justify-between">
                          <span className="text-xs text-slate-500">Confidence:</span>
                          <span className="text-xs font-semibold text-slate-700 bg-slate-100 px-2 py-1 rounded-full">
                            {formatConfidence(entry.confidence[entry.sentiment])}
                          </span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-12 py-8">
          <div className="bg-white/60 backdrop-blur-sm rounded-2xl border border-white/20 p-6 max-w-2xl mx-auto">
            <p className="text-slate-500">
              Built with React, Flask, and scikit-learn for accurate sentiment classification
            </p>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes blob {
          0% { transform: translate(0px, 0px) scale(1); }
          33% { transform: translate(30px, -50px) scale(1.1); }
          66% { transform: translate(-20px, 20px) scale(0.9); }
          100% { transform: translate(0px, 0px) scale(1); }
        }
        .animate-blob {
          animation: blob 7s infinite;
        }
        .animation-delay-2000 {
          animation-delay: 2s;
        }
        .animation-delay-4000 {
          animation-delay: 4s;
        }
        .animation-delay-100 {
          animation-delay: 0.1s;
        }
        .animation-delay-200 {
          animation-delay: 0.2s;
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fadeIn {
          animation: fadeIn 0.5s ease-out;
        }
        @keyframes slideIn {
          from { opacity: 0; transform: translateX(-20px); }
          to { opacity: 1; transform: translateX(0); }
        }
        .animate-slideIn {
          animation: slideIn 0.3s ease-out forwards;
        }
        @keyframes shake {
          0%, 100% { transform: translateX(0); }
          25% { transform: translateX(-5px); }
          75% { transform: translateX(5px); }
        }
        .animate-shake {
          animation: shake 0.5s ease-in-out;
        }
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: #f1f5f9;
          border-radius: 3px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #cbd5e1;
          border-radius: 3px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: #94a3b8;
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
};

export default SentimentAnalysisApp;