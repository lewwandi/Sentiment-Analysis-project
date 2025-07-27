from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
from nltk.stem import PorterStemmer
import pandas as pd
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
import re
from collections import Counter
import logging

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ps = PorterStemmer()

# Global variables for model components
model = None
tfidf_vec = None
label_encoder = None
#eature_pipeline = None

# Data tracking
data = {
    'reviews': [],
    'positive': 0,
    'negative': 0,
    'neutral': 0
}



def load_model_components():
    global model, tfidf_vec, label_encoder, feature_pipeline
    try:
        # Load model
        with open('best_model_regularized_logistic_regression.pkl', 'rb') as file:
            model = joblib.load(file)
        logger.info("Model loaded successfully!")

        # Load TF-IDF vectorizer
        with open('tfidf_vectorizer.pkl', 'rb') as file:
            tfidf_vec = joblib.load(file)
        logger.info("TF-IDF vectorizer loaded successfully!")

        # # # Load feature pipeline
        # with open('feature_pipeline.pkl', 'rb') as file:
        #     feature_pipeline = joblib.load(file)
        # logger.info("Feature pipeline loaded successfully!")

        # Initialize label encoder
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(['negative', 'neutral', 'positive'])
        logger.info("Label encoder initialized successfully!")

        # Verify all components are fitted
        if not hasattr(tfidf_vec, 'vocabulary_'):
            raise ValueError("TF-IDF vectorizer is not fitted")
        # if not hasattr(feature_pipeline.named_steps['feature_selection'], 'pvalues_'):
        #     raise ValueError("Feature selection step is not fitted")
        # if not hasattr(feature_pipeline.named_steps['svd'], 'components_'):
        #     raise ValueError("TruncatedSVD step is not fitted")
        # if not hasattr(feature_pipeline.named_steps['scaler'], 'mean_'):
        #     raise ValueError("StandardScaler step is not fitted")

        return True

    except FileNotFoundError as e:
        logger.error(f"Model file not found: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error loading model components: {str(e)}")
        return False

def text_preprocessing(text):
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text).lower()
    
    # Handle contractions
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'m", " am", text)
    
    # Remove URLs, HTML tags, and email addresses
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    
    # Handle repeated characters
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Remove numbers but keep emoticons
    text = re.sub(r'\d+', '', text)
    
    # Keep some punctuation
    text = re.sub(r'[^\w\s!?]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def create_vectorizer():
    return TfidfVectorizer(
        max_features=8000,
        min_df=10,
        max_df=0.7,
        stop_words='english',
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm='l2',
        use_idf=True,
        smooth_idf=True,
        lowercase=True,
        token_pattern=r'[a-zA-Z]{3,}',
        binary=False,
        strip_accents='unicode'
    )

def create_feature_pipeline():
    return Pipeline([
        ('feature_selection', SelectPercentile(chi2, percentile=80)),
        ('svd', TruncatedSVD(n_components=2000, random_state=42)),
        ('scaler', StandardScaler())
    ])

def predict_sentiment(text):
    global model, tfidf_vec, label_encoder, feature_pipeline
    
    try:
        if any(component is None for component in [model, tfidf_vec, label_encoder, feature_pipeline]):
            raise ValueError("Model components not properly loaded")
    
        processed_text = text_preprocessing(text)
        vectorized_text = tfidf_vec.transform([processed_text])
        processed_features = feature_pipeline.transform(vectorized_text)
        prediction_encoded = model.predict(processed_features)[0]
        prediction_decoded = label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get prediction probabilities
        confidence_scores = {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33}
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(processed_features)[0]
                confidence_scores = {
                    label: float(prob) for label, prob in zip(label_encoder.classes_, probabilities)
                }
            except Exception as e:
                logger.warning(f"Could not get prediction probabilities: {e}")
        
        return prediction_decoded, confidence_scores
        
    except Exception as e:
        logger.error(f"Error in predict_sentiment: {str(e)}")
        return 'neutral', {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33}

@app.route("/", methods=['GET'])
def index():
    return jsonify({
        'message': 'Sentiment Analysis API is running',
        'total_reviews': len(data['reviews']),
        'positive': data['positive'],
        'negative': data['negative'],
        'neutral': data['neutral'],
        'recent_reviews': data['reviews'][-5:] if data['reviews'] else []
    })

@app.route("/health", methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": tfidf_vec is not None,
        "label_encoder_loaded": label_encoder is not None,
        "feature_pipeline_loaded": feature_pipeline is not None
    })

@app.route("/predict", methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        
        if not input_data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        comment = input_data.get('comment')
        if not comment:
            return jsonify({'error': 'No comment provided'}), 400
        
        if not comment.strip():
            return jsonify({'error': 'Empty comment provided'}), 400
        
        # Get sentiment prediction
        sentiment, confidence_scores = predict_sentiment(comment.strip())
        
        # Update global counters
        review_entry = {
            'comment': comment.strip(),
            'sentiment': sentiment,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        data['reviews'].append(review_entry)
        
        # Update sentiment counters
        if sentiment == 'positive':
            data['positive'] += 1
        elif sentiment == 'negative':
            data['negative'] += 1
        elif sentiment == 'neutral':
            data['neutral'] += 1
        
        # Keep only last 100 reviews
        if len(data['reviews']) > 20:
            data['reviews'] = data['reviews'][-100:]
        
        return jsonify({
            'comment': comment.strip(),
            'sentiment': sentiment,
            'confidence_scores': confidence_scores,
            'total_counts': {
                'positive': data['positive'],
                'negative': data['negative'],
                'neutral': data['neutral']
            }
        })
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route("/batch_predict", methods=['POST'])
def batch_predict():
    try:
        input_data = request.get_json()
        
        if not input_data or 'comments' not in input_data:
            return jsonify({'error': 'No comments provided'}), 400
        
        comments = input_data['comments']
        if not isinstance(comments, list):
            return jsonify({'error': 'Comments should be a list'}), 400
        
        results = []
        for comment in comments:
            if comment and comment.strip():
                sentiment, confidence_scores = predict_sentiment(comment.strip())
                results.append({
                    'comment': comment.strip(),
                    'sentiment': sentiment,
                    'confidence_scores': confidence_scores
                })
                
                # Update counters
                if sentiment == 'positive':
                    data['positive'] += 1
                elif sentiment == 'negative':
                    data['negative'] += 1
                elif sentiment == 'neutral':
                    data['neutral'] += 1
        
        return jsonify({
            'results': results,
            'total_processed': len(results),
            'total_counts': {
                'positive': data['positive'],
                'negative': data['negative'],
                'neutral': data['neutral']
            }
        })
        
    except Exception as e:
        logger.error(f"Error in batch_predict endpoint: {str(e)}")
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

@app.route("/stats", methods=['GET'])
def get_stats():
    try:
        total_reviews = len(data['reviews'])
        return jsonify({
            'total_reviews': total_reviews,
            'sentiment_distribution': {
                'positive': data['positive'],
                'negative': data['negative'],
                'neutral': data['neutral']
            },
            'sentiment_percentages': {
                'positive': round((data['positive'] / max(1, data['positive'] + data['negative'] + data['neutral'])) * 100, 2),
                'negative': round((data['negative'] / max(1, data['positive'] + data['negative'] + data['neutral'])) * 100, 2),
                'neutral': round((data['neutral'] / max(1, data['positive'] + data['negative'] + data['neutral'])) * 100, 2)
            },
            'recent_reviews': data['reviews'][-10:] if data['reviews'] else []
        })
    except Exception as e:
        logger.error(f"Error in stats endpoint: {str(e)}")
        return jsonify({'error': f'Stats retrieval failed: {str(e)}'}), 500

@app.route("/reset", methods=['POST'])
def reset_data():
    try:
        global data
        data = {
            'reviews': [],
            'positive': 0,
            'negative': 0,
            'neutral': 0
        }
        return jsonify({'message': 'Data reset successfully'})
    except Exception as e:
        logger.error(f"Error in reset endpoint: {str(e)}")
        return jsonify({'error': f'Reset failed: {str(e)}'}), 500

if __name__ == '__main__':
    logger.info("Starting Flask Server...")
    
    # Load model components
    if load_model_components():
        logger.info("All components loaded successfully!")
        logger.info("Starting server on http://localhost:3001")
        app.run(debug=True, host='0.0.0.0', port=3001)
    else:
        logger.error("Failed to load required components. Please check your model files.")
     