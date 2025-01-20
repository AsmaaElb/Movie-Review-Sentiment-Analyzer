import streamlit as st
import joblib
import numpy as np
from typing import Tuple, Dict, Optional
import os
from pathlib import Path
import time
import re
from contextlib import contextmanager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@contextmanager
def timer():
    """Context manager to time code execution."""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    st.sidebar.text(f"Processing time: {end - start:.2f} seconds")

class ModelCache:
    """Singleton class to cache loaded models."""
    _instance = None
    _model = None
    _vectorizer = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
        return cls._instance

    @classmethod
    def get_models(cls):
        return cls._model, cls._vectorizer

    @classmethod
    def set_models(cls, model, vectorizer):
        cls._model = model
        cls._vectorizer = vectorizer

class SentimentAnalyzer:
    """A class to handle sentiment analysis of movie reviews using an SVM model."""
    
    def __init__(self, model_path: str, vectorizer_path: str):
        """Initialize the SentimentAnalyzer with model and vectorizer paths."""
        self.model_path = Path(model_path)
        self.vectorizer_path = Path(vectorizer_path)
        self.model_cache = ModelCache()
        self.svm_w = None
        self.svm_b = None
        
    def load_models(self) -> bool:
        """Load the trained model and vectorizer from files or cache."""
        try:
            # Check cache first
            model, vectorizer = self.model_cache.get_models()
            if model is not None and vectorizer is not None:
                self._set_model_parameters(model)
                return True

            # Validate file paths
            if not self._validate_model_files():
                return False
            
            # Load models
            model = joblib.load(self.model_path)
            vectorizer = joblib.load(self.vectorizer_path)
            
            # Cache models
            self.model_cache.set_models(model, vectorizer)
            self._set_model_parameters(model)
            
            logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            st.error(f"Error loading models: {str(e)}")
            return False

    def _validate_model_files(self) -> bool:
        """Validate model file existence and format."""
        if not (self.model_path.exists() and self.vectorizer_path.exists()):
            st.error("Model files not found. Please check the file paths.")
            return False
        
        if self.model_path.suffix != '.joblib' or self.vectorizer_path.suffix != '.joblib':
            st.error("Invalid model file format. Expected .joblib files.")
            return False
            
        return True

    def _set_model_parameters(self, model: Dict):
        """Set SVM parameters from model."""
        try:
            self.svm_w = model["weights"]
            self.svm_b = model["bias"]
        except KeyError as e:
            logger.error(f"Missing key in model dictionary: {str(e)}")
            raise ValueError("Invalid model format")

    def _preprocess_text(self, text: str) -> str:
        """Preprocess the input text."""
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        return text.lower().strip()

    def predict_sentiment(self, review: str) -> Tuple[Optional[str], float, Dict]:
        """
        Predict sentiment of a given review.
        
        Returns:
            Tuple[Optional[str], float, Dict]: Predicted sentiment, confidence score, and additional metrics
        """
        try:
            with timer():
                # Get models from cache
                model, vectorizer = self.model_cache.get_models()
                if model is None or vectorizer is None:
                    raise ValueError("Models not loaded")

                # Preprocess review
                processed_review = self._preprocess_text(review)
                
                # Transform review to feature vector
                features = vectorizer.transform([processed_review]).toarray()
                
                # Calculate raw score
                raw_score = np.dot(features, self.svm_w) - self.svm_b
                
                # Calculate confidence using sigmoid function
                confidence = 1 / (1 + np.exp(-abs(raw_score[0])))
                
                # Determine sentiment
                sentiment = "Positive" if raw_score > 0 else "Negative"

                # Additional metrics
                metrics = {
                    "raw_score": float(raw_score[0]),
                    "feature_count": int(np.count_nonzero(features)),
                    "processed_length": len(processed_review),
                }
                
                return sentiment, None, metrics

            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            st.error(f"Error during prediction: {str(e)}")
            return None, 0.0, {}

def setup_page_config():
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="Movie Review Sentiment Analyzer",
        page_icon="🎬",
        layout="centered",
        initial_sidebar_state="expanded"
    )

def apply_custom_css():
    """Apply custom CSS styling."""
    st.markdown("""
        <style>
        .stTextArea textarea {
            font-size: 16px;
            border: 2px solid #f0f2f6;
            border-radius: 10px;
        }
        .sentiment-box {
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .positive {
            background-color: rgba(144, 238, 144, 0.6);
        }
        .negative {
            background-color: rgba(255, 182, 198, 0.6);
        }
        .metric-container {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)

def validate_input(review: str) -> bool:
    """Validate the user input."""
    if not review.strip():
        st.warning("⚠️ Please enter a review to analyze.")
        return False
    
    if len(review.split()) < 3:
        st.warning("⚠️ Please enter a longer review (at least 3 words).")
        return False
    
    if len(review) > 5000:
        st.warning("⚠️ Review is too long. Please keep it under 5000 characters.")
        return False
        
    return True

def display_results(sentiment: str, confidence: float, metrics: Dict, review: str):
    """Display the sentiment analysis results."""
    sentiment_color = "positive" if sentiment == "Positive" else "negative"
    
    st.markdown(f"""
        <div class="sentiment-box {sentiment_color}">
    <h3>Analysis Results</h3>
    <p><strong>Sentiment:</strong> {sentiment}</p>
    </div>
""", unsafe_allow_html=True)

    
    # Display metrics in columns
    st.markdown("### Review Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Word Count", len(review.split()))
    with col2:
        st.metric("Character Count", len(review))
    with col3:
        st.metric("Feature Count", metrics.get("feature_count", 0))
    
    # Display additional metrics in sidebar
    st.sidebar.markdown("### Technical Metrics")
    st.sidebar.json({
        "raw_score": round(metrics.get("raw_score", 0), 3),
        "processed_length": metrics.get("processed_length", 0)
    })

def main():
    """Main function to run the Streamlit app."""
    setup_page_config()
    apply_custom_css()
    
    st.title("🎬 Movie Review Sentiment Analyzer")
    st.markdown("""
        This app analyzes the sentiment of movie reviews using machine learning.
        Enter your review below to see if it's classified as positive or negative.
    """)
    
    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer(
        model_path="svm_model.joblib",
        vectorizer_path="vectorizer.joblib"
    )
    
    # Load models
    if not analyzer.load_models():
        st.stop()
    
    # Get user input
    user_review = st.text_area(
        "Enter your movie review:",
        height=150,
        placeholder="Type your review here...",
        key="user_review"
    )
    
    if st.button("Analyze Sentiment", type="primary", key="analyze_button"):
        if validate_input(user_review):
            with st.spinner("🔄 Analyzing sentiment..."):
                sentiment, confidence, metrics = analyzer.predict_sentiment(user_review)
                
                if sentiment is not None:
                    display_results(sentiment, confidence, metrics, user_review)
    
    # Display processing information in sidebar
    st.sidebar.markdown("### App Information")
    st.sidebar.text("Ready to analyze your review")

if __name__ == "__main__":
    main()