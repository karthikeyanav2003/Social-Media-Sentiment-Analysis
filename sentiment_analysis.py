
import pandas as pd
import numpy as np
import json
import os
import threading
import webbrowser
import time
import re
import warnings
from collections import defaultdict, Counter
from datetime import datetime
from flask import Flask, jsonify, request

# ML and NLP imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

class SentimentAnalyzer:
    def __init__(self, csv_file='sentimentdataset.csv'):
        self.csv_file = csv_file
        self.df = None
        self.df_clean = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vectorizer = None
        self.model = None
        self.results = []
        self.summary = {}
        self.eda_results = {}
        self.model_results = {}
        
    def load_data(self):
        """Load and initial data inspection"""
        try:
            print(f"📂 Loading dataset from {self.csv_file}...")
            self.df = pd.read_csv(self.csv_file)
            
            print(f"✅ Dataset loaded successfully!")
            print(f"📊 Shape: {self.df.shape}")
            print(f"📋 Columns: {list(self.df.columns)}")
            print(f"📝 First 3 rows:")
            print(self.df.head(3))
            
            # Check for missing values
            print(f"\n🔍 Missing Values:")
            missing = self.df.isnull().sum()
            for col, count in missing.items():
                if count > 0:
                    print(f"   {col}: {count} ({count/len(self.df)*100:.1f}%)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
    
    def clean_text(self, text):
        """Advanced text cleaning"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def preprocess_data(self):
        """Complete data preprocessing pipeline"""
        print("\n🧹 STARTING DATA PREPROCESSING...")
        print("="*50)
        
        # Create a copy for processing
        self.df_clean = self.df.copy()
        
        # 1. Handle missing values
        print("1️⃣ Handling missing values...")
        initial_count = len(self.df_clean)
        self.df_clean = self.df_clean.dropna(subset=['Text', 'Sentiment'])
        final_count = len(self.df_clean)
        print(f"   Removed {initial_count - final_count} rows with missing Text/Sentiment")
        
        # 2. Text cleaning
        print("2️⃣ Cleaning text data...")
        self.df_clean['Text_Original'] = self.df_clean['Text'].copy()
        self.df_clean['Text_Clean'] = self.df_clean['Text'].apply(self.clean_text)
        
        # Remove empty texts after cleaning
        self.df_clean = self.df_clean[self.df_clean['Text_Clean'].str.len() > 0]
        print(f"   Cleaned {len(self.df_clean)} text entries")
        
        # 3. Sentiment label standardization
        print("3️⃣ Standardizing sentiment labels...")
        self.df_clean['Sentiment'] = self.df_clean['Sentiment'].astype(str).str.lower().str.strip()
        
        # Advanced sentiment mapping
        sentiment_mapping = {
            # Positive sentiments
            'happy': 'positive', 'joy': 'positive', 'love': 'positive', 'excitement': 'positive',
            'positive': 'positive', 'optimism': 'positive', 'gratitude': 'positive', 'pride': 'positive',
            'relief': 'positive', 'contentment': 'positive', 'enthusiasm': 'positive',
            
            # Negative sentiments  
            'sad': 'negative', 'sadness': 'negative', 'anger': 'negative', 'fear': 'negative',
            'disgust': 'negative', 'negative': 'negative', 'frustration': 'negative',
            'disappointment': 'negative', 'anxiety': 'negative', 'guilt': 'negative',
            'shame': 'negative', 'hate': 'negative', 'worry': 'negative',
            
            # Neutral sentiments
            'neutral': 'neutral', 'surprise': 'neutral', 'curiosity': 'neutral',
            'confusion': 'neutral', 'boredom': 'neutral'
        }
        
        self.df_clean['Sentiment_Category'] = self.df_clean['Sentiment'].map(sentiment_mapping)
        
        # Handle unmapped sentiments
        unmapped = self.df_clean['Sentiment_Category'].isna().sum()
        if unmapped > 0:
            print(f"   ⚠️ {unmapped} unmapped sentiments, setting to 'neutral'")
            self.df_clean['Sentiment_Category'] = self.df_clean['Sentiment_Category'].fillna('neutral')
        
        # 4. Feature engineering
        print("4️⃣ Engineering features...")
        self.df_clean['Text_Length'] = self.df_clean['Text_Clean'].str.len()
        self.df_clean['Word_Count'] = self.df_clean['Text_Clean'].str.split().str.len()
        
        # Handle numeric columns
        for col in ['Likes', 'Retweets']:
            if col in self.df_clean.columns:
                self.df_clean[col] = pd.to_numeric(self.df_clean[col], errors='coerce').fillna(0)
        
        # Clean categorical columns
        for col in ['Platform', 'Country']:
            if col in self.df_clean.columns:
                self.df_clean[col] = self.df_clean[col].astype(str).str.lower().str.strip()
        
        print(f"✅ Preprocessing complete! Final dataset: {len(self.df_clean)} rows")
        print(f"📊 Sentiment distribution:")
        sentiment_counts = self.df_clean['Sentiment_Category'].value_counts()
        for sentiment, count in sentiment_counts.items():
            pct = (count / len(self.df_clean)) * 100
            print(f"   {sentiment}: {count} ({pct:.1f}%)")
        
        return True
    
    def perform_eda(self):
        """Comprehensive Exploratory Data Analysis"""
        print("\n📊 PERFORMING EXPLORATORY DATA ANALYSIS...")
        print("="*50)
        
        # Basic statistics
        print("1️⃣ Dataset Overview:")
        print(f"   Total samples: {len(self.df_clean):,}")
        print(f"   Features: {len(self.df_clean.columns)}")
        print(f"   Text length stats:")
        print(f"      Mean: {self.df_clean['Text_Length'].mean():.1f} characters")
        print(f"      Median: {self.df_clean['Text_Length'].median():.1f} characters")
        print(f"      Max: {self.df_clean['Text_Length'].max()} characters")
        print(f"      Min: {self.df_clean['Text_Length'].min()} characters")
        
        # Sentiment distribution
        print("\n2️⃣ Sentiment Distribution:")
        sentiment_dist = self.df_clean['Sentiment_Category'].value_counts()
        for sentiment, count in sentiment_dist.items():
            pct = (count / len(self.df_clean)) * 100
            emoji = "😊" if sentiment == "positive" else "😢" if sentiment == "negative" else "😐"
            print(f"   {emoji} {sentiment.title()}: {count:,} ({pct:.1f}%)")
        
        # Check class balance
        min_class = sentiment_dist.min()
        max_class = sentiment_dist.max()
        balance_ratio = min_class / max_class
        print(f"   📊 Class balance ratio: {balance_ratio:.2f} {'✅ Balanced' if balance_ratio > 0.5 else '⚠️ Imbalanced'}")
        
        # Platform analysis
        if 'Platform' in self.df_clean.columns:
            print("\n3️⃣ Platform Analysis:")
            platform_counts = self.df_clean['Platform'].value_counts().head(5)
            for platform, count in platform_counts.items():
                pct = (count / len(self.df_clean)) * 100
                print(f"   📱 {platform.title()}: {count:,} ({pct:.1f}%)")
        
        # Text length analysis by sentiment
        print("\n4️⃣ Text Length by Sentiment:")
        for sentiment in ['positive', 'negative', 'neutral']:
            if sentiment in sentiment_dist.index:
                subset = self.df_clean[self.df_clean['Sentiment_Category'] == sentiment]
                avg_len = subset['Text_Length'].mean()
                print(f"   {sentiment.title()}: {avg_len:.1f} avg characters")
        
        # Most common words analysis
        print("\n5️⃣ Most Common Words:")
        all_words = ' '.join(self.df_clean['Text_Clean']).split()
        word_freq = Counter(all_words)
        print("   Overall top words:")
        for word, count in word_freq.most_common(10):
            if len(word) > 2:  # Skip very short words
                print(f"      '{word}': {count}")
        
        # Store EDA results
        self.eda_results = {
            'total_samples': len(self.df_clean),
            'sentiment_distribution': dict(sentiment_dist),
            'class_balance_ratio': balance_ratio,
            'avg_text_length': self.df_clean['Text_Length'].mean(),
            'text_length_by_sentiment': {
                sentiment: self.df_clean[self.df_clean['Sentiment_Category'] == sentiment]['Text_Length'].mean()
                for sentiment in sentiment_dist.index
            },
            'top_words': dict(word_freq.most_common(20))
        }
        
        print("✅ EDA completed!")
        return True
    
    def prepare_features(self):
        """Prepare features for machine learning"""
        print("\n🔧 PREPARING FEATURES FOR ML...")
        print("="*50)
        
        # Prepare text data and labels
        X = self.df_clean['Text_Clean'].values
        y = self.df_clean['Sentiment_Category'].values
        
        print(f"1️⃣ Dataset size: {len(X)} samples")
        print(f"2️⃣ Classes: {np.unique(y)}")
        
        # Split the data
        print("3️⃣ Splitting data (80% train, 20% test)...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   Training set: {len(self.X_train)} samples")
        print(f"   Test set: {len(self.X_test)} samples")
        
        # Create TF-IDF features
        print("4️⃣ Creating TF-IDF features...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # Unigrams and bigrams
            stop_words='english',
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95  # Ignore terms that appear in more than 95% of documents
        )
        
        self.X_train_tfidf = self.vectorizer.fit_transform(self.X_train)
        self.X_test_tfidf = self.vectorizer.transform(self.X_test)
        
        print(f"   Feature matrix shape: {self.X_train_tfidf.shape}")
        print(f"   Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        print("✅ Feature preparation completed!")
        return True
    
    def train_and_evaluate_models(self):
        """Train multiple models and compare performance"""
        print("\n🤖 TRAINING AND EVALUATING MODELS...")
        print("="*50)
        
        # Define models to test
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Naive Bayes': MultinomialNB(),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='linear', random_state=42, probability=True)  # Add probability=True
        }
        
        model_results = {}
        
        print("🔄 Training models...")
        for name, model in models.items():
            print(f"\n📊 Training {name}...")
            
            # Train the model
            model.fit(self.X_train_tfidf, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test_tfidf)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train_tfidf, self.y_train, cv=5, scoring='accuracy')
            
            # Store results
            model_results[name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(self.y_test, y_pred, output_dict=True)
            }
            
            print(f"   ✅ Accuracy: {accuracy:.4f}")
            print(f"   📊 F1-Score: {f1:.4f}")
            print(f"   🔄 CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        # Select best model
        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['accuracy'])
        self.model = models[best_model_name]
        self.model.fit(self.X_train_tfidf, self.y_train)
        
        print(f"\n🏆 BEST MODEL: {best_model_name}")
        print(f"   🎯 Accuracy: {model_results[best_model_name]['accuracy']:.4f}")
        print(f"   📊 F1-Score: {model_results[best_model_name]['f1_score']:.4f}")
        
        # Detailed classification report for best model
        print(f"\n📋 DETAILED CLASSIFICATION REPORT ({best_model_name}):")
        y_pred_best = self.model.predict(self.X_test_tfidf)
        print(classification_report(self.y_test, y_pred_best))
        
        # Confusion Matrix
        print("📊 CONFUSION MATRIX:")
        cm = confusion_matrix(self.y_test, y_pred_best)
        labels = np.unique(self.y_test)
        print("     ", "  ".join(f"{label:>8}" for label in labels))
        for i, label in enumerate(labels):
            print(f"{label:>8}", "  ".join(f"{cm[i][j]:>8}" for j in range(len(labels))))
        
        # Store model results
        self.model_results = {
            'best_model': best_model_name,
            'all_models': model_results,
            'confusion_matrix': cm.tolist(),
            'test_accuracy': model_results[best_model_name]['accuracy'],
            'feature_count': self.X_train_tfidf.shape[1]
        }
        
        print("✅ Model training and evaluation completed!")
        return True
    
    def analyze_predictions(self):
        """Analyze model predictions on the full dataset"""
        print("\n🔍 ANALYZING PREDICTIONS ON FULL DATASET...")
        print("="*50)
        
        # Transform all data
        X_all = self.vectorizer.transform(self.df_clean['Text_Clean'])
        predictions = self.model.predict(X_all)
        
        # Handle prediction probabilities safely
        try:
            prediction_proba = self.model.predict_proba(X_all)
            confidence_scores = np.max(prediction_proba, axis=1)
        except AttributeError:
            # For models that don't support predict_proba, use decision function or set default confidence
            try:
                decision_scores = self.model.decision_function(X_all)
                if len(decision_scores.shape) == 1:  # Binary classification
                    confidence_scores = np.abs(decision_scores)
                else:  # Multi-class
                    confidence_scores = np.max(decision_scores, axis=1)
                # Normalize to 0-1 range
                confidence_scores = (confidence_scores - confidence_scores.min()) / (confidence_scores.max() - confidence_scores.min())
            except:
                # Fallback: set all confidences to 0.5
                confidence_scores = np.full(len(predictions), 0.5)
        
        # Add predictions to dataframe
        self.df_clean['Predicted_Sentiment'] = predictions
        self.df_clean['Prediction_Confidence'] = confidence_scores
        
        # Create results for web dashboard
        self.results = []
        for idx, row in self.df_clean.iterrows():
            self.results.append({
                'id': str(idx),
                'text': str(row['Text_Original'])[:200] + "..." if len(str(row['Text_Original'])) > 200 else str(row['Text_Original']),
                'actual_sentiment': str(row['Sentiment_Category']),
                'predicted_sentiment': str(row['Predicted_Sentiment']),
                'confidence': float(row['Prediction_Confidence']),
                'platform': str(row.get('Platform', 'unknown')),
                'country': str(row.get('Country', 'unknown')),
                'likes': float(row.get('Likes', 0)),
                'retweets': float(row.get('Retweets', 0)),
                'text_length': int(row['Text_Length']),
                'word_count': int(row['Word_Count'])
            })
        
        # Calculate prediction accuracy on full dataset
        full_accuracy = accuracy_score(self.df_clean['Sentiment_Category'], predictions)
        print(f"📊 Full dataset prediction accuracy: {full_accuracy:.4f}")
        
        # Prediction distribution
        pred_dist = pd.Series(predictions).value_counts()
        print(f"🔮 Prediction distribution:")
        for sentiment, count in pred_dist.items():
            pct = (count / len(predictions)) * 100
            emoji = "😊" if sentiment == "positive" else "😢" if sentiment == "negative" else "😐"
            print(f"   {emoji} {sentiment}: {count:,} ({pct:.1f}%)")
        
        # Confidence analysis
        avg_confidence = np.mean(confidence_scores)
        print(f"🎯 Average prediction confidence: {avg_confidence:.4f}")
        
        return True
    
    def generate_insights(self):
        """Generate comprehensive insights"""
        print("\n💡 GENERATING INSIGHTS...")
        
        insights = []
        
        # Model performance insights
        best_model = self.model_results['best_model']
        accuracy = self.model_results['test_accuracy']
        insights.append(f"🤖 Best performing model: {best_model} with {accuracy:.1%} accuracy")
        
        # Data insights
        sentiment_dist = self.eda_results['sentiment_distribution']
        if sentiment_dist:
            dominant_sentiment = max(sentiment_dist, key=sentiment_dist.get)
            pct = (sentiment_dist[dominant_sentiment] / sum(sentiment_dist.values())) * 100
            insights.append(f"😊 {dominant_sentiment.title()} sentiment dominates with {pct:.1f}% of posts")
        
        # Feature insights
        feature_count = self.model_results['feature_count']
        insights.append(f"🔧 Model uses {feature_count:,} TF-IDF features for classification")
        
        # Text length insights
        avg_length = self.eda_results['avg_text_length']
        insights.append(f"📝 Average text length is {avg_length:.0f} characters")
        
        # Class balance insight
        balance_ratio = self.eda_results['class_balance_ratio']
        if balance_ratio < 0.5:
            insights.append(f"⚠️ Dataset is imbalanced (ratio: {balance_ratio:.2f}) - consider balancing techniques")
        else:
            insights.append(f"✅ Dataset is well-balanced (ratio: {balance_ratio:.2f})")
        
        return insights
    
    def create_summary(self):
        """Create comprehensive summary"""
        insights = self.generate_insights()
        
        # Platform analysis
        platform_breakdown = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0})
        if 'Platform' in self.df_clean.columns:
            for _, row in self.df_clean.iterrows():
                platform_breakdown[row['Platform']][row['Predicted_Sentiment']] += 1
        
        # Country analysis
        country_breakdown = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0})
        if 'Country' in self.df_clean.columns:
            for _, row in self.df_clean.iterrows():
                country_breakdown[row['Country']][row['Predicted_Sentiment']] += 1
        
        # Time-based analysis (if timestamp available)
        time_analysis = {}
        if 'Timestamp' in self.df_clean.columns:
            try:
                self.df_clean['Date'] = pd.to_datetime(self.df_clean['Timestamp']).dt.date
                time_analysis = self.df_clean.groupby(['Date', 'Predicted_Sentiment']).size().unstack(fill_value=0).to_dict()
            except:
                pass
        
        self.summary = {
            'total_posts': len(self.df_clean),
            'model_performance': {
                'best_model': self.model_results['best_model'],
                'test_accuracy': self.model_results['test_accuracy'],
                'all_models': {name: results['accuracy'] for name, results in self.model_results['all_models'].items()}
            },
            'sentiment_distribution': dict(pd.Series([r['predicted_sentiment'] for r in self.results]).value_counts()),
            'actual_sentiment_distribution': self.eda_results['sentiment_distribution'],
            'platform_breakdown': dict(platform_breakdown),
            'country_breakdown': dict(country_breakdown),
            'time_analysis': time_analysis,
            'eda_results': self.eda_results,
            'insights': insights,
            'timestamp': datetime.now().isoformat(),
            'preprocessing_stats': {
                'original_samples': len(self.df) if self.df is not None else 0,
                'cleaned_samples': len(self.df_clean),
                'feature_count': self.model_results['feature_count'],
                'avg_confidence': np.mean([r['confidence'] for r in self.results])
            }
        }
        
        return True
    
    def run_complete_analysis(self):
        """Run the complete ML pipeline"""
        print("🚀 STARTING COMPLETE SENTIMENT ANALYSIS PIPELINE")
        print("="*60)
        
        # Step 1: Load data
        if not self.load_data():
            return False
        
        # Step 2: Preprocess data
        if not self.preprocess_data():
            return False
        
        # Step 3: Perform EDA
        if not self.perform_eda():
            return False
        
        # Step 4: Prepare features
        if not self.prepare_features():
            return False
        
        # Step 5: Train and evaluate models
        if not self.train_and_evaluate_models():
            return False
        
        # Step 6: Analyze predictions
        if not self.analyze_predictions():
            return False
        
        # Step 7: Create summary
        if not self.create_summary():
            return False
        
        # Print final summary
        print("\n🎉 ANALYSIS PIPELINE COMPLETED!")
        print("="*60)
        print(f"📊 Dataset: {len(self.df_clean):,} samples processed")
        print(f"🤖 Best Model: {self.model_results['best_model']}")
        print(f"🎯 Test Accuracy: {self.model_results['test_accuracy']:.4f} ({self.model_results['test_accuracy']:.1%})")
        print(f"🔧 Features: {self.model_results['feature_count']:,} TF-IDF features")
        print(f"📈 Avg Confidence: {self.summary['preprocessing_stats']['avg_confidence']:.4f}")
        
        print(f"\n😊 Predicted Sentiment Distribution:")
        for sentiment, count in self.summary['sentiment_distribution'].items():
            pct = (count / self.summary['total_posts']) * 100
            emoji = "😊" if sentiment == "positive" else "😢" if sentiment == "negative" else "😐"
            print(f"   {emoji} {sentiment.title()}: {count:,} ({pct:.1f}%)")
        
        print(f"\n💡 Key Insights:")
        for insight in self.summary['insights']:
            print(f"   • {insight}")
        
        return True

# Flask Web App
app = Flask(__name__)
analyzer = None

def load_html_template():
    """Load HTML template from file"""
    try:
        with open('dashboard.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <html>
        <head><title>Template Not Found</title></head>
        <body>
            <h1>❌ Error: dashboard.html template not found!</h1>
            <p>Please ensure dashboard.html is in the same directory as sentiment_analysis.py</p>
        </body>
        </html>
        """

@app.route('/')
def dashboard():
    """Enhanced dashboard with HTML template"""
    if not analyzer:
        return "<h1>❌ No Data</h1><p>Run analysis first</p>"
    
    # Load HTML template
    html_template = load_html_template()
    
    # Prepare data for template
    s = analyzer.summary
    model_perf = s.get('model_performance', {})
    sentiment_data = s.get('sentiment_distribution', {})
    platform_data = s.get('platform_breakdown', {})
    
    # Get top platforms
    platform_totals = {p: sum(sentiments.values()) for p, sentiments in platform_data.items()}
    top_platforms = sorted(platform_totals.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Get top words
    top_words = list(s.get('eda_results', {}).get('top_words', {}).items())[:10]
    
    # Create model performance HTML
    model_html = ''.join([
        f'<div class="model-item"><span class="model-name">{name}</span><span class="model-score">{acc:.1%}</span></div>'
        for name, acc in model_perf.get('all_models', {}).items()
    ])
    
    # Create platform stats HTML
    platform_html = ''.join([
        f'<div class="platform-item"><span>{platform.title()}</span><span><strong>{count:,}</strong></span></div>'
        for platform, count in top_platforms
    ])
    
    # Create word cloud HTML
    words_html = ''.join([
        f'<div class="word-item"><span>"{word}"</span><span><strong>{count}</strong></span></div>'
        for word, count in top_words
    ])
    
    # Create insights HTML
    insights_html = ''.join([
        f'<div class="insight">• {insight}</div>'
        for insight in s.get('insights', [])
    ])
    
    # Replace template placeholders
    html_content = html_template.format(
        total_posts=len(analyzer.results),
        test_accuracy=model_perf.get('test_accuracy', 0),
        best_model=model_perf.get('best_model', 'Unknown'),
        feature_count=s.get('preprocessing_stats', {}).get('feature_count', 0),
        avg_confidence=s.get('preprocessing_stats', {}).get('avg_confidence', 0),
        positive_count=sentiment_data.get('positive', 0),
        negative_count=sentiment_data.get('negative', 0),
        neutral_count=sentiment_data.get('neutral', 0),
        model_performance_html=model_html,
        platform_stats_html=platform_html,
        word_cloud_html=words_html,
        insights_html=insights_html,
        platform_labels=json.dumps([platform for platform, _ in top_platforms[:5]]),
        platform_data=json.dumps([count for _, count in top_platforms[:5]])
    )
    
    return html_content

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    """API endpoint for real-time sentiment prediction"""
    if not analyzer:
        return jsonify({'error': 'Model not loaded'})
    
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'})
    
    # Clean and predict
    cleaned_text = analyzer.clean_text(text)
    text_vector = analyzer.vectorizer.transform([cleaned_text])
    prediction = analyzer.model.predict(text_vector)[0]
    
    try:
        confidence = analyzer.model.predict_proba(text_vector)[0].max()
    except:
        confidence = 0.5
    
    return jsonify({
        'text': text,
        'sentiment': prediction,
        'confidence': float(confidence),
        'timestamp': datetime.now().isoformat()
    })

def main():
    """Main function with complete ML pipeline"""
    global analyzer
    
    print("🚀 ADVANCED SOCIAL MEDIA SENTIMENT ANALYSIS WITH ML")
    print("="*60)
    
    # Check for dataset
    if not os.path.exists('sentimentdataset.csv'):
        print("❌ ERROR: 'sentimentdataset.csv' not found!")
        print("💡 Place your CSV file in the same directory")
        input("Press Enter to exit...")
        return
    
    # Run complete analysis pipeline
    analyzer = SentimentAnalyzer()
    if not analyzer.run_complete_analysis():
        print("❌ Analysis pipeline failed!")
        input("Press Enter to exit...")
        return
    
    # Save comprehensive results
    try:
        results_data = {
            'summary': analyzer.summary,
            'results': analyzer.results,
            'model_performance': analyzer.model_results,
            'eda_results': analyzer.eda_results
        }
        with open('ml_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"💾 Complete results saved to 'ml_results.json'")
    except Exception as e:
        print(f"⚠️ Could not save results: {e}")
    
    # Start web dashboard
    print("\n" + "="*60)
    response = input("🚀 Start enhanced ML dashboard? (y/n): ").lower()
    
    if response in ['y', 'yes', '']:
        print("🌐 Starting enhanced dashboard at http://localhost:5000")
        print("🔮 Real-time prediction API at http://localhost:5000/predict")
        print("⏹️ Press Ctrl+C to stop")
        
        # Auto-open browser
        def open_browser():
            time.sleep(2)
            try:
                webbrowser.open('http://localhost:5000')
                print("🌐 Browser opened automatically")
            except:
                print("💡 Manually open: http://localhost:5000")
        
        threading.Thread(target=open_browser, daemon=True).start()
        
        try:
            app.run(host='0.0.0.0', port=5000, debug=False)
        except KeyboardInterrupt:
            print("\n👋 Dashboard stopped")
    else:
        print("✅ ML Analysis complete! Results saved to 'ml_results.json'")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
