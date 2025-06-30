# 🚀 Advanced Social Media Sentiment Analysis with Machine Learning

A comprehensive sentiment analysis system that processes social media data using advanced machine learning techniques and provides an interactive web dashboard for visualization and insights.

## 📊 Project Overview

This project implements a complete machine learning pipeline for sentiment analysis of social media posts, featuring:

- **Advanced Data Preprocessing** with text cleaning and feature engineering
- **Multiple ML Model Comparison** (Logistic Regression, Naive Bayes, SVM, Random Forest)
- **Interactive Web Dashboard** with real-time charts and insights
- **Cross-validation and Performance Metrics** for model evaluation
- **Clean Architecture** with separated Python logic and HTML templates

## 🎯 Features

### 🔧 Machine Learning Pipeline
- **Data Cleaning & Preprocessing**: Advanced text cleaning, sentiment mapping, feature engineering
- **Exploratory Data Analysis (EDA)**: Comprehensive statistical analysis and visualization
- **Feature Engineering**: TF-IDF vectorization with unigrams and bigrams
- **Model Training**: Multiple algorithms with cross-validation
- **Performance Evaluation**: Accuracy, F1-score, confusion matrix, classification reports

### 📈 Web Dashboard
- **Real-time Analytics**: Interactive charts showing sentiment distribution
- **Model Performance**: Comparison of different ML algorithms
- **Platform Analysis**: Breakdown by social media platforms
- **Word Analysis**: Most common words and trends
- **AI-Powered Insights**: Automated analysis and recommendations

### 🎨 Technical Features
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Clean Architecture**: Separated Python backend and HTML frontend
- **RESTful API**: Endpoint for real-time sentiment prediction
- **Error Handling**: Robust error handling and validation

## 📁 Dataset Requirements

The system expects a CSV file named `sentimentdataset.csv` with the following structure:

### Required Columns:
- **`Text`**: The social media post content (string)
- **`Sentiment`**: The sentiment label (string)

### Optional Columns:
- **`Platform`**: Social media platform (e.g., "twitter", "facebook", "instagram")
- **`User`**: Username or user identifier
- **`Likes`**: Number of likes/reactions (numeric)
- **`Retweets`**: Number of shares/retweets (numeric)
- **`Country`**: Geographic location (string)
- **`Timestamp`**: Post timestamp (datetime)

### Example Dataset Structure:
\`\`\`csv
Text,Sentiment,Platform,User,Likes,Retweets,Country
"I love this new product!",positive,twitter,user123,45,12,USA
"This is terrible service",negative,facebook,user456,3,1,UK
"Not sure about this",neutral,instagram,user789,8,0,Canada
\`\`\`

### Supported Sentiment Labels:
The system automatically maps various sentiment labels to three main categories:

**Positive**: `happy`, `joy`, `love`, `excitement`, `positive`, `optimism`, `gratitude`, `pride`, `relief`, `contentment`, `enthusiasm`

**Negative**: `sad`, `sadness`, `anger`, `fear`, `disgust`, `negative`, `frustration`, `disappointment`, `anxiety`, `guilt`, `shame`, `hate`, `worry`

**Neutral**: `neutral`, `surprise`, `curiosity`, `confusion`, `boredom`

## 🛠️ Technology Stack

### Backend (Python)
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms and metrics
- **numpy**: Numerical computing
- **flask**: Web framework for dashboard
- **matplotlib/seaborn**: Data visualization (backend processing)

### Frontend (Web Dashboard)
- **HTML5/CSS3**: Modern responsive design
- **JavaScript**: Interactive functionality
- **Chart.js**: Dynamic charts and visualizations
- **Bootstrap-inspired**: Grid system and components

### Machine Learning
- **TF-IDF Vectorization**: Text feature extraction
- **Logistic Regression**: Linear classification
- **Naive Bayes**: Probabilistic classification
- **Support Vector Machine (SVM)**: Margin-based classification
- **Random Forest**: Ensemble learning
- **Cross-validation**: Model validation and selection

## 🚀 Installation & Setup

### Prerequisites
\`\`\`bash
Python 3.7+ required
\`\`\`

### Install Dependencies
\`\`\`bash
pip install pandas flask scikit-learn matplotlib seaborn numpy
\`\`\`

## 🎮 Usage

### 1. Prepare Your Dataset
Place your CSV file as `sentimentdataset.csv` in the project directory with the required columns.

### 2. Run the Analysis
\`\`\`bash
python sentiment_analysis.py
\`\`\`

### 3. View Results
The script will:
1. **Load and clean** your dataset
2. **Perform EDA** with statistical analysis
3. **Train multiple ML models** and compare performance
4. **Generate predictions** for all data
5. **Save results** to `ml_results.json`
6. **Start web dashboard** at `http://localhost:5000`

### 4. Explore the Dashboard
- **📊 Overview**: Total posts and sentiment distribution
- **🤖 Model Performance**: Accuracy comparison of different algorithms
- **📱 Platform Analysis**: Breakdown by social media platforms
- **💡 AI Insights**: Automated analysis and recommendations
- **📈 Interactive Charts**: Pie charts and bar graphs

## 📊 Machine Learning Pipeline

### 1. Data Preprocessing
```python
# Text cleaning pipeline
- Remove URLs, mentions, hashtags
- Convert to lowercase
- Remove special characters
- Handle missing values
- Standardize sentiment labels
