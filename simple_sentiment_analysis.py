"""
Simple Social Media Sentiment Analysis System
No complex templates - just basic HTML generation
"""

import pandas as pd
import json
import os
import sys
import threading
import webbrowser
import time
from collections import Counter, defaultdict
from datetime import datetime
from flask import Flask, jsonify

# Set UTF-8 encoding for Windows compatibility
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

class SentimentAnalyzer:
    def __init__(self, csv_file='sentimentdataset.csv'):
        """Initialize the sentiment analyzer"""
        self.csv_file = csv_file
        self.df = None
        self.results = []
        self.summary = {}
        
    def load_and_clean_data(self):
        """Load and clean the dataset"""
        try:
            print(f"📂 Loading dataset from {self.csv_file}...")
            self.df = pd.read_csv(self.csv_file)
            
            print(f"✅ Dataset loaded successfully!")
            print(f"📊 Shape: {self.df.shape}")
            print(f"📋 Columns: {list(self.df.columns)}")
            
            # Clean the data
            self.clean_data()
            return True
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            self.df = pd.DataFrame()
            return False
    
    def clean_data(self):
        """Clean and preprocess the dataset"""
        if self.df.empty:
            return
        
        print("🧹 Cleaning data...")
        
        # Handle missing values
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['Text', 'Sentiment'])
        final_count = len(self.df)
        
        if initial_count != final_count:
            print(f"🗑️ Removed {initial_count - final_count} rows with missing data")
        
        # Clean text and sentiment columns
        self.df['Text'] = self.df['Text'].astype(str).str.strip()
        self.df['Sentiment'] = self.df['Sentiment'].str.lower().str.strip()
        
        # Map sentiment categories to standard format
        sentiment_mapping = {
            'happy': 'positive', 'joy': 'positive', 'love': 'positive', 'excitement': 'positive',
            'positive': 'positive', 'optimism': 'positive', 'gratitude': 'positive', 'pride': 'positive',
            'sad': 'negative', 'sadness': 'negative', 'anger': 'negative', 'fear': 'negative',
            'disgust': 'negative', 'negative': 'negative', 'frustration': 'negative', 'disappointment': 'negative',
            'neutral': 'neutral', 'surprise': 'neutral', 'curiosity': 'neutral', 'confusion': 'neutral'
        }
        
        self.df['Sentiment_Category'] = self.df['Sentiment'].map(sentiment_mapping)
        self.df['Sentiment_Category'] = self.df['Sentiment_Category'].fillna('neutral')
        
        # Handle numeric columns safely
        for col in ['Retweets', 'Likes']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
        
        # Clean platform names
        if 'Platform' in self.df.columns:
            self.df['Platform'] = self.df['Platform'].str.lower().str.strip()
        
        print(f"✅ Data cleaning completed - Final shape: {self.df.shape}")
        sentiment_counts = dict(self.df['Sentiment_Category'].value_counts())
        print(f"📈 Sentiment distribution: {sentiment_counts}")
    
    def analyze_dataset(self):
        """Perform comprehensive analysis"""
        if self.df.empty:
            print("❌ No data available for analysis")
            return False
        
        print("🔍 Performing comprehensive analysis...")
        
        # Basic statistics
        total_posts = len(self.df)
        sentiment_dist = dict(self.df['Sentiment_Category'].value_counts())
        
        # Platform analysis
        platform_sentiment = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0})
        if 'Platform' in self.df.columns:
            for _, row in self.df.iterrows():
                platform = row['Platform']
                sentiment = row['Sentiment_Category']
                platform_sentiment[platform][sentiment] += 1
        
        # Country analysis
        country_analysis = {'positive': {}, 'negative': {}, 'neutral': {}}
        if 'Country' in self.df.columns:
            for _, row in self.df.iterrows():
                country = row['Country']
                sentiment = row['Sentiment_Category']
                if country not in country_analysis[sentiment]:
                    country_analysis[sentiment][country] = 0
                country_analysis[sentiment][country] += 1
        
        # Text statistics
        text_lengths = self.df['Text'].str.len()
        text_stats = {
            'avg_text_length': float(text_lengths.mean()),
            'max_text_length': int(text_lengths.max()),
            'min_text_length': int(text_lengths.min())
        }
        
        # Prepare results for JSON
        self.results = []
        for idx, row in self.df.iterrows():
            result = {
                'id': str(idx),
                'text': str(row['Text']),
                'sentiment': str(row['Sentiment_Category']),
                'original_sentiment': str(row['Sentiment']),
                'platform': str(row.get('Platform', 'unknown')),
                'user': str(row.get('User', 'anonymous')),
                'likes': float(row.get('Likes', 0)),
                'retweets': float(row.get('Retweets', 0)),
                'country': str(row.get('Country', 'unknown'))
            }
            self.results.append(result)
        
        # Generate insights
        insights = self.generate_insights()
        
        # Create summary
        self.summary = {
            'total_posts': total_posts,
            'sentiment_distribution': sentiment_dist,
            'platform_breakdown': dict(platform_sentiment),
            'country_analysis': country_analysis,
            'text_statistics': text_stats,
            'key_insights': insights,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        return True
    
    def generate_insights(self):
        """Generate key insights from the analysis"""
        insights = []
        
        if self.df.empty:
            return insights
        
        # Sentiment insights
        sentiment_counts = self.df['Sentiment_Category'].value_counts()
        total_posts = len(self.df)
        
        if not sentiment_counts.empty:
            dominant_sentiment = sentiment_counts.index[0]
            dominant_percentage = (sentiment_counts.iloc[0] / total_posts) * 100
            insights.append(f"{dominant_sentiment.title()} sentiment dominates with {dominant_percentage:.1f}% of all posts")
        
        # Platform insights
        if 'Platform' in self.df.columns:
            platform_counts = self.df['Platform'].value_counts()
            if not platform_counts.empty:
                top_platform = platform_counts.index[0]
                platform_percentage = (platform_counts.iloc[0] / total_posts) * 100
                insights.append(f"{top_platform.title()} is the most active platform with {platform_percentage:.1f}% of posts")
        
        # Country insights
        if 'Country' in self.df.columns:
            country_counts = self.df['Country'].value_counts()
            if not country_counts.empty:
                top_country = country_counts.index[0]
                country_percentage = (country_counts.iloc[0] / total_posts) * 100
                insights.append(f"{top_country} contributes {country_percentage:.1f}% of all posts")
        
        return insights
    
    def print_summary(self):
        """Print analysis summary to console"""
        if not self.summary:
            print("❌ No analysis results available")
            return
        
        print("\n" + "="*60)
        print("🎉 ANALYSIS COMPLETE!")
        print("="*60)
        
        print(f"📊 Dataset Overview:")
        print(f"   • Total Posts: {len(self.results):,}")
        print(f"   • Unique Platforms: {len(set(r['platform'] for r in self.results))}")
        print(f"   • Unique Countries: {len(set(r['country'] for r in self.results))}")
        
        print(f"\n😊 Sentiment Distribution:")
        for sentiment, count in self.summary['sentiment_distribution'].items():
            percentage = (count / self.summary['total_posts']) * 100
            emoji = "😊" if sentiment == "positive" else "😢" if sentiment == "negative" else "😐"
            print(f"   {emoji} {sentiment.title()}: {count:,} ({percentage:.1f}%)")
        
        print(f"\n📱 Top Platforms:")
        platform_totals = {}
        for platform, sentiments in self.summary['platform_breakdown'].items():
            total = sum(sentiments.values())
            platform_totals[platform] = total
        
        for platform, total in sorted(platform_totals.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   • {platform.title()}: {total:,} posts")
        
        print(f"\n💡 Key Insights:")
        for insight in self.summary['key_insights']:
            print(f"   • {insight}")

# Flask Web Application
app = Flask(__name__)

# Global analyzer instance
analyzer = None

def create_simple_html():
    """Create a simple HTML dashboard"""
    if not analyzer or not analyzer.results:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sentiment Analysis - No Data</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; padding: 50px; background: #f0f0f0; }
                .container { background: white; padding: 40px; border-radius: 10px; max-width: 600px; margin: 0 auto; }
                .error { color: #e74c3c; font-size: 1.2em; margin: 20px 0; }
                .btn { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Social Media Sentiment Analysis</h1>
                <div class="error">❌ No data available</div>
                <p>Please make sure 'sentimentdataset.csv' is in the same directory and restart the application.</p>
                <button onclick="location.reload()" class="btn">🔄 Refresh Page</button>
            </div>
        </body>
        </html>
        """
    
    # Get data for charts
    summary = analyzer.summary
    sentiment_data = summary.get('sentiment_distribution', {})
    
    # Create HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>📊 Sentiment Analysis Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                margin: 0; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                min-height: 100vh; 
                padding: 20px; 
            }}
            .container {{ 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white; 
                border-radius: 15px; 
                overflow: hidden; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.2); 
            }}
            .header {{ 
                background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%); 
                color: white; 
                padding: 30px; 
                text-align: center; 
            }}
            .header h1 {{ font-size: 2.5em; margin: 0 0 10px 0; }}
            .stats-grid {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                gap: 20px; 
                padding: 30px; 
                background: #f8f9fa; 
            }}
            .stat-card {{ 
                background: white; 
                padding: 25px; 
                border-radius: 10px; 
                text-align: center; 
                box-shadow: 0 5px 15px rgba(0,0,0,0.1); 
            }}
            .stat-number {{ font-size: 2.5em; font-weight: bold; margin-bottom: 10px; }}
            .positive {{ color: #27ae60; }}
            .negative {{ color: #e74c3c; }}
            .neutral {{ color: #95a5a6; }}
            .total {{ color: #3498db; }}
            .chart-section {{ padding: 30px; }}
            .chart-container {{ 
                background: white; 
                padding: 20px; 
                border-radius: 10px; 
                margin: 20px 0; 
                box-shadow: 0 5px 15px rgba(0,0,0,0.1); 
            }}
            .chart-title {{ font-size: 1.5em; text-align: center; margin-bottom: 20px; color: #2c3e50; }}
            .insights {{ 
                background: white; 
                margin: 20px 0; 
                border-radius: 10px; 
                overflow: hidden; 
                box-shadow: 0 5px 15px rgba(0,0,0,0.1); 
            }}
            .insights-header {{ background: #34495e; color: white; padding: 20px; font-size: 1.3em; }}
            .insights-content {{ padding: 20px; }}
            .insight-item {{ padding: 10px 0; border-bottom: 1px solid #eee; }}
            .controls {{ text-align: center; padding: 30px; background: #ecf0f1; }}
            .btn {{ 
                background: #3498db; 
                color: white; 
                border: none; 
                padding: 12px 25px; 
                border-radius: 25px; 
                margin: 0 10px; 
                cursor: pointer; 
                text-decoration: none; 
                display: inline-block; 
                font-size: 1em;
            }}
            .btn:hover {{ background: #2980b9; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Social Media Sentiment Analysis</h1>
                <p>🚀 Real-time insights from your social media data</p>
                <p><small>📈 Total posts analyzed: {len(analyzer.results):,} | ⏰ Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number total">{summary.get('total_posts', 0):,}</div>
                    <div>📊 Total Posts</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number positive">{sentiment_data.get('positive', 0):,}</div>
                    <div>😊 Positive Posts</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number negative">{sentiment_data.get('negative', 0):,}</div>
                    <div>😢 Negative Posts</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number neutral">{sentiment_data.get('neutral', 0):,}</div>
                    <div>😐 Neutral Posts</div>
                </div>
            </div>
            
            <div class="chart-section">
                <div class="chart-container">
                    <h3 class="chart-title">📈 Sentiment Distribution</h3>
                    <canvas id="sentimentChart" width="400" height="200"></canvas>
                </div>
            </div>
            
            <div class="insights">
                <div class="insights-header">💡 Key Insights</div>
                <div class="insights-content">
    """
    
    # Add insights
    for insight in summary.get('key_insights', []):
        html += f'<div class="insight-item">• {insight}</div>'
    
    html += f"""
                </div>
            </div>
            
            <div class="controls">
                <a href="/data" class="btn">📥 View Raw Data</a>
                <button onclick="location.reload()" class="btn">🔄 Refresh Dashboard</button>
            </div>
        </div>

        <script>
            // Create sentiment chart
            const ctx = document.getElementById('sentimentChart').getContext('2d');
            new Chart(ctx, {{
                type: 'doughnut',
                data: {{
                    labels: ['😊 Positive', '😢 Negative', '😐 Neutral'],
                    datasets: [{{
                        data: [{sentiment_data.get('positive', 0)}, {sentiment_data.get('negative', 0)}, {sentiment_data.get('neutral', 0)}],
                        backgroundColor: ['#27ae60', '#e74c3c', '#95a5a6'],
                        borderWidth: 3,
                        borderColor: '#fff'
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{
                            position: 'bottom',
                            labels: {{
                                padding: 20,
                                font: {{
                                    size: 14
                                }}
                            }}
                        }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    return html

@app.route('/')
def dashboard():
    """Main dashboard route"""
    return create_simple_html()

@app.route('/data')
def data():
    """Raw data endpoint"""
    if not analyzer:
        return jsonify({'error': 'No data available'})
    
    return jsonify({
        'summary': analyzer.summary,
        'sample_results': analyzer.results[:50],  # Show first 50 results
        'total_count': len(analyzer.results)
    })

def run_analysis():
    """Run the sentiment analysis"""
    global analyzer
    
    print("🚀 SOCIAL MEDIA SENTIMENT ANALYSIS")
    print("="*50)
    
    # Check if dataset exists
    if not os.path.exists('sentimentdataset.csv'):
        print("❌ ERROR: Dataset file 'sentimentdataset.csv' not found!")
        print("💡 Please ensure the CSV file is in the same directory.")
        return False
    
    # Initialize and run analysis
    analyzer = SentimentAnalyzer()
    
    if not analyzer.load_and_clean_data():
        print("❌ ERROR: Failed to load dataset")
        return False
    
    if not analyzer.analyze_dataset():
        print("❌ ERROR: Failed to analyze dataset")
        return False
    
    # Print summary to console
    analyzer.print_summary()
    
    # Save results to files
    try:
        with open('sentiment_results.json', 'w', encoding='utf-8') as f:
            json.dump(analyzer.results, f, indent=2, ensure_ascii=False)
        with open('sentiment_summary.json', 'w', encoding='utf-8') as f:
            json.dump(analyzer.summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to:")
        print(f"   • sentiment_results.json ({len(analyzer.results):,} records)")
        print(f"   • sentiment_summary.json (analysis summary)")
    except Exception as e:
        print(f"⚠️ Warning: Could not save files: {e}")
    
    return True

def start_web_server():
    """Start the Flask web server"""
    print("\n🌐 Starting web dashboard...")
    print("🔗 Dashboard URL: http://localhost:5000")
    print("📊 Data API: http://localhost:5000/data")
    print("⏹️ Press Ctrl+C to stop the server")
    
    # Auto-open browser
    def open_browser():
        time.sleep(2)
        try:
            webbrowser.open('http://localhost:5000')
            print("🌐 Browser opened automatically")
        except:
            print("💡 Please manually open: http://localhost:5000")
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n👋 Server stopped successfully")

def main():
    """Main execution function"""
    # Run analysis first
    if not run_analysis():
        print("\n❌ Analysis failed. Please check your dataset and try again.")
        input("Press Enter to exit...")
        return
    
    # Ask user about web dashboard
    print("\n" + "="*50)
    print("✅ Analysis completed successfully!")
    response = input("🚀 Start web dashboard? (y/n): ").lower().strip()
    
    if response in ['y', 'yes', '']:
        start_web_server()
    else:
        print("\n✅ Analysis complete!")
        print("💡 Results are saved in JSON files")
        print("🔄 Run this script again and choose 'y' to start the web dashboard")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
