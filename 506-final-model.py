import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import re
from typing import Dict, Tuple

class HousingPricePredictionPipeline:
    def __init__(self):
        """Initialize the pipeline with necessary components."""
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.linear_model = LinearRegression()
        self.rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        
        try:
            nltk.download('vader_lexicon', quiet=True)
            self.sia = SentimentIntensityAnalyzer()
        except Exception as e:
            print(f"Warning: Could not initialize NLTK components. Error: {e}")
            self.sia = None

    def parse_walkscore(self, walkscore_str: str) -> Tuple[float, float, float]:
        """Parse walkscore string into separate numeric scores."""
        try:
            if pd.isna(walkscore_str):
                return 0, 0, 0
                
            scores = {'walk': 0, 'transit': 0, 'bike': 0}
            parts = str(walkscore_str).split(';')
            
            for part in parts:
                part = part.strip()
                if 'Walk:' in part:
                    scores['walk'] = float(re.search(r'(\d+)/100', part).group(1))
                elif 'Transit:' in part:
                    scores['transit'] = float(re.search(r'(\d+)/100', part).group(1))
                elif 'Bike:' in part:
                    scores['bike'] = float(re.search(r'(\d+)/100', part).group(1))
                    
            return scores['walk'], scores['transit'], scores['bike']
        except Exception as e:
            print(f"Error parsing walkscore: {e}")
            return 0, 0, 0

    def extract_text_features(self, text: str) -> tuple:
        """Extract sentiment features from text."""
        if pd.isna(text):
            return 0, 0
        
        text = str(text)
        sentiment = self.sia.polarity_scores(text)['compound'] if self.sia else 0
        subjectivity = TextBlob(text).sentiment.subjectivity
        return sentiment, subjectivity
    
    def optimize_random_forest(self):
        """Optimize Random Forest hyperparameters using GridSearchCV."""
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1
        )
        
        grid_search.fit(self.X_train_scaled, self.y_train)
        self.rf_model = grid_search.best_estimator_
        
        return grid_search.best_params_, grid_search.best_score_

    def load_and_preprocess_data(self, filepath: str) -> None:
        """Load and preprocess the housing data."""
        try:
            print(f"\nLoading data from {filepath}...")
            self.data = pd.read_csv(filepath)
            
            # Remove parking space listings and outliers
            self.data = self.data[self.data['sq_ft'].notna()]
            self.data = self.data[self.data['price'] < self.data['price'].quantile(0.99)]
            self.data = self.data[self.data['sq_ft'] < self.data['sq_ft'].quantile(0.99)]
            
            print(f"After removing outliers: {len(self.data)} properties")
            
            # Parse walkscores
            walk_scores = self.data['walkscore'].apply(self.parse_walkscore)
            self.data['walk_score'] = [score[0] for score in walk_scores]
            self.data['transit_score'] = [score[1] for score in walk_scores]
            self.data['bike_score'] = [score[2] for score in walk_scores]
            
            # Extract text features
            sentiment_subj = [self.extract_text_features(desc) for desc in self.data['description']]
            self.data['description_sentiment'] = [s[0] for s in sentiment_subj]
            self.data['description_subjectivity'] = [s[1] for s in sentiment_subj]
            
            # Select features for modeling
            features = ['beds', 'baths', 'sq_ft', 'walk_score', 'transit_score', 'bike_score',
                       'description_sentiment', 'description_subjectivity']
            
            # Prepare data for modeling
            X = self.data[features]
            y = self.data['price']
            
            # Handle missing values
            X = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            print("\nData preprocessing completed successfully")
            print(f"Final dataset shape: {self.data.shape}")
            self.print_data_summary()
            
        except Exception as e:
            print(f"Error in data preprocessing: {e}")
            raise

    def print_data_summary(self):
        """Print summary statistics of the dataset."""
        print("\nDataset Summary:")
        print(f"Total properties: {len(self.data)}")
        print(f"Average price: ${self.data['price'].mean():,.2f}")
        print(f"Median price: ${self.data['price'].median():,.2f}")
        print(f"Average square footage: {self.data['sq_ft'].mean():,.2f}")
        print(f"Average bedrooms: {self.data['beds'].mean():.1f}")
        print(f"Average bathrooms: {self.data['baths'].mean():.1f}")
        print(f"Average walk score: {self.data['walk_score'].mean():.1f}")
        print(f"Average transit score: {self.data['transit_score'].mean():.1f}")
        print(f"Average bike score: {self.data['bike_score'].mean():.1f}")

    def evaluate_model(self, y_pred, model_name: str) -> Dict[str, float]:
        """Evaluate model performance with multiple metrics."""
        return {
            'r2': r2_score(self.y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred)),
            'mae': mean_absolute_error(self.y_test, y_pred),
            'mape': np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100
        }

    def train_models(self):
        """Train and evaluate both models."""
        # Linear Regression
        self.linear_model.fit(self.X_train_scaled, self.y_train)
        linear_pred = self.linear_model.predict(self.X_test_scaled)
        linear_metrics = self.evaluate_model(linear_pred, "Linear Regression")
        
        # Optimize and train Random Forest
        print("\nOptimizing Random Forest parameters...")
        best_params, best_score = self.optimize_random_forest()
        rf_pred = self.rf_model.predict(self.X_test_scaled)
        rf_metrics = self.evaluate_model(rf_pred, "Random Forest")
        
        return linear_metrics, rf_metrics, best_params

    def perform_clustering(self) -> None:
        """Perform K-means clustering on the dataset."""
        try:
            # Use relevant features for clustering
            clustering_features = ['sq_ft', 'beds', 'baths', 'price']
            cluster_data = self.X_train.copy()
            cluster_data['price'] = self.y_train
            
            # Scale the data for clustering
            cluster_data_scaled = self.scaler.fit_transform(cluster_data[clustering_features])
            
            # Perform clustering
            self.cluster_labels = self.kmeans.fit_predict(cluster_data_scaled)
            
            # Analyze clusters
            cluster_stats = pd.DataFrame({
                'Cluster': self.cluster_labels,
                'Price': self.y_train,
                'Square_Feet': self.X_train['sq_ft'],
                'Bedrooms': self.X_train['beds']
            }).groupby('Cluster').agg({
                'Price': ['mean', 'count'],
                'Square_Feet': 'mean',
                'Bedrooms': 'mean'
            })
            
            print("\nCluster Statistics:")
            print(cluster_stats)
            
        except Exception as e:
            print(f"Error in clustering: {e}")
            self.cluster_labels = None

    def analyze_price_segments(self):
        """Analyze price segments in the market."""
        price_segments = pd.qcut(self.data['price'], q=4, labels=['Low', 'Medium', 'High', 'Luxury'])
        segment_stats = self.data.groupby(price_segments).agg({
            'price': ['mean', 'count'],
            'sq_ft': 'mean',
            'beds': 'mean',
            'baths': 'mean',
            'walk_score': 'mean'
        })
        return segment_stats

    def generate_visualizations(self) -> None:
        """Generate comprehensive visualizations."""
        plt.style.use('default')
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(20, 25))
        
        # 1. Price Distribution
        ax1 = plt.subplot(421)
        sns.histplot(data=self.data, x='price', bins=30, ax=ax1)
        ax1.set_title('Price Distribution')
        ax1.set_xlabel('Price ($)')
        ax1.set_ylabel('Count')
        
        # 2. Price vs Square Footage
        ax2 = plt.subplot(422)
        sns.regplot(data=self.data, x='sq_ft', y='price', ax=ax2, scatter_kws={'alpha': 0.5})
        ax2.set_title('Price vs Square Footage')
        ax2.set_xlabel('Square Footage')
        ax2.set_ylabel('Price ($)')
        
        # 3. Feature Correlations
        ax3 = plt.subplot(423)
        features_for_corr = ['price', 'beds', 'baths', 'sq_ft', 'walk_score', 'transit_score', 'bike_score']
        correlation = self.data[features_for_corr].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax3)
        ax3.set_title('Feature Correlation Heatmap')
        
        # 4. Model Performance
        ax4 = plt.subplot(424)
        linear_pred = self.linear_model.predict(self.X_test_scaled)
        rf_pred = self.rf_model.predict(self.X_test_scaled)
        
        models = ['Linear Regression', 'Random Forest']
        r2_scores = [r2_score(self.y_test, linear_pred), r2_score(self.y_test, rf_pred)]
        ax4.bar(models, r2_scores)
        ax4.set_title('Model R² Scores Comparison')
        ax4.set_ylabel('R² Score')
        ax4.set_ylim(0, 1)
        
        # 5. Price Segments
        ax5 = plt.subplot(425)
        segments = pd.qcut(self.data['price'], q=4)
        sns.boxplot(x=segments, y='sq_ft', data=self.data, ax=ax5)
        ax5.set_title('Square Footage by Price Segment')
        ax5.set_xlabel('Price Segment')
        ax5.set_ylabel('Square Footage')
        plt.xticks(rotation=45)
        
        # 6. Location Scores vs Price
        ax6 = plt.subplot(426)
        sns.scatterplot(data=self.data, x='walk_score', y='price', alpha=0.5, ax=ax6)
        ax6.set_title('Walk Score vs Price')
        ax6.set_xlabel('Walk Score')
        ax6.set_ylabel('Price ($)')
        
        # 7. Sentiment Analysis Impact
        ax7 = plt.subplot(427)
        sns.scatterplot(data=self.data, x='description_sentiment', y='price', alpha=0.5, ax=ax7)
        ax7.set_title('Description Sentiment vs Price')
        ax7.set_xlabel('Sentiment Score')
        ax7.set_ylabel('Price ($)')
        
        # 8. Clustering Results
        ax8 = plt.subplot(428)
        if hasattr(self, 'cluster_labels'):
            scatter = ax8.scatter(
                self.X_train_scaled[:, 2],  # sq_ft
                self.X_train_scaled[:, 0],  # beds
                c=self.cluster_labels,
                cmap='viridis'
            )
            plt.colorbar(scatter, ax=ax8)
            ax8.set_title('Property Clusters')
            ax8.set_xlabel('Square Footage (scaled)')
            ax8.set_ylabel('Bedrooms (scaled)')
        
        plt.tight_layout()
        plt.savefig('housing_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualizations saved successfully to 'housing_analysis_results.png'")


def main():
    # Initialize pipeline
    pipeline = HousingPricePredictionPipeline()
    
    print("Starting housing price prediction pipeline...")
    
    # Load and preprocess data
    print("\nStep 1: Loading and preprocessing data...")
    pipeline.load_and_preprocess_data('redfin_processed_listings.csv')
    
    # Train and evaluate models
    print("\nStep 2: Training and evaluating models...")
    linear_metrics, rf_metrics, best_rf_params = pipeline.train_models()
    
    print("\nLinear Regression Results:")
    for metric, value in linear_metrics.items():
        print(f"- {metric}: {value:.4f}")
    
    print("\nRandom Forest Results:")
    print("Best parameters:", best_rf_params)
    for metric, value in rf_metrics.items():
        print(f"- {metric}: {value:.4f}")
    
    # Analyze price segments
    print("\nStep 3: Analyzing price segments...")
    segment_stats = pipeline.analyze_price_segments()
    print("\nPrice Segment Statistics:")
    print(segment_stats)
    
    # Perform clustering
    print("\nStep 4: Performing clustering analysis...")
    pipeline.perform_clustering()
    
    # Generate visualizations
    print("\nStep 5: Generating visualizations...")
    pipeline.generate_visualizations()

if __name__ == "__main__":
    main()
