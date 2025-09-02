# CS 506 Final Project: Predicting Housing Prices Using Real Estate Data
- Description
  - The project aims to predict housing prices based on the size of the house, location, number of rooms, proximity to amenities, and neighborhood characteristics. This will be done using both traditional regression models and more machine learning techniques. Additionally, the project will explore the impact of listing descriptions on prices using sentiment analysis, and segment neighborhoods using clustering methods.
- Goal
  - Develop accurate housing price prediction models using both traditional regression
and advanced machine learning techniques
  - Analyze the impact of listing descriptions on house prices through sentiment analysis
  - Identify and categorize neighborhood segments using clustering methods
  - Create an interpretable and practical tool for real estate price estimation

- Dataset Description
  - The primary dataset consists of Boston housing listings collected from Redfin and stored in
processed_listings.csv. Key features include:
    - Unique listing identifiers
    - Price points
    - Physical characteristics (bedrooms, bathrooms, square footage)
    - Detailed property descriptions
    - Key property details including lot size, year built, and additional amenities
    - Transportation accessibility scores (walking, transit, and biking)
  - Initially, we planned to obtain the data from Zillow, however, Redfin had fewer protections regarding web scraping; no captcha, and less IP limiting, as well as a better page layout to perform the scraping.

- Data Collection
  - To collect the data, we used Selenium WebDriver alongside with Mullvad to rotate IP's to scrape the data from Redfin.
  - We first designed the web scraper (redfin-scrape.py && redfin-process.py && runner.sh), where we implemented pagination handling to collect links from all available pages.
  - We stored these links in links.csv with unique indices.
  - Used Firefox WebDriver with appropriate wait times to ensure reliable data collection.
  - Then we designed redfin-scrape.py && redfin-process.py to actually scrape the individual information of each lisitng.

- Data Cleaning
  - Numerical features processing
    - Standardized price values by removing non-numeric characters
    - Converted square footage to numerical format
    - Cleaned bedroom and bathroom counts to pure numeric values
    - Extracted numerical scores from walkscore ratings (e.g., "93/100" → 93)
  - Text data processing
    - Parsed key_details field to extract structured information:
      - Property age/year built
      - Lot size
      - Price per square foot
      - Parking information
      - Additional amenities
    - Cleaned and standardized property descriptions
    - Extracted and separated walk, transit, and bike scores from the walkscore text
  -  This process also enabled sentiment analysis by preparing the text for NLP tools.
- Preliminary Visualization
  - A right-skewed price distribution highlighted the prevalence of lower-priced properties with a few luxury outliers.
  - Scatter plots revealed a positive correlation between square footage and price, with variance increasing for larger properties.
  - Heatmaps demonstrated strong correlations between square footage and price, bathrooms and price, and moderate correlations with Walk Score and property price.
- Model Development
  - Linear Regression:
    - Served as the baseline model, leveraging features like location, size, and room count. Achieved an R² score of 0.68 on test data.
  - Random Forest Regression:
    - Significantly improved accuracy, achieving an R² score of 0.85. Features like Walk Score and Transit Score were strong contributors to the model’s predictive capability.
   
- Clustering Analysis
  - Neighborhood clustering using KMeans provided insights into market segmentation. Properties were grouped into three primary clusters: affordable (≤ $600,000), mid-range ($600,001 - $1,200,000), and luxury (≥ $1,200,001). Scatter plots illustrated clear segmentation, with luxury properties concentrated in high-demand areas like downtown Boston.
- Sentiment Analysis:
  - Sentiment analysis of listing descriptions revealed interesting trends. Positive sentiment correlated with higher prices, while neutral or negative sentiment had limited impact. However, the overall effect of sentiment scores on model performance was minimal.
 
- Results
  - Model Evaluation Metrics
    - Linear Regression
      - Mean Absolute Error (MAE): $124,000
      - Mean Squared Error (MSE): $275,000,000
      - R² Score: 0.68
    - Random Forest Regression
      - MAE: $95,000
      - MSE: $180,000,000
      - R² Score: 0.85
  - Clustering Insights
    - Affordable Properties: Priced at or below $600,000, these properties are typically smaller and located further from city centers.
    - Mid-Range Properties: Falling between $600,001 and $1,200,000, these homes represent the largest market segment, balancing size and loc
    - Luxury Properties: Priced above $1,200,000, these listings dominate prime locations and often boast extensive amenities.


  
 ![housing_analysis_results](https://github.com/user-attachments/assets/c15b71d9-1637-4fc7-bdd9-650e16bc8cfe)

- Video presentation
  - https://youtu.be/gFyMkkLTR7I

### How To Run  
- requirements for scraping
  - python, pip, mullvad cli
  - first run `make venv` followed by `make install`
  - then run `make scrape` followed by `make extract`
  - this will create our data sets
- requirements for running the model
  - simply run `make`
- cleaning up
  - `make clean`