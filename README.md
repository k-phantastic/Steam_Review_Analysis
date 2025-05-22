# Steam Review Analysis

## Overview

In this project we will examine a dataset containing more than 100 million Steam reviews and attempt to create two machine learning models: one to predict the helpfulness of user reviews, and the other to create distinct clusters of Steam reviewers. Data will be processed and stored on a cluster on San Diego Computer Center (SDSC) with primary usage of Spark through PySpark. 

Initial data from Kaggle includes statistics about the review and reviewer including their playtime, number of games owned, and number of positive ratings that the review got. Steam also provides a parameter described as a "weighted vote score" of how useful a review is. 

## Methods

#### Preprocessing
* Performed initial cleanup by filtering out weighted vote scores of 0 to capture removal of reviews that are not useful, are from bots, or are potentially spam. Parameter can be tweaked further as needed for machine learning methods
* Re-established dataframe schema and dropped null values for corrupted rows
* Filtered by weighted vote score between 0-1
* Dropped columns that are for sure not to be used: hidden_in_steam_china, steam_china_location, review
* Converted playtime statistics to floats and converted from minutes played to hours played (to align with Steam current interface)
* Dropped any potential duplicates in the sampling of the data

#### Visualizations

General statistics obtained to see popularity of games such as total playtime of reviewers and total number of reviews

##### Figure 1
![Figure 1](/resources/screenshots/top10gamesReviewStatistics_ALLDATA.png)

##### Figure 2
![Figure 2](/resources/screenshots/boxplot.png)

##### Figure 3
![Figure 3](/resources/screenshots/heatmap.png)

##### Figure 4
![Figure 4](/resources/screenshots/histogram.png)

##### Figure 5
![Figure 5](/resources/screenshots/scatter.png)


## Team Members
* Danny Xia ([@dannyxia7](https://github.com/dannyxia7))
* Khanh Phan ([@khp023](https://github.com/khp023))
* Layth Marabeh ([@lmarabeh](https://github.com/lmarabeh))

## Sources
* [100 Million+ Steam Reviews | Kaggle](https://www.kaggle.com/datasets/kieranpoc/steam-reviews/data)

# Steam Review Analysis: Milestone 3

## Pre Processing 
All major preprocessing was completed in milestone 1.
    •	Filtered by weighted vote score between 0-1
	•	Dropped columns that are for sure not to be used: hidden_in_steam_china, steam_china_location, review
	•	Converted playtime statistics to floats and converted from minutes played to hours played (to align with Steam current interface)
	•	Dropped any potential duplicates in the sampling of the data

## Model 1 Overview: Helpfulness Score Regression
#### Objective: 
The goal of this project was to build a regression model to predict the helpfulness score (represented by the weighted_vote_score) of Steam reviews using various numeric features derived from the review metadata and the reviewer’s profile.

#### Methods
The dataset included a pre-processed DataFrame (df_cleaned) with various attributes related to:

* Reviewer behavior (e.g., number of games owned, playtime stats)
* Engagement metrics (e.g., votes, funny votes, comment count)
* Review context (e.g., whether the review was written during early access)

For this initial model, we selected only numerical features to keep the pipeline simple and interpretable. The target variable weighted_vote_score was cast to a DoubleType and renamed as label for compatibility with Spark ML.

The particular columns selected include:
* author_num_games_owned
* author_num_reviews
* author_playtime_forever
* author_playtime_last_two_weeks
* author_playtime_at_review
* author_last_played
* voted_up
* votes_up
* votes_funny
* comment_count
* steam_purchase
* received_for_free
* written_during_early_access

Modeling Approach

We used the Spark ML pipeline to streamline preprocessing and model training:
* VectorAssembler: Combined all numeric features into a single vector.
* StandardScaler: Standardized features to zero mean and unit variance to improve gradient descent convergence.
* LinearRegression: Used as an interpretable baseline model.

In order to accurately assess model performance, we tested on various train/test splits with the train values set to: [0.1, 0.2, 0.5, 0.8, 0.9] The model was evaluated using Root Mean Squared Error (RMSE).

Ground truth: weighted_vote_score
#### Results 
Train error Vs Test error accross train/test splits:

Training RMSE values: [0.06780513676116458, 0.06809054266893783, 0.06793147050810851, 0.06786924970348243, 0.06783383929472699]

Testing RMSE values: [0.0677858225116503, 0.06795994752549109, 0.06785859680397434, 0.06774141874455455, 0.06766356133129071]

##### Figure 1
![Figure 1](/resources/screenshots/train_vs_test_RMSE.png)

As can be seen by the graph abpve, train and test error are similar accross all the train/test splits. This indicates that the model is neither overfitting or underfitting.

#### Next models 
Currently, we are considering using Gradient Boosted Trees (GBTRegressor) and PCA with K-Means Clustering.

Gradient Boosted Trees are well-suited for capturing complex, non-linear relationships in the data. They often outperform linear models in accuracy and provide feature importance scores, offering insight into which factors (like playtime or votes) most influence helpfulness.

PCA + K-Means is an unsupervised approach to uncover patterns in reviewer behavior. PCA reduces noise and highlights key feature combinations, while K-Means can group similar reviewers. This can reveal distinct user types (e.g., casual vs. dedicated reviewers) and generate new features to improve predictive models.

Both models aid in our goal in this project of identifying potential contributions to algorithms that highlight relevant, personalized user recommendations or improve developer pricing.

#### Conclusion
The linear regression model achieved consistent performance across multiple runs, with training RMSE values averaging around 0.0679 and testing RMSE values averaging around 0.0678. This indicates that the model is not overfitting and generalizes well to unseen data within the current feature set. Although the model performed reasonably well, ways to improve its performance incluce, incorperating additional features such as text based sentiment analysis, tuneing hyperparameters by exploring methods such as cross validation, and applying target transformations such as log scale. 

## Team Members
* Danny Xia ([@dannyxia7](https://github.com/dannyxia7))
* Khanh Phan ([@khp023](https://github.com/khp023))
* Layth Marabeh ([@lmarabeh](https://github.com/lmarabeh))

## Sources
* [100 Million+ Steam Reviews | Kaggle](https://www.kaggle.com/datasets/kieranpoc/steam-reviews/data)
