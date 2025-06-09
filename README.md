# Steam Review Analysis

## Introduction
We chose this dataset because all of us happened to be avid gamers, so we naturally had an interest in the topic. For the dataset itself, there was a sufficient amount of data with more than enough attributes to have a lot of options for creating models. This includes how we would pre-process the data, which attributes to use, the purpose of our model, and the actual type of model itself. In terms of having a good predictive mode, our model predicting the helpfulness of user reviews can be helpful for highlighting the most relevant reviews when users are thinking about buying a game, thus boosting sales, or it could give insight into the most common sentiments on the pros and cons of the game, thus allowing developers to target aspects of their game that they can improve.  

## Overview

In this project we will examine a dataset containing more than 100 million Steam reviews and attempt to create two machine learning models: one to predict the helpfulness of user reviews, and the other to create distinct clusters of Steam reviewers. 

Data will be processed and stored on the San Diego Computer Center (SDSC) and their Expanse system with primary usage of Spark through PySpark. 

Initial data from Kaggle includes statistics about the review and reviewer including their playtime, number of games owned, and number of positive ratings that the review got. Steam also provides a parameter described as a "weighted vote score" of how useful a review is. 

Summary of the column labels and features can be found in [DATA_LABELS.md](/resources/DATA_LABELS.md)

## Setup
With the prerequisite usage of SDSC Expanse the following were key parameters outside of performance ([reference class setup](https://github.com/ucsd-dsc232r/group-project)): 
* Singularity Image File Location (course's singularity image): `~/esolares/spark_py_latest_jupyter_dsc232r.sif` 
* Environment Modules to be loaded: `singularitypro`
* Cores: 32
* Memory per node (gb): 128
* Working directory: home
* Type: JupyterLab

Provided in [NB0_Original_Sampled_Data.ipynb](/notebooks/NB0_Original_Sampled_Data.ipynb) is a cell in which `pip install -r ../requirements.txt` can be performed to install necessary packages. Additionally, due to favorable runtimes in the pre-processing, slurm scheduling was not needed. 

> Source of the data is located here: [100 Million+ Steam Reviews | Kaggle](https://www.kaggle.com/datasets/kieranpoc/steam-reviews/data)

## [Notebook](/notebooks/) Index 

* [NB0_Original_Sampled_Data.ipynb](/notebooks/NB0_Original_Sampled_Data.ipynb) -  Used to create n samples of overall data 
* [NB1_Sampled_EDA.ipynb](/notebooks/NB1_Sampled_EDA.ipynb.ipynb) - Exploratory visualizations on the sampled data
* [NB2_Data_Cleaning.ipynb](/notebooks/NB0_Original_Sampled_Data.ipynb) - Processing of total dataset, creation of cleaned file
* [NB3_Additional_Visualizations.ipynb](/notebooks/NB3_Additional_Visualizations.ipynb) - Visualizations on cleaned data
* [NB4_Model_1.ipynb](/notebooks/NB4_Model_1.ipynb) - Initial machine learning model on processed data

## Methods
The dataset uses the pre-processed DataFrame (df_cleaned) with various attributes related to:

* Reviewer behavior (e.g., number of games owned, playtime stats)
* Engagement metrics (e.g., votes, funny votes, comment count)
* Review context (e.g., whether the review was written during early access)

For this initial model, we selected only numerical features to keep the pipeline simple and interpretable. The target variable weighted_vote_score was cast to a DoubleType and renamed as label for compatibility with Spark ML.

The particular columns selected include (Reference: [DATA_LABELS.md](/resources/DATA_LABELS.md)):
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

### Initial Visualizations
>Key Files: [NB1_Sampled_EDA.ipynb](/notebooks/NB1_Sampled_EDA.ipynb.ipynb) and [NB3_Additional_Visualizations.ipynb](/notebooks/NB3_Additional_Visualizations.ipynb) used to generate visualizations, with the former used on a sample of the data

General statistics and visualizations shown from clean data, used to brainstorm future ML models and processing

##### Figure 1: Which games have the most reviews or most passionate reviewers (as seen in total hours)
![Figure 1](/resources/screenshots/top10gamesReviewStatistics_ALLDATA.png)

##### Figure 2 (sampled data): Weighted Vote Score vs Review Vote (1 being upvote) 
![Figure 2](/resources/screenshots/boxplot.png)

##### Figure 3 (sampled data): Heatmap of a features hypothesized to be useful for ML
![Figure 3](/resources/screenshots/heatmap.png)

##### Figure 4 (sampled data): Histogram of Weighted Vote Score (pre-filtered score of >0.8)
![Figure 4](/resources/screenshots/histogram.png)

##### Figure 5 (sampled data): Scatterplot of Author Playtime (at time of review) vs Weighted Vote Score, with weak correlation 
![Figure 5](/resources/screenshots/scatter.png)

### Preprocessing (Milestone 2) 
> Key Files: [NB0_Original_Sampled_Data.ipynb](/notebooks/NB0_Original_Sampled_Data.ipynb) in creation of sampled datasets, [NB2_Data_Cleaning.ipynb](/notebooks/NB0_Original_Sampled_Data.ipynb) was used to process initial data
* Created sample set of data using NB0 for initial brainstorming and visualization. 
* Performed initial cleanup by filtering out weighted vote scores of 0 to capture removal of reviews that are not useful, are from bots, or are potentially spam. Parameter can be filtered further as needed for machine learning methods
* Re-established dataframe schema and dropped null values for corrupted rows (comma handling errors)
* Filtered by weighted vote score between 0-1 as additional means of removing errant data
* Dropped columns that are for sure not to be used: hidden_in_steam_china, steam_china_location, review
* Converted playtime statistics to floats and converted from minutes played to hours played (to align with Steam current interface)
* Dropped any potential duplicates in the sampling of the data

---

## Model 1 Overview: Helpfulness Score Regression (Milestone 3)

> Key Files: [Model 1 Notebook](/notebooks/NB4_Model_1.ipynb), used in conjunction with the [cleaned data set](https://drive.google.com/file/d/12S7orw3WFnilznJpWCwaLdhBPNwHmMcu/view?usp=sharing) (obtained following initial [cleanup](/notebooks/NB0_Original_Sampled_Data.ipynb))

### Objective: 
The goal of this model is to build a regression model to predict the helpfulness score (represented by the weighted_vote_score) of Steam reviews using various numeric features derived from the review metadata and the reviewer’s profile.

### Modeling Approach

We used the Spark ML pipeline to streamline preprocessing and model training:   
* VectorAssembler: Combined all numeric features into a single vector.
* StandardScaler: Standardized features to zero mean and unit variance to improve gradient descent convergence.
* LinearRegression: Used as an interpretable baseline model.

In order to accurately assess model performance, we tested on various train/test splits with the train values set to: [0.1, 0.2, 0.5, 0.8, 0.9]. The model was evaluated using Root Mean Squared Error (RMSE).

Ground truth: weighted_vote_score

##### Figure 6
![Figure 6](/resources/screenshots/ground_truth.png)
The above figure depicts the first 100 values of the ground truth with its corresponding index. 

## Results 
#### 🔍 Feature Importance Ranking

| Rank | Feature                          | RMSE Reduction   |
|------|----------------------------------|------------------|
| 1    | `votes_up`                       | 0.2027           |
| 2    | `voted_up`                       | 0.1329           |
| 3    | `votes_funny`                    | 0.1186           |
| 4    | `comment_count`                  | 0.0695           |
| 5    | `author_num_reviews`             | 0.0669           |
| 6    | `author_num_games_owned`         | 0.0304           |
| 7    | `author_playtime_at_review`      | 0.0170           |
| 8    | `steam_purchase`                 | 0.0165           |
| 9    | `author_playtime_forever`        | 0.0146           |
| 10   | `written_during_early_access`    | 0.0076           |
| 11   | `received_for_free`              | 0.0054           |
| 12   | `author_playtime_last_two_weeks` | 0.0053           |
| 13   | `author_last_played`             | 0.0025           |

**Recommended number of features (elbow point):** 4 or 5  
**Selected features:** `votes_up`, `voted_up`, `votes_funny`, `comment_count`, 'author_num_reviews'

---

#### 📈 Performance Summary (Forward Feature Selection)

| Features Used | Test RMSE | Improvement |
|---------------|-----------|-------------|
| 1             | 0.0693    | 0.0000      |
| 2             | 0.0684    | 0.0009      |
| 3             | 0.0682    | 0.0011      |
| 4             | 0.0681    | 0.0012      |
| 5             | 0.0679    | 0.0014      |
| 6             | 0.0678    | 0.0015      |
| 7             | 0.0678    | 0.0015      |
| 8             | 0.0678    | 0.0015      |
| 9             | 0.0678    | 0.0015      |
| 10            | 0.0677    | 0.0015      |
| 11            | 0.0677    | 0.0015      |
| 12            | 0.0677    | 0.0015      |
| 13            | 0.0677    | 0.0015      |

##### Figure 7
![Figure 7](/resources/screenshots/rmse_features.png)

The linear regression model achieved consistent performance across multiple runs, with training RMSE values converging to around 0.0679 and test RMSE values converging to around 0.0677. We can see that the improvement in the RMSEs seems to drop off around 4 or 5 features, and this finding is in line with the rank and RMSE reduction of each feature. So for our model selection, to avoid overfitting, we should choose a model using the top 4 or 5 features for the best performance.

## Model 1 Discussion:


##### Figure 8
![Figure 8](/resources/screenshots/pred_overlay.png)

In the figure above, an overlay of the first 100 ground truths and the first 100 predictions of the 80/20 train-test split, suggests that the model may actually be underfitting and may need tuning in order to properly and accurately predict the weighted_vote_score.

## Model 1 Conclusion
There are multiple potential ways to improve the model performance, including incorporating additional features such as text-based sentiment analysis, tuning hyperparameters by exploring methods such as cross-validation, and applying target transformations such as log scale. 

* Reducing regularization may assist in preserving the real signal in our chosen features to survive.
* Incorporating additional features such as text-based sentiment analysis.
* Utilizing cross-validation so the real signal in our chosen features survives.

Currently, we are considering using Gradient Boosted Trees (GBTRegressor) and PCA with K-Means Clustering.

Gradient Boosted Trees are well-suited for capturing complex, non-linear relationships in the data. They often outperform linear models in accuracy and provide feature importance scores, offering insight into which factors (like playtime or votes) most influence helpfulness.

PCA + K-Means is an unsupervised approach to uncover patterns in reviewer behavior. PCA reduces noise and highlights key feature combinations, while K-Means can group similar reviewers. This can reveal distinct user types (e.g., casual vs. dedicated reviewers) and generate new features to improve predictive models.

Both models aid in our goal in this project of identifying potential contributions to algorithms that highlight relevant, personalized user recommendations or improve developer pricing.

---
## Statement of Collaboration
Layth Marabeh: Coder/Writer
* He participated in group meetings once/twice per week to discuss progress updates, solve bugs, suggest additions to be made in the code and README. He wrote most of the code for Milestone 3 data pre-processing, such as scaling the data, imputing the data, encoding the data, feature expansion, filtering by weighted vote score, and dropping unnecessary columns. He is also responsible for the coding of model 1 and writing the majority of the Milestone 3 README.

Danny Xia
* He wrote or edited many of the written documents, including the abstract from Milestone 2 and the final written report/README.md. He also created data visualizations as part of the data exploration milestone and conceptualized the idea for the models. Finally, he wrote and edited the code for finding the best features and models for the linear regression model and creating the plots that compared the train RMSE and test RMSE in relation to the number of features. He collaborated with group members in meetings to discuss progress updates and provide suggestions on the best approaches to solving the objectives from the milestones.
## Team Members
* Danny Xia ([@dannyxia7](https://github.com/dannyxia7))
* Khanh Phan ([@khp023](https://github.com/k-phantastic))
* Layth Marabeh ([@lmarabeh](https://github.com/lmarabeh))

## Sources
* [100 Million+ Steam Reviews | Kaggle](https://www.kaggle.com/datasets/kieranpoc/steam-reviews/data)
