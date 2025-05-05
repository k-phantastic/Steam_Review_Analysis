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
![Figure 1](/resources/screenshots/top10gamesReviewStatistics_SAMPLEDDATA.png)

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
