# Customer Intelligence Platform: Predictive Analytics for Proactive Marketing
### _Transforming Raw Data into Revenue with a Deployed, End-to-End ML System_

As part of the Data Science and Machine Learning Certificate program at University of Toronto's Data Sciences Institute, our team selected the "Customer Purchasing Behaviors" dataset to demonstrate the technical and analytical skills developed throughout the course. 

## Business Case & Objectives
**The Problem:** In today's hyper-competitive retail market, businesses face the challenge of acquiring and retaining customers to drive sustainable growth in the industry. Understanding the relationship between consumer demographics, loyalty, and spending patterns is crucial to identify ideal customer profiles to develop targeted strategies. This project explores whether customer spending can be accurately predicted from key variables such as age, annual income, and geographic region.

**Objectives:** The goals of this project are to:
1.  Build a regression model to predict customer spending based on demographic and behavioural factors.
2.  Identify the most influential features that driver purchasing behaviours.
3.  Evaluate model performance through regression metrics and analysis.

## Success Metrics
We will develop a multi-output model or a suite of interconnected models to achieve the following performance goals. *(Note: These are initial targets and may be refined after exploratory data analysis.)*
*   **CLV Prediction (Regression):**
    *   **Goal:** Accurately forecast the potential future spend of a customer.
    *   **Success Metric:** Achieve an **R² > 0.80** and **RMSE** significantly lower than the standard deviation of the purchase amount.
*   **Churn Risk Classification (Classification):**
    *   **Goal:** Identify customers have the potential to spend more. We will engineer a `growth_potential_score` target label based on low loyalty, frequency, and spend.
    *   **Success Metric:** Target a **F1-score > 0.80**, with a strong focus on **Precision (>0.75)** to ensure retention efforts are not wasted.
*   **Customer Segmentation (Clustering):**
    *   **Goal:** Discover meaningful, data-driven customer personas.
    *   **Success Metric:** Achieve a **Silhouette Score > 0.55**, indicating distinct and well-separated clusters.



## Dataset
**Source:** [Customer Purchasing Behaviors Dataset](https://www.kaggle.com/datasets/hanaksoy/customer-purchasing-behaviors)

**Profile:** 238 records, 7 features (`user_id`, `age`, `annual_income`, `purchase_amount`, `loyalty_score`, `region`, `purchase_frequency`).

**Data Strategy:** The dataset's simplicity is our opportunity to showcase advanced **feature engineering**. The objective is to create feature sets capturing behavioral patterns (e.g., `spend_per_purchase`, `spend_to_income_ratio`) that will be the true drivers of our model's performance. 


**Dataset Features:** 
| Feature             | Type          | Description                                                        |
|---------------------|---------------|--------------------------------------------------------------------|
| customer_id         | int64         | Unique ID of the customer                                          |
| age                 | int64         | The age of the customer                                            |
| annual_income       | int64         | The customer's annual income (in USD)                              |
| purchase_amount     | int64         | The annual amount of purchases made by the customer (in USD)       |
| purchase_frequency  | float64       | Frequency of customer purchases (number of times per year)         |
| region              | object        | The region where the customer lives (North, South, East, West)     |
| loyalty_score       | int64         | Customer's loyalty score (a value between 0-10)                    |


## Technical Architecture:
Our architechture is designed for reproductibility, scalability, and ease of deployment. The following tools have been used in this analysis:
| Tool/Package             | Version       | 
|--------------------------|---------------|
| jupyter notebook         | 7.4.4         | 
| python                   | 3.9.15        | 
| conda                    | 25.1.1        | 
| mlflow                   | 2.8.1         | 
| scikit-learn             | 1.3.2         | 
| pandas                   | 2.1.4         | 
| numpy                    | 1.24.3        | 
| matplotlib               | 3.8.2         | 
| seaborn                  | 0.13.0        | 
| jupyter                  | 1.0.0         | 
| python-dotenv            | 1.0.0         | 
| plotly                   | 6.0.1         | 
| click                    | 8.1.7         | 
| kneed                    | 0.8.5         | 
| os                       | Standard      | 
| math                     | Standard      | 

## Risks & Mitigation Plan
*   **Risk: Low Data Volume (238 records).**
    *   **Description:** High risk of overfitting and poor generalization to new data.
    *   **Mitigation:** This project will be framed as a **methodological proof-of-concept**. Our primary deliverable is a robust, reusable pipeline. We will use **rigorous cross-validation**, **regularization (L1/L2)**, and data augmentation techniques like **SMOTE** (for the classification task) while clearly documenting their impact.
*   **Risk: Synthetic Data Lacks Real-World Complexity.**
    *   **Description:** The data may not contain the noise, outliers, and complex correlations of a real business dataset.
    *   **Mitigation:** We will focus on building an **agnostic and robust framework**. The value lies in the pipeline's design, feature engineering logic, and the dashboard's utility, which can be easily adapted to a real-world dataset. We may introduce synthetic noise to test model robustness.
*   **Risk: Potential for Model Drift.**
    *   **Description:** In a real-world scenario, customer behavior changes over time, degrading model performance.
    *   **Mitigation:** We will design our system with this in mind. The MLOps component (MLflow) will track performance, and we will architect a **simulated model retraining pipeline**, demonstrating how the system would stay current in a production environment.
*   **Risk: High instability due to multicollinearity.**
    *   **Description:** Model coefficients will be mathematically unreliable and uninterpretable.
    *   **Mitigation:** We will avoid using linear models for final predictive task. However we will train one to demonstrate the instability of using a linear model for this dataset. 
*   **Risk: Severe class imbalance for Region Variable.**
    *   **Description:** The model has such a small sample size for the "East" category, making any conclusions made on it unreliable.
    *   **Mitigation:** We will group the "East" category with the another category to stabilize model and prevent it from being trained on statistical noise.


## Timeline
Our current timeline for the project as of Tuesday, July 22, 2025 is as follows
| Day                 | Targets                                                                    | 
|---------------------|----------------------------------------------------------------------------|
| Wednesday, July 23  | Continue progress on individually assigned model runs and check in         | 
| Thursday, July 24   | Complete all model runs, begin building conclusions                        | 
| Friday, July 25     | Final conclusions, video reflections, final documentation updates          | 
| Saturday, July 26   | Final presentation                                                         |


## Exploratory Data Analysis
*   **TBD**


## Regression Analysis & Testing
*   **TBD**

## KMeans Clustering 
<ins>**METHOD**</ins>

KMeans Clustering on the updated dataset with engineered features - two methods were reviewed for to achieve the best value of k:

   1. **Elbow method** - this is tested over a range of k clusters to calculate WCSS (*within cluster sum of squares*), and want lower values. We then identify the "elbow", which shows a sharp decrease in rate of change of WCSS as k increases. This identifies the best value of k to use as the number of clusters.
   2. **Silhouette Scoring** - for KMeans clustering, the silhouette score is a metric use to evaluate the quality of clustering, and seeing values closer to 1 indicate better clustering.<br/><br/>


For our analysis, we use the k value from the silhouette scoring, because the score helps measure how well the cluster values are distinct from the others. Furthermore:
   - We want to try and distinguish clear customer groups to identify customer purchasing behaviours,
   - Businesses looking to identify and understand customer profiles may need the additional differentiation or nuance,
   - Having more quality separation will improve targeting their ideal market(s). 

For testing, we make the following assumptions/restrictions:
   - maximum number of clusters is set to 10  
   - test iterations are also set to 10
   - random seed is set to 42<br/><br/>


To allow KMeans clustering the following fields were also encoded as follows:
| region_grouped      | age_group          | income_bracket          |  frequency_percentile          | 
|---------------------|--------------------|-------------------------|--------------------------------|
| North + East: 0     | Young_Adult: 0     | Low_Income: 0           | 0-25%: 0.25                    |
| West: 1             | Adult: 1           | Medium_Income: 1        | 25-50%: 0.50                   |
| South: 2            | Middle_Aged: 2     | High_Income: 2          | 50-75%: 0.75                   |
| -                   | Senior: 3          | -                       | 75-100%: 1.00                  |
<br/><br/>

<ins>**CLUSTERING RESULTS**</ins>

The KMeans Clustering would first run on the original base features, and then tested with base features alongside different engineered features to see if there were meaningful changes in clusters.

The silhouette scores and k values for each run were:
| Test                                | Elbow K   | Silhouette K  |  Silhouette Score  | 
|-------------------------------------|-----------|---------------|--------------------|
| Base Fields                         | 5         | 10            | 0.6110             |
| Base Fields + Core Scores           | 6         | 8             | 0.5466             |
| Base Fields + Behavioral Ratios     | 4         | 10            | 0.5574             |
| Base Fields + Key Segments/Flags    | 3         | 2             | 0.6411             |
| Base Fields + Demographics/Income   | 4         | 9             | 0.5723             |
| Base Fields + Percentiles           | 5         | 7             | 0.5654             |
| Base Fields + Log Transformed       | 5         | 9             | 0.5734             |

From the results, we see that the base run had the best silhouette score, and generated the following clusters with the following features below. Note that the clusters are ranked in descending order from highest average `loyalty_score`, `purchase_amount`, `purchase_frequency` and `annual_income`:

| Cluster                     | 0          | 6          |  8         | 4          | 2          | 9          | 5          | 7          | 1          | 3          | 
|-----------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| Customer                    | 15         | 37         | 17         | 36         | 22         | 19         | 19         | 17         | 42         | 14         |
| Size Proportion of Dataset  | 6.30%      | 15.55%     | 7.14%      | 15.13%     | 9.24%      | 7.98%      | 7.98%      | 7.14%      | 17.65%     | 5.88%      |
| Average Age                 | 54.33      | 50.05      | 47.53      | 41.89      | 39.36      | 37.42      | 32.68      | 31.41      | 27.50      | 24.00      |
| Average Income              | $74,333.33 | $70,027.03 | $67,529.41 | $61,472.22 | $59,590.91 | $57,157.89 | $52,631.58 | $51,294.12 | $44,738.10      | $32,000.00 |
| Average Purchase Amount     | $633.33    | $520.27    | $550.00    | $479.72    | $448.64    | $416.84    | $366.32    | $345.88    | $245.95      | $170.00    |
| Average Loyalty Score       | 9.43       | 8.92       | 8.50       | 7.73       | 7.21       | 6.77       | 5.91       | 5.65       | 4.30       | 3.30       |
| Average Region              | 1.60       | 1.00       | 0.00       | 2.00       | 1.00       | 0.00       | 2.00       | 0.53       | 0.05       | 1.93       |
| Average Purchase Frequency  | 27.40      | 24.81      | 23.41      | 21.53      | 20.59      | 19.74      | 17.79      | 17.35      | 14.14      | 11.07      |

Over all the clustering runs (*with the exception of the Key Segments/Flags one, as it only has 2 clusters*), all clusters formed show a very strong almost linear trend indicating that as age increases, so does purchase amount, loyalty, and purchase frequency. I.e. they all have similar results, indicating that the base fields may be enough to make informed clustering choices for this dataset. <br/><br/>

<ins>**TAKEAWAYS**</ins>

Comparing the silhouette scores across all the runs, the base set has the highest score, and combining with the engineered features lowered the score slightly - which indicate that the engineered feature sets do not significantly improve clustering quality. This also suggests that the base fields are already sufficient to capture the customer segmentation for this dataset (`age`, `annual_income`, `purchase_amount`, `loyalty_score`, `annual_income`, `purchase_frequency`). 

The feature `region_grouped` did not seem to specifically provide further insight into further behaviour - for example, customers in a designated region did not fully dominate the highest spending clusters. It may be worth exploring further clustering by region due to potential differences in regional purchasing behaviours.

Throughout all except the Key Segment Flag runs, most of the clusters are made of a progression from young, low income, low spending customers to senior, high income, high spending customer segments, which are consistent with the base run. This does not mean the engineered features should be ignored for clustering with other customer datasets - they can be useful to provide further insight, or highly targeted segmenting based on the business needs. For example:
*   **Core Scores & Behavioural Ratios:** The clusters provide further information on customers that directly translate to business strategy for each segment. It helps stakeholders understand the risks and opportunities that come with each segment, and the actionable appropriate methods can be used.
*   **Percentiles & Key Segments and Flags:** The clusters formed from this approach provide clear indicators of who the top customers are, and automatically adjusts for different markets by using a relational comparison. Stakeholders using these features have a full understanding that most of their resources should be spent understanding this customer base further and to retain or grow them.
*   **Log Transformations:** While for this dataset it may not have shown significant results, this can be very useful for the analytics side for businesses trying to handle outliers. With much larger datasets, it is worth trying to cluster these to analyze further customer spending habits.
<br/><br/>

That being said, there are some caveats which may have led to these results:
*   **Dataset Size:** The dataset may be too small to show the benefits from feature engineering. 238 customers is quite a small sample size when considering B2C situations.
*   **Correlation Too High:** The synthetic data set shows a strong linear relationship between age and income across the entire dataset. This will not be true for real customer datasets, and the engineered features would likely provide more value and insight into the different customer clusters.
<br/><br/>

With regards to the business objectives, the clustering with the base features is the predictable and sufficient to identify customer segments when using this synthetic dataset. However, it is not conclusive if these features are sufficient when handling real production data which tends to be messier. It's likely using them would be a strong starting point, but will require further testing to confirm this conclusion. 

<br/><br/>


## Random Forest
*   **TBD**

## Neural Network 
*   **TBD**


## XG Boost 
*   **TBD**





## Conclusions, Findings and Insights
Our results from our different analyses all show that they are "too clean", which implies that the true performance is limited by our data:
   - **Synthetic Nature:** the data appears to be generated by a formula, and the high multicollinearity makes it hard to test it's true performance
   - **Lack of Volume:** the dataset only has a limited number of records, which make it harder to uncover more nuances in customer behaviour
   - **Lack of Randomness:** the cleanness of the dataset may not represent a real world scenario, for example an increase in age does not always result in an increase in spending
   - **Lack of Sample Region:** the lacking sample of the East region may make it difficult to predict changes and trends in spending behaviour for that region.

While we were able to create highly accurate predictions with this dataset, it should be tested in reality to confirm its performance and the models should be able to find a solution with an improved data source.


## Future Recommendations:
Given the time constraint for this project and the findings from testing, it is recommended to do the perform additional steps to further enhance our project:

**Methodology Refinement**
   - Test the model on larger and random dataset - ideally production customer data 
        - Confirm if model accuracy is consistent with updated dataset
   - Rerun other models to see if new insights are derived:
        - Regression - can be retested to observe important if features are not as multicollinear
        - Clustering - can identify new customer segments in spending behaviour 

**Project Deployment**
   - Monitor model performance for consistency with new datasets
   - Perform A/B testing against simple baselines to confirm what can improve model predictions
   - Confirm that the results are reliable and still relevant to the business objectives and can resolve client's problem

     

## Credits & Contributors
This project was developed through a collaborative effort as our Team Project for the Machine Learning Software Foundations Certificate. All team members contributed meaningfully to the outcome and are listed in alphabetical order below.

| Member Name              | GitHub Account                                                       | Reflection Video                                                                             |
|--------------------------|----------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| Antonio M. Lancuentra    | [AntonioMLancuentra](https://github.com/AntonioMLancuentra)          | [Link](paste_your_link_here)|
| Calvin Ho                | [c5ho](https://github.com/c5ho)                                      | [Link](paste_your_link_here)|
| Eliot Choy               | [elioc1341](https://github.com/elioc1341)                            | [Link](paste_your_link_here)|
| Henry Wong               | [eternal-loading-screen](https://github.com/eternal-loading-screen)  | [Link](paste_your_link_here)|
| Vinod Anbalagan          | [VinodAnbalagan](https://github.com/VinodAnbalagan)                  | [Link](paste_your_link_here)|
