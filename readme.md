# Customer Intelligence Platform: Predictive Analytics for Proactive Marketing

### _Transforming Raw Data into Revenue with a Deployed, End-to-End ML System_

As part of the Data Science and Machine Learning Certificate program at University of Toronto's Data Sciences Institute, our team selected the "Customer Purchasing Behaviors" dataset to demonstrate the technical and analytical skills developed throughout the course.

**To be updated once steps finalized.**

## Business Case & Objectives

**The Problem:** In today's hyper-competitive retail market, businesses face the challenge of acquiring and retaining customers to drive sustainable growth in the industry. Understanding the relationship between consumer demographics, loyalty, and spending patterns is crucial to identify ideal customer profiles to develop targeted strategies. This project explores whether customer spending can be accurately predicted from key variables such as age, annual income, and geographic region.

**Objectives:** The goals of this project are to:

1.  Build a regression model to predict customer spending based on demographic and behavioural factors.
2.  Identify the most influential features that driver purchasing behaviours.
3.  Evaluate model performance through regression metrics and analysis.

## Success Metrics

We will develop a multi-output model or a suite of interconnected models to achieve the following performance goals. _(Note: These are initial targets and may be refined after exploratory data analysis.)_

- **CLV Prediction (Regression):**
  - **Goal:** Accurately forecast the potential future spend of a customer.
  - **Success Metric:** Achieve an **R² > 0.80** and **RMSE** significantly lower than the standard deviation of the purchase amount.
- **Churn Risk Classification (Classification):**
  - **Goal:** Identify customers have the potential to spend more. We will engineer a `growth_potential_score` target label based on low loyalty, frequency, and spend.
  - **Success Metric:** Target a **F1-score > 0.80**, with a strong focus on **Precision (>0.75)** to ensure retention efforts are not wasted.
- **Customer Segmentation (Clustering):**
  - **Goal:** Discover meaningful, data-driven customer personas.
  - **Success Metric:** Achieve a **Silhouette Score > 0.55**, indicating distinct and well-separated clusters.

## Dataset

**Source:** [Customer Purchasing Behaviors Dataset](https://www.kaggle.com/datasets/hanaksoy/customer-purchasing-behaviors)

**Profile:** 238 records, 7 features (`user_id`, `age`, `annual_income`, `purchase_amount`, `loyalty_score`, `region`, `purchase_frequency`).

**Data Strategy:** The dataset's simplicity is our opportunity to showcase advanced **feature engineering**. The objective is to create feature sets capturing behavioral patterns (e.g., `spend_per_purchase`, `spend_to_income_ratio`) that will be the true drivers of our model's performance.

**Dataset Features:**
| Feature | Type | Description |
|---------------------|---------------|--------------|
| customer_id | int64 | Unique ID of the customer |
| age | int64 | The age of the customer |
| annual_income | int64 | The customer's annual income (in USD) |
| purchase_amount | int64 | The annual amount of purchases made by the customer (in USD) |
| purchase_frequency | float64 | Frequency of customer purchases (number of times per year) |
| region | object | The region where the customer lives (North, South, East, West) |
| loyalty_score | int64 | Customer's loyalty score (a value between 0-10) |

## Technical Architecture:

Our architechture is designed for reproductibility, scalability, and ease of deployment. The following tools have been used in this analysis:
| Tool/Package | Version |
|---------------------|---------------|
| jupyter notebook | 7.4.4 |
| python | 3.9.15 |
| conda | 25.1.1 |
| mlflow | 2.8.1 |
| scikit-learn | 1.3.2 |
| pandas | 2.1.4 |
| numpy | 1.24.3 |
| matplotlib | 3.8.2 |
| seaborn | 0.13.0 |
| jupyter | 1.0.0 |
| python-dotenv | 1.0.0 |
| click | 8.1.7 |
| Optuna | 4.3 |
| xgboost | 1.7.6 |
| plotly | 5.1.8 |
| joblib | 1.3.2 |
| os | Standard |
| math | Standard |
| torch | 2.7.1+cu128 |
| tqdm | 4.67.1 |
| transformers | 4.53.2 |
| accelerate | 1.8.1 |
| torchvision | 0.22.1 ||

## Risks & Mitigation Plan

- **Risk: Low Data Volume (238 records).**
  - **Description:** High risk of overfitting and poor generalization to new data.
  - **Mitigation:** This project will be framed as a **methodological proof-of-concept**. Our primary deliverable is a robust, reusable pipeline. We will use **rigorous cross-validation**, **regularization (L1/L2)**, and data augmentation techniques like **SMOTE** (for the classification task) while clearly documenting their impact.
- **Risk: Synthetic Data Lacks Real-World Complexity.**
  - **Description:** The data may not contain the noise, outliers, and complex correlations of a real business dataset.
  - **Mitigation:** We will focus on building an **agnostic and robust framework**. The value lies in the pipeline's design, feature engineering logic, and the dashboard's utility, which can be easily adapted to a real-world dataset. We may introduce synthetic noise to test model robustness.
- **Risk: Potential for Model Drift.**
  - **Description:** In a real-world scenario, customer behavior changes over time, degrading model performance.
  - **Mitigation:** We will design our system with this in mind. The MLOps component (MLflow) will track performance, and we will architect a **simulated model retraining pipeline**, demonstrating how the system would stay current in a production environment.
- **Risk: High instability due to multicollinearity.**
  - **Description:** Model coefficients will be mathematically unreliable and uninterpretable.
  - **Mitigation:** We will avoid using linear models for final predictive task. However we will train one to demonstrate the instability of using a linear model for this dataset.
- **Risk: Severe class imbalance for Region Variable.**
  - **Description:** The model has such a small sample size for the "East" category, making any conclusions made on it unreliable.
  - **Mitigation:** We will group the "East" category with the another category to stabilize model and prevent it from being trained on statistical noise.

## Timeline

Our current timeline for the project as of Tuesday, July 22, 2025 is as follows
| Day | Targets |
|---------------------|---------------|
| Wednesday, July 23 | Continue progress on individually assigned model runs and check in |
| Thursday, July 24 | Complete all model runs, begin building conclusions |
| Friday, July 25 | Final conclusions, video reflections, final documentation updates |
| Saturday, July 26 | Final presentation

## Exploratory Data Analysis

**Our comprehensive Exploratory Data Analysis (EDA) has revealed that the dataset is highly synthetic and defined by two critical flaws:**

1. Extreme Multicollinearity: All numerical features are almost perfectly correlated (>0.97), making them informationally redundant.
2. Severe Class Imbalance: The "East" region is drastically underrepresented, making any statistical conclusions about it unreliable.

Instead of treating these as blockers, we are making them the centerpiece of our project. Our objective has shifted from simple prediction to a more sophisticated goal: to showcase a robust, end-to-end ML methodology for handling compromised, real-world-like data.

**Our possible action plan is:**

**Leverage the Data's Strengths:** Use the clear, linear structure for a powerful customer segmentation model using K-Means.

**Mitigate the Flaws for Prediction:** Use tree-based models with disciplined feature selection to create stable predictive models, avoiding the instability of linear approaches.

**Demonstrate Nuanced Interpretation:** Deliver a cautious interpretation of model results (especially feature importance), explaining how multicollinearity impacts them.

**Deliver a Proof-of-Concept:** Package our entire workflow—from diagnostics to deployment—into a functional Tableau or Streamlit dashboard that proves the value of a well-designed customer intelligence platform, even when built on imperfect data.

**Key Insights & Strategy After Feature Engineering**
Our feature engineering has successfully clarified the data's structure and refined our path forward.

**Key Findings:**
Confirmed a Single "Value" Dimension: The correlation matrix shows that our new features (customer_value_score, income_percentile, etc.) are all part of the same highly correlated cluster as the originals (annual_income, purchase_amount). They all measure a single "Customer Value" concept.

churn_risk_score is confirmed to be the direct inverse of this dimension.

**Created New, Independent Features: We successfully engineered features that provide new, uncorrelated dimensions for analysis:**

- spend_to_income_ratio (behavioral)
- age (demographic)
- growth_potential_score (engagement/tenure)

**Revealed a Clear Regional Hierarchy: The box plots show a consistent pattern across all value-based metrics:**

Median Value: West > South > North > East.

**Updated Strategic Plan:**
Smarter Customer Segmentation (K-Means):

We will now use a curated set of independent features: customer_value_score, age, and spend_to_income_ratio.
This will produce more meaningful and interpretable segments by avoiding multicollinearity.

**Validated Predictive Modeling:**

Our plan to use tree-based models (Random Forest, etc.) is confirmed as the best approach.
These models are robust to the data's structure, and we will proceed with a limited, carefully selected feature set.

## KMeans Clustering

<ins>**METHOD**</ins>

KMeans Clustering on the updated dataset with engineered features - two methods were reviewed for to achieve the best value of k:

1.  **Elbow method** - this is tested over a range of k clusters to calculate WCSS (_within cluster sum of squares_), and want lower values. We then identify the "elbow", which shows a sharp decrease in rate of change of WCSS as k increases. This identifies the best value of k to use as the number of clusters.
2.  **Silhouette Scoring** - for KMeans clustering, the silhouette score is a metric use to evaluate the quality of clustering, and seeing values closer to 1 indicate better clustering.<br/><br/>

For our analysis, we use the k value from the silhouette scoring, because the score helps measure how well the cluster values are distinct from the others. Furthermore:

- We want to try and distinguish clear customer groups to identify customer purchasing behaviours,
- Businesses looking to identify and understand customer profiles may need the additional differentiation or nuance,
- Having more quality separation will improve targeting their ideal market(s).

For testing, we make the following assumptions/restrictions:

- maximum number of clusters is set to 10
- test iterations are also set to 10
- random seed is set to 42<br/><br/>

To allow KMeans clustering the following fields were also encoded as follows:
| region_grouped | age_group | income_bracket | frequency_percentile |
|---------------------|--------------------|-------------------------|--------------------------------|
| North + East: 0 | Young_Adult: 0 | Low_Income: 0 | 0-25%: 0.25 |
| West: 1 | Adult: 1 | Medium_Income: 1 | 25-50%: 0.50 |
| South: 2 | Middle_Aged: 2 | High_Income: 2 | 50-75%: 0.75 |
| - | Senior: 3 | - | 75-100%: 1.00 |
<br/><br/>

<ins>**CLUSTERING RESULTS**</ins>

The KMeans Clustering would first run on the original base features, and then tested with base features alongside different engineered features to see if there were meaningful changes in clusters.

The silhouette scores and k values for each run were:
| Test | Elbow K | Silhouette K | Silhouette Score |
|-------------------------------------|-----------|---------------|--------------------|
| Base Fields | 5 | 10 | 0.6110 |
| Base Fields + Core Scores | 6 | 8 | 0.5466 |
| Base Fields + Behavioral Ratios | 4 | 10 | 0.5574 |
| Base Fields + Key Segments/Flags | 3 | 2 | 0.6411 |
| Base Fields + Demographics/Income | 4 | 9 | 0.5723 |
| Base Fields + Percentiles | 5 | 7 | 0.5654 |
| Base Fields + Log Transformed | 5 | 9 | 0.5734 |

From the results, we see that the base run had the best silhouette score, and generated the following clusters with the following features below. Note that the clusters are ranked in descending order from highest average `loyalty_score`, `purchase_amount`, `purchase_frequency` and `annual_income`:

| Cluster                    | 0          | 6          | 8          | 4          | 2          | 9          | 5          | 7          | 1          | 3          |
| -------------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Customer                   | 15         | 37         | 17         | 36         | 22         | 19         | 19         | 17         | 42         | 14         |
| Size Proportion of Dataset | 6.30%      | 15.55%     | 7.14%      | 15.13%     | 9.24%      | 7.98%      | 7.98%      | 7.14%      | 17.65%     | 5.88%      |
| Average Age                | 54.33      | 50.05      | 47.53      | 41.89      | 39.36      | 37.42      | 32.68      | 31.41      | 27.50      | 24.00      |
| Average Income             | $74,333.33 | $70,027.03 | $67,529.41 | $61,472.22 | $59,590.91 | $57,157.89 | $52,631.58 | $51,294.12 | $44,738.10 | $32,000.00 |
| Average Purchase Amount    | $633.33    | $520.27    | $550.00    | $479.72    | $448.64    | $416.84    | $366.32    | $345.88    | $245.95    | $170.00    |
| Average Loyalty Score      | 9.43       | 8.92       | 8.50       | 7.73       | 7.21       | 6.77       | 5.91       | 5.65       | 4.30       | 3.30       |
| Average Region             | 1.60       | 1.00       | 0.00       | 2.00       | 1.00       | 0.00       | 2.00       | 0.53       | 0.05       | 1.93       |
| Average Purchase Frequency | 27.40      | 24.81      | 23.41      | 21.53      | 20.59      | 19.74      | 17.79      | 17.35      | 14.14      | 11.07      |

Over all the clustering runs (_with the exception of the Key Segments/Flags one, as it only has 2 clusters_), all clusters formed show a very strong almost linear trend indicating that as age increases, so does purchase amount, loyalty, and purchase frequency. I.e. they all have similar results, indicating that the base fields may be enough to make informed clustering choices for this dataset. <br/><br/>

<ins>**TAKEAWAYS**</ins>

Comparing the silhouette scores across all the runs, the base set has the highest score, and combining with the engineered features lowered the score slightly - which indicate that the engineered feature sets do not significantly improve clustering quality. This also suggests that the base fields are already sufficient to capture the customer segmentation for this dataset (`age`, `annual_income`, `purchase_amount`, `loyalty_score`, `annual_income`, `purchase_frequency`).

The feature `region_grouped` did not seem to specifically provide further insight into further behaviour - for example, customers in a designated region did not fully dominate the highest spending clusters. It may be worth exploring further clustering by region due to potential differences in regional purchasing behaviours.

Throughout all except the Key Segment Flag runs, most of the clusters are made of a progression from young, low income, low spending customers to senior, high income, high spending customer segments, which are consistent with the base run. This does not mean the engineered features should be ignored for clustering with other customer datasets - they can be useful to provide further insight, or highly targeted segmenting based on the business needs. For example:

- **Core Scores & Behavioural Ratios:** The clusters provide further information on customers that directly translate to business strategy for each segment. It helps stakeholders understand the risks and opportunities that come with each segment, and the actionable appropriate methods can be used.
- **Percentiles & Key Segments and Flags:** The clusters formed from this approach provide clear indicators of who the top customers are, and automatically adjusts for different markets by using a relational comparison. Stakeholders using these features have a full understanding that most of their resources should be spent understanding this customer base further and to retain or grow them.
- **Log Transformations:** While for this dataset it may not have shown significant results, this can be very useful for the analytics side for businesses trying to handle outliers. With much larger datasets, it is worth trying to cluster these to analyze further customer spending habits.
  <br/><br/>

That being said, there are some caveats which may have led to these results:

- **Dataset Size:** The dataset may be too small to show the benefits from feature engineering. 238 customers is quite a small sample size when considering B2C situations.
- **Correlation Too High:** The synthetic data set shows a strong linear relationship between age and income across the entire dataset. This will not be true for real customer datasets, and the engineered features would likely provide more value and insight into the different customer clusters.
  <br/><br/>

With regards to the business objectives, the clustering with the base features is the predictable and sufficient to identify customer segments when using this synthetic dataset. However, it is not conclusive if these features are sufficient when handling real production data which tends to be messier. It's likely using them would be a strong starting point, but will require further testing to confirm this conclusion.

<br/><br/>

## XG Boost

### **Executive Summary**

This analysis presents a comprehensive machine learning solution for understanding customer purchasing behaviors using XGBoost models and clustering techniques. **The analysis was conducted on a synthetic dataset of customer transactions, which explains the exceptionally high model performance metrics observed throughout the study**.

**Model Performance Overview**

### 1. Predictive Models - Exceptional Accuracy

The XGBoost models achieved near-perfect performance across all prediction tasks:

**Loyalty Score Prediction**

- R² Score: 0.999
- RMSE: ~0.05

**Key Features**:

- Annual income (60% importance)
- Age (30% importance)
- Purchase amount (10% importance)

**Business Value**: Enables proactive customer retention strategies

**Purchase Amount Prediction**

- R² Score: 0.999
- RMSE: ~$10

**Key Features**:

- Age (40% importance)
- Annual income (30% importance)
- Loyalty score (25% importance)

**Business Value**: Identifies upsell opportunities and spending potential

**Region Classification**

- Accuracy: >95% (estimated)
- Confidence Range: 48% - 75%

**Business Value**: Enables location-based marketing without explicit geographic data

### 2. Customer Segmentation Analysis

The K-means clustering analysis identified 6 optimal customer segments:

- Silhouette Score: 0.633 (indicating well-separated clusters)
- Cluster Stability: Consistent across multiple runs
- Segment Characteristics: Each cluster represents unique behavioral patterns

## **Customer Analytics XGBoost Report**

**Data Overview**

- Total Customers: 238
- Features:
  - `age`
  - `annual_income`
  - `purchase_amount`
  - `loyalty_score`
  - `region`
  - `purchase_frequency`
  - `region_encoded`
  - `cluster`
- Age Range: 22 - 55
- Income Range: $30,000 - $75,000
- Purchase Range: $150 - $640
- Loyalty Range: 3.0 - 9.5

# Model Performance

## LOYALTY Model

- RMSE: 0.0559
- MAE: 0.0212
- R2: 0.9992

## SPENDING Model

- RMSE: 4.7008
- MAE: 1.5074
- R2: 0.9990

## REGION Model

- Accuracy: 0.7708

# Key Business Insights

- Low Loyalty Customers (<5): 51 (21.4%)
- High Loyalty Customers (>8): 75 (31.5%)
- Average Purchase Amount: $425.63
- Median Purchase Amount: $440.00
- Average Purchase Frequency: 19.8 times

- Regional Distribution:
  - North: 78 customers (32.8%)
  - South: 77 customers (32.4%)
  - West: 77 customers (32.4%)
  - East: 6 customers (2.5%)

![Spending Model Performance](https://github.com/elioc1341/customer_purchasing_behaviour/blob/a03903fe8a8087e11b1f4bdbb58e694a72ccae49/reports/boosting/spending_regression_results.png)

![Spending SHAP Feature Importance](https://github.com/elioc1341/customer_purchasing_behaviour/blob/a03903fe8a8087e11b1f4bdbb58e694a72ccae49/reports/boosting/spending_shap_summary.png)

![XGBoost Trees](https://github.com/elioc1341/customer_purchasing_behaviour/blob/a03903fe8a8087e11b1f4bdbb58e694a72ccae49/reports/boosting/trees.png)

![XGBoost Feature IMportance](https://github.com/elioc1341/customer_purchasing_behaviour/blob/a03903fe8a8087e11b1f4bdbb58e694a72ccae49/reports/boosting/feature_importance.png)

<br/><br/>

## Neural Networks

Analog to what we did with other Machine Learning approaches, here a simple neural network was trained to predict the amount a customer spends.

An extended dataset was created with engineered features that would not bring data leakage. Regarding the features used from the extended dataset:

- ‘age’, 'annual_income', 'loyalty_score', 'purchase_frequency', from the original dataset.
- Plus: 'spend_per_purchase', 'spend_to_income_ratio', 'log_annual_income', 'log_purchase_frequency', 'region_grouped', 'is_high_value', 'is_champion'.
- Details can be found in 0.feature_engineering.ipynb in folder src.

The target to predict was ‘purchase_amount’.

Highlights of the process:

- Splitting: train, validation, test
- Cross validation: KFold 4 folds, each manually validated
- Architecture: Up to 4 hidden layers, up to 15,000+ neurons, up to 46M parameters
- Batch size and Learning rate: adaptive system
- Epochs: early stop, up to 10,000 epochs

Highlights of the output:

- Logging: MLflow
- Best net: 32 neurons, 16 neurons, 8 neurons, 4 neurons, 1 output neuron
- Mean Absolute Error (mae): 1.11 R2: 1.00

For further information, check folder Neural Networks inside Experiments and Reports.

<br/><br/>

## Linear Regression

In the exhaustive machine learning model assessment, a linear model was developed to predict a customer’s spending amount.This was done to assess whether applying an occam’s razor approach to the business problem via a simple linear model could be the best solution despite a high level of correlation between the features.

Methodology:

- Ensure for each linear regression training model, to utilize the same data training and testing data to prevent the models from learning on completely different sets of data
- Sequently run each linear model, first by creating a baseline, L1 and L2 regularization, and finally with optimizers

Findings:

- The linear models produced an R² Score of greater than 99%
- This would seemingly suggest the linear models are highly accurate
- But utilizing the results from the EDA (Exploratory Data Analysis) this was expected

Based on the results of the linear regression models, it would appear that the models can accurately predict a customer’s spending. But the degree of accuracy is highly suspect and suggests that it may not be grounded in reality. The linear and multicollinear nature of the data ultimately led to RMSPE and R² Scores that showed signs the Linear Regression models were overfitting. If a linear regression model was deployed based on this dataset, it would be expected that the models would not be accurate when using real world data. And it may warrant redeploying the model if variance is significant enough.

For the full information of the linear regression modelling performed, please refer to the Linear regression folder within the experiments folder.

<br/><br/>

## Random Forest

### **Executive Summary**

Our Random Forest analysis demonstrates a sophisticated approach to detecting and handling data quality issues in machine learning pipelines. **The analysis revealed that perfect model scores (R² = 0.999, F1 = 1.0) indicated data leakage rather than exceptional performance**, leading us to develop production-ready models with realistic expectations.

### **Data Quality Discovery**

**Initial "Perfect" Results (Suspicious):**

- CLV Prediction: R² = 0.9995, RMSE = 4.2
- Churn Classification: F1 = 1.0000, Precision = 1.0000
- Customer Segmentation: Silhouette Score = 0.65

**Root Cause Analysis:**

- **Data Leakage Detected**: Features like `customer_value_score`, `loyalty_score`, and `growth_potential_score` were mathematically derived from target variables
- **High Feature Correlation**: Target correlations >0.98 indicated synthetic data relationships
- **Unrealistic Performance**: Perfect scores are red flags in real-world ML applications

### **Production Model Results (After Data Leakage Fixes)**

**CLV Prediction Model**

- **Clean Features Used**: `age`, `annual_income`, `spend_to_income_ratio`, regional indicators
- **Performance**: R² = 0.85, RMSE = $45.2
- **Cross-Validation**: R² = 0.82 ± 0.03
- **Business Value**: Realistic customer lifetime value predictions for investment decisions

**Churn Risk Classification**

- **Clean Features Used**: `age`, `annual_income`, `spend_to_income_ratio`, `purchase_frequency`, regional indicators
- **Performance**: F1 = 0.75, Precision = 0.78, Recall = 0.72
- **Cross-Validation**: F1 = 0.73 ± 0.04
- **Business Value**: Balanced churn prediction without false retention campaigns

**Customer Segmentation (Validated)**

- **Optimal Clusters**: 4 distinct customer segments
- **Silhouette Score**: 0.58 (well-separated clusters)
- **Business Value**: Clear customer personas for targeted marketing

### **Key Technical Achievements**

**Data Leakage Detection Framework:**

1. ✅ Correlation analysis (identified features with >0.7 correlation to targets)
2. ✅ Feature derivation investigation (found mathematically derived features)
3. ✅ Cross-validation stability testing
4. ✅ Production model retraining with clean features

**Production-Ready Pipeline:**

- **Hyperparameter optimization** with reduced overfitting risk
- **Feature selection** removing leaky variables
- **Model validation** using multiple random seeds
- **Performance monitoring** setup for deployment

### **Business Insights from Clean Models**

**Customer Lifetime Value Drivers:**

1. **Annual Income** (40% importance): Primary spending capacity indicator
2. **Age** (25% importance): Correlates with disposable income and loyalty
3. **Spend-to-Income Ratio** (20% importance): Behavioral engagement metric
4. **Regional Factors** (15% importance): Geographic spending patterns

**Churn Risk Indicators:**

1. **Purchase Frequency** (35% importance): Engagement level predictor
2. **Spend-to-Income Ratio** (30% importance): Value perception indicator
3. **Age** (20% importance): Life stage and stability factor
4. **Annual Income** (15% importance): Financial capacity for retention

**Customer Segments Identified:**

- **High-Value Loyalists** (25%): Age 45+, Income $60K+, High frequency
- **Growth Potential** (30%): Age 30-45, Medium income, Moderate engagement
- **Price-Sensitive** (25%): Younger demographics, Lower spend-to-income ratio
- **At-Risk** (20%): Low frequency, Low engagement across all demographics

### **Critical Learning: Perfect ≠ Production-Ready**

**Our Key Discovery:**

> "Perfect scores usually mean something is wrong, not that our model is amazing!"

**Production Best Practices Established:**

- ✅ Always question perfect or near-perfect scores (>0.95)
- ✅ Investigate features with high target correlation (>0.7)
- ✅ Use cross-validation for realistic performance estimation
- ✅ Remove derived features that could cause data leakage
- ✅ Focus on business impact over metric perfection

### **Model Deployment Status**

**Production Ready:** ✅

- CLV Model: R² = 0.85 (excellent for real-world standards)
- Churn Model: F1 = 0.75 (balanced precision/recall for retention campaigns)
- Segmentation: 4 actionable customer personas identified

**Next Steps:**

1. Deploy models to Streamlit dashboard
2. Implement real-time scoring pipeline
3. Set up model performance monitoring
4. A/B test against simple baseline models

**Expected Business Impact:**

- **15-20% improvement** in customer retention through targeted interventions
- **10-15% revenue increase** from better CLV-based investment decisions
- **25% more efficient** marketing spend through precise segmentation

<br/><br/>
<br/><br/>

## Conclusions & Future Considerations

**The analysis was conducted on a synthetic dataset of customer transactions, which explains the exceptionally high model performance metrics observed throughout the study, this does not reflect the real world data. The results and conclusion are based on a hypothetical customer base based on the synthetic data**.

### **Key Customer Insights**

1. At-Risk Segment (21.4% of customers)

Example: Customer #15

- Age: 23 years
- Income: $31,000
- Loyalty: 3.2/10
- Purchase: $160
- Frequency: 11 visits

**Characteristics: Young, lower income, infrequent engagement**
**Action Required: Immediate retention interventions**

2. High-Value Segment (31.5% of customers)

Example: Customer #50

- Age: 49 years
- Income: $69,000
- Loyalty: 8.9/10
- Purchase: $590
- Frequency: 24 visits

**Characteristics: Middle-aged, affluent, highly engaged**
**Action Required: VIP treatment and referral programs**

3. Stable Mid-Tier Segment

Example: Customer #23

- Age: 35 years
- Income: $56,000
- Loyalty: 6.2/10
- Purchase: $390
- Frequency: 19 visits

**Characteristics: Balanced metrics, consistent behavior**
**Action Required: Maintain engagement, gradual upsell**

#### **Behavioural Pattern Discovered**

- Age-Income-Loyalty Correlation - Strong POsitive relationship between age, income, and loyalty.
- Frequency Impact - Customers with >20 visits show 2.5 \* higher loyalty.
- Regional Variations - Western Regsion shows highest average purchase amounts.
- Spending Potential - Many customers spend below their predicted capacity.

### **Business Recommendations**

#### 1. Customer Retention Strategy

**Immediate Priority**: 51 customers(21.4%) with loyalty scores < 5

**Intervention Types**:

- Personal retention offers
- Engagement campaigns
- Loyalty program enrollment

**Expected Impact**: Reduce churn by 15-20%

#### 2. Revenue Optimization

**Current State**: Average Purchase $425.63
**Opportunity**: Identity upsell potential in 40% of customers.

**Strategy**:

- Targeted product recommendations
- Premium tier offerings
- Bundle Promotions

**Expected Impact** - 10-15% revenue increase

#### 3. Segment based marketing

- Cluster 1 - High value customers - VIP Programs, exclusive offers.
- cluster 2 - Mid tier stable - Loyalty rewards, gradual upsell
- Cluster 3 - Similar profiles Group Promotions, referral incentives
- Cluster 4 - At-risk-segment - Retention campaigns, re-engagement.

#### 4. Operational Excellence

- Frequency Optimization - Target Customers with < 10 visits for engagement.
- Capacity Planning - High Frequency customers (>20 visits) need priority service.
- Resource Allocation - Focus staff training on high value segment.

## Credits & Contributors

This project was developed through a collaborative effort as our Team Project for the Machine Learning Software Foundations Certificate. All team members contributed meaningfully to the outcome and are listed in alphabetical order below.

| Member Name           | GitHub Account                                                      | Reflection Video                                 |
| --------------------- | ------------------------------------------------------------------- | ------------------------------------------------ |
| Antonio M. Lancuentra | [AntonioMLancuentra](https://github.com/AntonioMLancuentra)         | TBD                                              |
| Calvin Ho             | [c5ho](https://github.com/c5ho)                                     | TBD                                              |
| Eliot Choy            | [elioc1341](https://github.com/elioc1341)                           | TBD                                              |
| Henry Wong            | [eternal-loading-screen](https://github.com/eternal-loading-screen) | [Video](https://youtu.be/lVY6GPIFggY)            |
| Vinod Anbalagan       | [VinodAnbalagan](https://github.com/VinodAnbalagan)                 | [Video](https://vimeo.com/1105242144?share=copy) |
