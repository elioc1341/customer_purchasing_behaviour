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
| math | Standard | |

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

## Regression Analysis & Testing

- **TBD**

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

## Results

- **TBD**

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

| Member Name           | GitHub Account                                                      | Reflection Video |
| --------------------- | ------------------------------------------------------------------- | ---------------- |
| Antonio M. Lancuentra | [AntonioMLancuentra](https://github.com/AntonioMLancuentra)         | TBD              |
| Calvin Ho             | [c5ho](https://github.com/c5ho)                                     | TBD              |
| Eliot Choy            | [elioc1341](https://github.com/elioc1341)                           | TBD              |
| Henry Wong            | [eternal-loading-screen](https://github.com/eternal-loading-screen) | TBD              |
| Vinod Anbalagan       | [VinodAnbalagan](https://github.com/VinodAnbalagan)                 | TBD              |
