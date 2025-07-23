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
| Feature             | Type          | Description  |
|---------------------|---------------|--------------|
| customer_id         | int64         | Unique ID of the customer      |
| age                 | int64         | The age of the customer    |
| annual_income       | int64         | The customer's annual income (in USD)     |
| purchase_amount     | int64         | The annual amount of purchases made by the customer (in USD)      |
| purchase_frequency  | float64       | Frequency of customer purchases (number of times per year)    |
| region              | object        | The region where the customer lives (North, South, East, West)     |
| loyalty_score       | int64         | Customer's loyalty score (a value between 0-10)      |


## Technical Architecture:
Our architechture is designed for reproductibility, scalability, and ease of deployment. The following tools have been used in this analysis:
| Tool/Package             | Version          | 
|---------------------|---------------|
| jupyter notebook         | 7.4.4         | 
| python         | 3.9.15         | 
| conda         | 25.1.1         | 
| mlflow         | 2.8.1         | 
| scikit-learn                 | 1.3.2         | 
| pandas       | 2.1.4         | 
| numpy     | 1.24.3         | 
| matplotlib  | 3.8.2       | 
| seaborn              | 0.13.0        | 
| jupyter       | 1.0.0         | 
| python-dotenv       | 1.0.0         | 
| click       | 8.1.7         | 
| os       | Standard         | 
| math       | Standard         |  | 


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
| Day             | Targets          | 
|---------------------|---------------|
| Wednesday, July 23         | Continue progress on individually assigned model runs and check in         | 
| Thursday, July 24                 | Complete all model runs, begin building conclusions         | 
| Friday, July 25       | Final conclusions, video reflections, final documentation updates         | 
| Saturday, July 26     | Final presentation        


## Exploratory Data Analysis
*   **TBD**


## Regression Analysis & Testing
*   **TBD**


## Results
*   **TBD**


## Conclusions & Future Considerations
**Key Takeaways:** **TBD**

**Future Considerations:** **TBD**


## Credits & Contributors
This project was developed through a collaborative effort as our Team Project for the Machine Learning Software Foundations Certificate. All team members contributed meaningfully to the outcome and are listed in alphabetical order below.

| Member Name              | GitHub Account   | Reflection Video |
|--------------------------|------------------|----------------- |
| Antonio M. Lancuentra    | [AntonioMLancuentra](https://github.com/AntonioMLancuentra) | TBD |
| Calvin Ho                | [c5ho](https://github.com/c5ho) | TBD |
| Eliot Choy               | [elioc1341](https://github.com/elioc1341) | TBD |
| Henry Wong               | [eternal-loading-screen](https://github.com/eternal-loading-screen) | TBD |
| Vinod Anbalagan          | [VinodAnbalagan](https://github.com/VinodAnbalagan) | TBD |
