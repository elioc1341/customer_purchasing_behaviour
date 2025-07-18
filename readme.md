# Customer Intelligence Platform: Predictive Analytics for Proactive Marketing
### _Transforming Raw Data into Revenue with a Deployed, End-to-End ML System_
## 1. Project Vision & Business Case
**The Problem:** In today's hyper-competitive retail market, customer acquisition costs are **5 to 25 times higher** than customer retention. Yet, many businesses use one-size-fits-all marketing strategies that are inefficient and fail to engage customers effectively. The key to sustainable growth lies in understanding and anticipating customer needs.
**Our Solution:** This project builds a deployable **Customer Intelligence Platform** that serves as a central engine for data-driven decision-making. We will develop a unified machine learning model that analyzes customer behavior to:
1.  **Segment Customers** into actionable groups (e.g., *Champions, Loyal, At-Risk*).
2.  **Predict Customer Lifetime Value (CLV)** to identify high-potential individuals.
3.  **Generate a Churn Risk Score** to enable proactive retention campaigns.
**Primary Objective:** To deliver a live, interactive web application that provides real-time, actionable insights, empowering marketing teams to personalize campaigns, reduce churn, and maximize return on investment (ROI).
---
## 2. Goals & Success Metrics
### Modeling Objectives
We will develop a multi-output model or a suite of interconnected models to achieve the following performance goals. *(Note: These are initial targets and may be refined after exploratory data analysis.)*
*   **CLV Prediction (Regression):**
    *   **Goal:** Accurately forecast the potential future spend of a customer.
    *   **Success Metric:** Achieve an **R² > 0.80** and **RMSE** significantly lower than the standard deviation of the purchase amount.
*   **Churn Risk Classification (Classification):**
    *   **Goal:** Precisely identify customers who are likely to churn. We will engineer a `is_at_risk` target label based on low loyalty, frequency, and spend.
    *   **Success Metric:** Target a **F1-score > 0.80**, with a strong focus on **Precision (>0.75)** to ensure retention efforts are not wasted.
*   **Customer Segmentation (Clustering):**
    *   **Goal:** Discover meaningful, data-driven customer personas.
    *   **Success Metric:** Achieve a **Silhouette Score > 0.55**, indicating distinct and well-separated clusters.
### Business Deliverables
*   **An Interactive Streamlit Dashboard:** A user-friendly interface for non-technical stakeholders to visualize segments, query individual customer risk/value scores, and understand model predictions (via SHAP).
*   **An Automated Reporting Pipeline:** A simulated report generation system that summarizes key business KPIs (e.g., % of customers at risk, average CLV of new customers) for marketing teams.
---
## 3. The Dataset
*   **Source:** Customer Purchasing Behaviors Dataset ([Link to Kaggle/Source])
*   **Profile:** 238 instances, 7 initial features (`user_id`, `age`, `annual_income`, `purchase_amount`, `loyalty_score`, `region`, `purchase_frequency`).
*   **Data Strategy:** The dataset's simplicity is our opportunity to showcase advanced **feature engineering**. We will create a rich feature set capturing behavioral patterns (e.g., `spend_per_purchase`, `income_to_spend_ratio`) that will be the true drivers of our model's performance.
---
## 4. Technical Architecture
Our architecture is designed for reproducibility, scalability, and ease of deployment.
*   **Core ML & Data Processing:** Python, Pandas, NumPy, Scikit-learn, XGBoost / LightGBM.
*   **Advanced Modeling:** TensorFlow or PyTorch will be considered for exploring complex, non-linear patterns if initial models hit a performance plateau.
*   **MLOps:** **MLflow** for robust experiment tracking, model registry, and versioning to ensure full reproducibility.
*   **Feature Engineering:** A modular and automated pipeline for generating and testing new features.
*   **Deployment & Visualization:** **Streamlit** for the interactive front-end, **Plotly** and **Seaborn** for dynamic visualizations, and **Docker** to containerize the entire application for one-command deployment.
---
## 5. Risks & Mitigation Plan
*   **Risk: Low Data Volume (238 records).**
    *   **Description:** High risk of overfitting and poor generalization to new data.
    *   **Mitigation:** This project will be framed as a **methodological proof-of-concept**. Our primary deliverable is a robust, reusable pipeline. We will use **rigorous cross-validation**, **regularization (L1/L2)**, and data augmentation techniques like **SMOTE** (for the classification task) while clearly documenting their impact.
*   **Risk: Synthetic Data Lacks Real-World Complexity.**
    *   **Description:** The data may not contain the noise, outliers, and complex correlations of a real business dataset.
    *   **Mitigation:** We will focus on building an **agnostic and robust framework**. The value lies in the pipeline's design, feature engineering logic, and the dashboard's utility, which can be easily adapted to a real-world dataset. We may introduce synthetic noise to test model robustness.
*   **Risk: Potential for Model Drift.**
    *   **Description:** In a real-world scenario, customer behavior changes over time, degrading model performance.
    *   **Mitigation:** We will design our system with this in mind. The MLOps component (MLflow) will track performance, and we will architect a **simulated model retraining pipeline**, demonstrating how the system would stay current in a production environment.
---
## 6. Credits & Contributors
This project is a collaborative effort for our Machine Learning university project.
*   [Antonio M. Lancuentra] 
*   [Vinod Anbalagan]
*   [Eliot Choy]
*   [Henry Wong]
*   [Calvin Ho]