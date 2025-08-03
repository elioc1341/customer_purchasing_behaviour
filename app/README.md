# Customer Intelligence Platform - Streamlit App

A comprehensive web application for customer analytics using machine learning models for Customer Lifetime Value prediction, churn risk assessment, and customer segmentation.

## 🚀 Quick Start

### From the app directory:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py

# Or use the deployment script
python deploy.py
```

The app will be available at: `http://localhost:8501`

## 📁 Directory Structure

```
app/
├── streamlit_app.py       # Main Streamlit application
├── utils.py              # Utility functions and helper classes
├── requirements.txt      # Python dependencies
├── deploy.py            # Deployment script
├── README.md            # This file
├── .streamlit/          # Streamlit configuration
│   └── config.toml      # Theme and server settings
├── pages/               # Additional pages (future)
└── assets/              # Static assets (future)
```

## 🎯 Features

- **Individual Customer Analysis**: Real-time predictions for single customers
- **Batch Analysis**: Analyze multiple customers from your dataset
- **Interactive Dashboard**: Visual insights and recommendations
- **Model Information**: Details about the ML models used

### Predictions Available:
1. **Customer Lifetime Value (CLV)** - Predict total customer value
2. **Churn Risk Assessment** - Identify customers likely to churn
3. **Customer Segmentation** - Group customers by behavior patterns

## 📋 Prerequisites

- Python 3.8 or higher
- Trained ML models in the `models/` directory
- Customer dataset in `data/processed/`

## 🛠️ Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Model Files

Ensure you have the following model files:
```
models/
├── random_forest/
│   └── models/
│       ├── clv_random_forest_model.pkl
│       ├── churn_random_forest_model.pkl
│       ├── segmentation_kmeans_model.pkl
│       └── segmentation_scaler.pkl
└── boosting/ (optional)
    ├── loyalty_score_model.joblib
    ├── purchase_amount_model.joblib
    └── customer_clustering_model.joblib
```

### 3. Verify Data Files

Ensure you have:
```
data/processed/df_eng_customer_purchasing_features.csv
```

## 🚀 Running the Application

### Local Development

```bash
streamlit run streamlit_app.py
```

The app will be available at: `http://localhost:8501`

### Production Deployment

#### Option 1: Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy!

#### Option 2: Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t customer-intelligence .
docker run -p 8501:8501 customer-intelligence
```

#### Option 3: Cloud Platforms

**Heroku:**
```bash
# Create Procfile
echo "web: streamlit run streamlit_app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

**AWS EC2:**
```bash
# On EC2 instance
sudo apt update
sudo apt install python3-pip
pip3 install -r requirements.txt
streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
```

## 📱 Usage Guide

### Individual Customer Prediction

1. **Enter Customer Information**:
   - Demographics (age, income, region)
   - Purchase behavior (amount, frequency, loyalty)

2. **Generate Predictions**:
   - Click "Generate Predictions" button
   - View CLV, churn risk, and segment predictions

3. **Review Recommendations**:
   - Get actionable insights for customer management
   - Understand prediction confidence levels

### Batch Analysis

1. Navigate to "Batch Analysis" page
2. View dataset overview and distributions
3. Analyze customer segments and patterns

## 🎯 Key Metrics Explained

- **Customer Lifetime Value (CLV)**: Predicted total revenue from customer
- **Churn Probability**: Likelihood of customer leaving (0-100%)
- **Customer Segment**: Behavioral group classification
- **Customer Score**: Composite metric (CLV × retention probability)

## 🔧 Configuration

### Streamlit Settings

Edit `.streamlit/config.toml` to customize:
- Theme colors
- Server settings
- Performance options

### Model Configuration

Models can be updated by replacing files in the `models/` directory. The app will automatically load new models on restart.

## 📊 Model Performance

| Model | Algorithm | Performance |
|-------|-----------|-------------|
| CLV Prediction | Random Forest | R² ≈ 0.85 |
| Churn Classification | Random Forest | F1 ≈ 0.82 |
| Customer Segmentation | K-Means | Silhouette ≈ 0.65 |

## 🔒 Security Considerations

- **Data Privacy**: Customer data is processed locally
- **Model Security**: Keep model files secure
- **Access Control**: Add authentication for production use

## 🐛 Troubleshooting

### Common Issues:

1. **Models not loading**:
   - Check file paths in `models/` directory
   - Verify model files are not corrupted

2. **Data not found**:
   - Ensure CSV file exists in `data/processed/`
   - Check file permissions

3. **Import errors**:
   - Run `pip install -r requirements.txt`
   - Check Python version compatibility

4. **Performance issues**:
   - Increase memory allocation
   - Use caching for large datasets

## 📈 Future Enhancements

- [ ] Real-time model retraining
- [ ] A/B testing framework
- [ ] Advanced visualizations
- [ ] Email/SMS alerts for high-risk customers
- [ ] API endpoints for integration
- [ ] Multi-tenant support

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For support and questions:
- Create an issue in the GitHub repository
- Contact the development team
- Check the documentation for common solutions
