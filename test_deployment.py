#!/usr/bin/env python3
"""
Test script to verify all dependencies and models are available for the Streamlit app.
Run this from the project root directory.
"""

import os
import sys
from pathlib import Path

def test_dependencies():
    """Test if all required Python packages are available."""
    print("🔍 Testing Python Dependencies...")
    
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'sklearn', 
        'plotly', 'seaborn', 'matplotlib', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT FOUND")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: cd app/ && pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies available!")
        return True

def test_model_files():
    """Test if all required model files exist."""
    print("\n🤖 Testing Model Files...")
    
    required_models = [
        'models/random_forest/models/clv_random_forest_model.pkl',
        'models/random_forest/models/churn_random_forest_model.pkl',
        'models/random_forest/models/segmentation_kmeans_model.pkl',
        'models/random_forest/models/segmentation_scaler.pkl'
    ]
    
    optional_models = [
        'models/boosting/loyalty_score_model.joblib',
        'models/boosting/purchase_amount_model.joblib',
        'models/boosting/customer_clustering_model.joblib',
        'models/boosting/data_preprocessor.joblib'
    ]
    
    missing_required = []
    
    # Check required models
    for model_path in required_models:
        if os.path.exists(model_path):
            print(f"✅ {model_path}")
        else:
            print(f"❌ {model_path} - NOT FOUND")
            missing_required.append(model_path)
    
    # Check optional models
    print("\nOptional boosting models:")
    for model_path in optional_models:
        if os.path.exists(model_path):
            print(f"✅ {model_path}")
        else:
            print(f"⚠️  {model_path} - NOT FOUND (optional)")
    
    if missing_required:
        print(f"\n❌ Missing required models: {len(missing_required)}")
        print("Please ensure you have trained and saved the Random Forest models.")
        return False
    else:
        print("\n✅ All required models available!")
        return True

def test_data_files():
    """Test if required data files exist."""
    print("\n📊 Testing Data Files...")
    
    data_file = 'data/processed/df_eng_customer_purchasing_features.csv'
    
    if os.path.exists(data_file):
        print(f"✅ {data_file}")
        
        # Check if file is readable and has data
        try:
            import pandas as pd
            df = pd.read_csv(data_file)
            print(f"   📈 Dataset shape: {df.shape}")
            print(f"   📋 Columns: {len(df.columns)}")
            return True
        except Exception as e:
            print(f"❌ Error reading data file: {e}")
            return False
    else:
        print(f"❌ {data_file} - NOT FOUND")
        print("Please ensure you have the processed dataset available.")
        return False

def test_streamlit_config():
    """Test Streamlit configuration."""
    print("\n⚙️  Testing Streamlit Configuration...")
    
    config_file = 'app/.streamlit/config.toml'
    app_file = 'app/streamlit_app.py'
    
    if os.path.exists(config_file):
        print(f"✅ {config_file}")
    else:
        print(f"⚠️  {config_file} - NOT FOUND (will use defaults)")
    
    if os.path.exists(app_file):
        print(f"✅ {app_file}")
        return True
    else:
        print(f"❌ {app_file} - NOT FOUND")
        return False

def main():
    """Run all tests."""
    print("🚀 Customer Intelligence Platform - Deployment Test")
    print("=" * 50)
    
    tests = [
        test_dependencies,
        test_model_files,
        test_data_files,
        test_streamlit_config
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("📋 SUMMARY")
    print("=" * 50)
    
    if all(results):
        print("🎉 ALL TESTS PASSED!")
        print("\nYou can now run the Streamlit app with:")
        print("   cd app/")
        print("   streamlit run streamlit_app.py")
        print("\nOr use the deployment script:")
        print("   cd app/ && python deploy.py")
        print("\nThe app will be available at: http://localhost:8501")
    else:
        print("❌ SOME TESTS FAILED!")
        print("\nPlease fix the issues above before running the app.")
        
        failed_tests = []
        test_names = ["Dependencies", "Model Files", "Data Files", "Streamlit Config"]
        for i, result in enumerate(results):
            if not result:
                failed_tests.append(test_names[i])
        
        print(f"\nFailed tests: {', '.join(failed_tests)}")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
