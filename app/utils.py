"""
Utility functions for the Customer Intelligence Streamlit App
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Tuple, Any
import pickle
import joblib
from pathlib import Path

# Define the project root path relative to this script
PROJECT_ROOT = Path(__file__).parent.parent
import os
from pathlib import Path

# Get the directory of this script and the project root
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent

class ModelLoader:
    """Class to handle model loading and caching."""
    
    @staticmethod
    @st.cache_resource  
    def load_random_forest_models() -> Dict[str, Any]:
        """Load Random Forest models if available."""
        models = {}
        model_base = PROJECT_ROOT / 'models' / 'random_forest' / 'models'
        model_paths = {
            'clv': model_base / 'clv_random_forest_model.pkl',
            'churn': model_base / 'churn_random_forest_model.pkl',
            'segmentation_kmeans': model_base / 'segmentation_kmeans_model.pkl',
            'scaler': model_base / 'segmentation_scaler.pkl'
        }
        
        for key, path in model_paths.items():
            if path.exists():
                try:
                    # Use pickle for .pkl files, joblib for .joblib files
                    if path.suffix == '.pkl':
                        with open(path, 'rb') as f:
                            models[key] = pickle.load(f)
                    else:
                        models[key] = joblib.load(str(path))
                    st.success(f"✅ Loaded random forest {key} model")
                except Exception as e:
                    st.warning(f"⚠️ Error loading random forest {key} model: {e}")
            else:
                st.warning(f"⚠️ Random forest {key} model not found at: {path}")
        
        return models
    
    @staticmethod
    @st.cache_resource
    def load_boosting_models() -> Dict[str, Any]:
        """Load Boosting models if available."""
        models = {}
        model_base = PROJECT_ROOT / 'models' / 'boosting'
        model_paths = {
            'loyalty': model_base / 'loyalty_score_model.joblib',
            'purchase': model_base / 'purchase_amount_model.joblib',
            'clustering': model_base / 'customer_clustering_model.joblib',
            'preprocessor': model_base / 'data_preprocessor.joblib'
        }
        
        for name, path in model_paths.items():
            if path.exists():
                try:
                    models[name] = joblib.load(str(path))
                    st.success(f"✅ Loaded boosting {name} model")
                except Exception as e:
                    st.warning(f"⚠️ Error loading boosting {name} model: {e}")
            else:
                st.warning(f"⚠️ Boosting {name} model not found at: {path}")
        
        return models

class FeatureEngineering:
    """Class for feature engineering and preprocessing."""
    
    @staticmethod
    def create_derived_features(customer_data: Dict) -> Dict:
        """Create derived features from basic customer data."""
        enhanced_data = customer_data.copy()
        
        # Calculate derived features
        enhanced_data['spend_per_purchase'] = (
            customer_data['purchase_amount'] / max(customer_data['purchase_frequency'], 1)
        )
        enhanced_data['spend_to_income_ratio'] = (
            customer_data['purchase_amount'] / customer_data['annual_income']
        )
        
        # Create categorical features
        enhanced_data['customer_value_score'] = (
            enhanced_data['spend_to_income_ratio'] * customer_data['loyalty_score'] / 10
        )
        enhanced_data['is_loyal'] = int(customer_data['loyalty_score'] >= 7.0)
        enhanced_data['is_frequent'] = int(customer_data['purchase_frequency'] >= 20)
        enhanced_data['is_high_value'] = int(customer_data['purchase_amount'] >= 500)
        
        # Age and income brackets
        if customer_data['age'] < 30:
            enhanced_data['age_group'] = 'Young_Adult'
        elif customer_data['age'] < 50:
            enhanced_data['age_group'] = 'Adult'
        else:
            enhanced_data['age_group'] = 'Middle_Aged'
        
        if customer_data['annual_income'] < 40000:
            enhanced_data['income_bracket'] = 'Low_Income'
        elif customer_data['annual_income'] < 70000:
            enhanced_data['income_bracket'] = 'Medium_Income'
        else:
            enhanced_data['income_bracket'] = 'High_Income'
        
        return enhanced_data
    
    @staticmethod
    def get_feature_sets() -> Dict[str, list]:
        """Get predefined feature sets for different models."""
        return {
            'clv_features': [
                'age', 'annual_income', 'loyalty_score', 'spend_to_income_ratio',
                'region_North', 'region_South', 'region_West'
            ],
            'churn_features': [
                'age', 'annual_income', 'customer_value_score', 'purchase_frequency',
                'is_loyal', 'is_frequent', 'region_North', 'region_South', 'region_West'
            ],
            'segmentation_features': [
                'age', 'annual_income', 'spend_to_income_ratio'
            ]
        }

class PredictionInterpreter:
    """Class for interpreting model predictions."""
    
    @staticmethod
    def interpret_clv(clv_value: float) -> Tuple[str, str]:
        """Interpret CLV prediction."""
        if clv_value > 800:
            return "High Value", "prediction-high"
        elif clv_value > 400:
            return "Medium Value", "prediction-medium"
        else:
            return "Low Value", "prediction-low"
    
    @staticmethod
    def interpret_churn(churn_prob: float) -> Tuple[str, str]:
        """Interpret churn probability."""
        if churn_prob > 0.7:
            return "High Risk", "prediction-high"
        elif churn_prob > 0.4:
            return "Medium Risk", "prediction-medium"
        else:
            return "Low Risk", "prediction-low"
    
    @staticmethod
    def interpret_segment(segment_id: int) -> str:
        """Interpret customer segment."""
        segment_names = {
            0: "Budget Conscious",
            1: "Premium Customers",
            2: "Growing Potential", 
            3: "High Spenders"
        }
        return segment_names.get(segment_id, f"Segment {segment_id}")
    
    @staticmethod
    def calculate_composite_score(clv: float, churn_prob: float) -> float:
        """Calculate a composite customer score."""
        return (clv / 1000) * (1 - churn_prob) * 100

class RecommendationEngine:
    """Class for generating customer recommendations."""
    
    @staticmethod
    def generate_recommendations(clv_category: str, churn_risk: str, 
                               segment_name: str, customer_data: Dict) -> list:
        """Generate actionable recommendations based on predictions."""
        recommendations = []
        
        # CLV-based recommendations
        if clv_category == "High Value":
            recommendations.extend([
                "🌟 **VIP Treatment**: Provide premium customer service and exclusive offers",
                "🎁 **Loyalty Rewards**: Enroll in highest tier loyalty program",
                "👑 **Exclusive Access**: Offer early access to new products"
            ])
        elif clv_category == "Medium Value":
            recommendations.extend([
                "📈 **Upsell Opportunities**: Recommend premium products and services",
                "🎯 **Targeted Campaigns**: Include in growth-focused marketing campaigns",
                "🔄 **Cross-sell**: Suggest complementary products"
            ])
        else:
            recommendations.extend([
                "💰 **Value Building**: Focus on increasing purchase frequency and amount",
                "📧 **Education**: Provide product education and usage tips",
                "🎁 **Incentives**: Offer first-purchase or loyalty discounts"
            ])
        
        # Churn-based recommendations
        if churn_risk == "High Risk":
            recommendations.extend([
                "🚨 **Immediate Attention**: Contact within 24 hours with retention offer",
                "💬 **Feedback**: Conduct survey to understand pain points",
                "🎯 **Retention Campaign**: Deploy targeted retention strategy"
            ])
        elif churn_risk == "Medium Risk":
            recommendations.extend([
                "🔔 **Engagement**: Increase touchpoints and personalized communication",
                "🎁 **Incentives**: Provide targeted discounts or bonuses",
                "📊 **Monitor**: Track engagement metrics closely"
            ])
        
        # Segment-based recommendations
        if "Premium" in segment_name:
            recommendations.append("👑 **Premium Experience**: Offer exclusive products and services")
        elif "Budget" in segment_name:
            recommendations.append("💵 **Value Focus**: Emphasize cost savings and value propositions")
        elif "Growing" in segment_name:
            recommendations.append("🌱 **Development**: Nurture with educational content and gradual upgrades")
        elif "High Spenders" in segment_name:
            recommendations.append("💎 **Luxury Options**: Present high-end product lines")
        
        # Behavior-specific recommendations
        if customer_data['loyalty_score'] < 5:
            recommendations.append("❤️ **Loyalty Building**: Implement loyalty program enrollment")
        
        if customer_data['purchase_frequency'] < 10:
            recommendations.append("🔄 **Frequency Boost**: Create purchase frequency incentives")
        
        return recommendations

class DataValidator:
    """Class for validating input data."""
    
    @staticmethod
    def validate_customer_data(customer_data: Dict) -> Tuple[bool, str]:
        """Validate customer input data."""
        required_fields = ['age', 'annual_income', 'purchase_amount', 
                          'loyalty_score', 'purchase_frequency']
        
        for field in required_fields:
            if field not in customer_data:
                return False, f"Missing required field: {field}"
            
            if customer_data[field] is None or customer_data[field] < 0:
                return False, f"Invalid value for {field}: must be positive"
        
        # Specific validations
        if customer_data['age'] < 18 or customer_data['age'] > 100:
            return False, "Age must be between 18 and 100"
        
        if customer_data['annual_income'] < 10000 or customer_data['annual_income'] > 1000000:
            return False, "Annual income must be between $10,000 and $1,000,000"
        
        if customer_data['loyalty_score'] < 0 or customer_data['loyalty_score'] > 10:
            return False, "Loyalty score must be between 0 and 10"
        
        return True, "Valid"
