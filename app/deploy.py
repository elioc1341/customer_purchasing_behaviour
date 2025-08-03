#!/usr/bin/env python3
"""
Deployment script for Customer Intelligence Streamlit App
Run this from the app/ directory to start the application.
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """Set up the environment for the Streamlit app."""
    print("🚀 Customer Intelligence Platform - App Deployment")
    print("=" * 50)
    
    # Check if we're in the app directory
    current_dir = Path.cwd()
    if current_dir.name != 'app':
        print("❌ Please run this script from the app/ directory")
        print(f"Current directory: {current_dir}")
        print("Expected: .../customer_purchasing_behaviour/app/")
        sys.exit(1)
    
    # Check parent directory structure
    parent_dir = current_dir.parent
    required_dirs = ['data', 'models']
    missing_dirs = []
    
    for dir_name in required_dirs:
        if not (parent_dir / dir_name).exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"❌ Missing required directories: {missing_dirs}")
        print("Please ensure you're in the correct project structure.")
        sys.exit(1)
    
    print("✅ Directory structure validated")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def run_app():
    """Run the Streamlit app."""
    print("\n🌐 Starting Streamlit application...")
    print("App will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        # Run streamlit with the config file
        subprocess.run([
            'streamlit', 'run', 'streamlit_app.py',
            '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false'
        ])
    except KeyboardInterrupt:
        print("\n\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

def main():
    """Main deployment function."""
    if not setup_environment():
        sys.exit(1)
    
    if not install_dependencies():
        print("⚠️  Continuing without installing dependencies...")
        print("Make sure to run: pip install -r requirements.txt")
    
    run_app()

if __name__ == "__main__":
    main()
