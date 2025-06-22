#!/usr/bin/env python3
"""
Test script for the Mental Health Predictor web application
"""

import requests
import time

def test_app():
    """Test the Flask application"""
    base_url = "http://localhost:5000"
    
    try:
        # Test the home page
        print("Testing home page...")
        response = requests.get(base_url)
        if response.status_code == 200:
            print("✅ Home page loaded successfully")
        else:
            print(f"❌ Home page failed with status code: {response.status_code}")
            return False
        
        # Test prediction endpoint
        print("Testing prediction endpoint...")
        test_data = {
            'Age': '25',
            'Gender': '0',  # Male
            'Family_history': '0'  # No family history
        }
        
        response = requests.post(f"{base_url}/predict", data=test_data)
        if response.status_code == 200:
            print("✅ Prediction endpoint working")
            print(f"Response length: {len(response.text)} characters")
        else:
            print(f"❌ Prediction failed with status code: {response.status_code}")
            return False
            
        print("🎉 All tests passed! The application is working correctly.")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the application. Make sure it's running on http://localhost:5000")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    print("Starting Mental Health Predictor test...")
    print("Make sure the Flask app is running on http://localhost:5000")
    print("-" * 50)
    
    # Wait a moment for the app to start
    time.sleep(2)
    
    success = test_app()
    
    if success:
        print("\n🚀 Ready for deployment! You can now create your pull request.")
    else:
        print("\n⚠️  Some tests failed. Please check the application.") 