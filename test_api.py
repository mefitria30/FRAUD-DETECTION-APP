"""
Test Script untuk Fraud Detection API
======================================
Script untuk testing API endpoints
"""

import requests
import json

# Base URL
BASE_URL = "http://localhost:5000"

def test_root():
    """Test root endpoint"""
    print("\n" + "="*70)
    print("Testing ROOT endpoint")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_health():
    """Test health endpoint"""
    print("\n" + "="*70)
    print("Testing HEALTH endpoint")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_single_prediction():
    """Test single prediction"""
    print("\n" + "="*70)
    print("Testing SINGLE PREDICTION")
    print("="*70)
    
    # Test case 1: Fraud text
    data = {
        "text": "kontak via aplikasi aman komunikasi rahasia"
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"\nTest 1 - Fraud Text:")
    print(f"Input: {data['text']}")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test case 2: Normal text
    data = {
        "text": "belanja online di toko resmi"
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"\nTest 2 - Normal Text:")
    print(f"Input: {data['text']}")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_batch_prediction():
    """Test batch prediction"""
    print("\n" + "="*70)
    print("Testing BATCH PREDICTION")
    print("="*70)
    
    data = {
        "texts": [
            "kontak via aplikasi aman komunikasi rahasia",
            "transfer uang segera urgent butuh cepat",
            "belanja online di toko resmi",
            "pembayaran bulanan tagihan listrik"
        ]
    }
    
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nTotal Predictions: {result['count']}")
        print(f"\nResults:")
        
        for i, item in enumerate(result['results'], 1):
            status = "FRAUD" if item['is_fraud'] else "NORMAL"
            print(f"\n{i}. {status}")
            print(f"   Text: {item['text']}")
            print(f"   Fraud Probability: {item['fraud_probability']:.2f}%")
    else:
        print(f"Error: {response.text}")

def test_model_info():
    """Test model info endpoint"""
    print("\n" + "="*70)
    print("Testing MODEL INFO endpoint")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("🧪 FRAUD DETECTION API - TESTING SUITE")
    print("="*70)
    
    try:
        test_root()
        test_health()
        test_model_info()
        test_single_prediction()
        test_batch_prediction()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS COMPLETED!")
        print("="*70 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API server!")
        print("Please make sure the server is running:")
        print("  python app.py")
        print("  or")
        print("  uvicorn app:app --reload")
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")

if __name__ == "__main__":
    run_all_tests()