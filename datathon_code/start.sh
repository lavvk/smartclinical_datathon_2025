#!/bin/bash

# SmartClinical Startup Script

echo "🏥 SmartClinical - Starting Application"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    exit 1
fi

# Check if model exists
if [ ! -f "risk_model.joblib" ]; then
    echo "📊 Model not found. Training model..."
    python3 train_model.py
    if [ $? -ne 0 ]; then
        echo "❌ Error: Model training failed. Please install dependencies first:"
        echo "   pip install -r requirements.txt"
        exit 1
    fi
    echo "✅ Model trained successfully"
    echo ""
fi

# Start API server in background
echo "🚀 Starting API server..."
python3 api.py &
API_PID=$!
sleep 3

# Check if API is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ API server is running on http://localhost:8000"
else
    echo "❌ Error: API server failed to start"
    kill $API_PID 2>/dev/null
    exit 1
fi

echo ""
echo "🌐 Starting Streamlit frontend..."
echo "   The dashboard will open in your browser"
echo ""
echo "⚠️  To stop the servers, press Ctrl+C"
echo ""

# Start Streamlit
streamlit run app.py

# Cleanup on exit
kill $API_PID 2>/dev/null
echo ""
echo "👋 Shutting down..."

