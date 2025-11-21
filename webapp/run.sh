#!/bin/bash

# Quick Start Script for AI Emotion Recognition Web App

echo "🚀 Starting AI Emotion Recognition Web Application..."
echo ""

# Check if model file exists in parent directory
if [ ! -f "../mod_my_model01.keras" ]; then
    echo "⚠️  Warning: Model file '../mod_my_model01.keras' not found!"
    echo "   Please train the model first using face022.ipynb in the parent directory"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

# Run Streamlit app
echo ""
echo "✅ Starting Streamlit application..."
echo "🌐 Open your browser at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run webapp.py

