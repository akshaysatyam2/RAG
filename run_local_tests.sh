#!/usr/bin/env bash
# Unified local test runner for Flask GraphRAG system
set -euo pipefail

echo "Starting unified local test runner for GraphRAG system..."

# Check Python version
python3 --version

# Create/activate virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

# Install requirements
echo "Installing requirements from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# Run backend pytest tests (excluding playwright frontend tests)
echo "Running backend tests..."
pytest tests/ -m "not frontend" -v

echo "All tests finished."
exit 0
