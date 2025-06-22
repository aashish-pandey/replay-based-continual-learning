#!/bin/bash

#Exit on error
set -e

# create virtual environment if not present
if [ ! -d "venv" ]; then
    echo "Creating python virtual environment!"
    python3 -m venv venv
fi

#Activate environment
echo "Activating virtual environment"
source venv/bin/activate

#install dependencies
echo "Installing dependencies from requirements.txt"
pip install --upgrade pip
pip install -r requirements.txt

echo "Environment ready. Run: source venv/bin/activate to activate manually next time"