#!/bin/bash
cd "$(dirname "$0")/.."

# Check if venv exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Warning: 'venv' not found. Using system Python."
fi

export PYTHONPATH=$PYTHONPATH:$(pwd)

# Optimize Memory Allocation
# MALLOC_ARENA_MAX=2 prevents glibc from creating too many memory arenas, reducing fragmentation
export MALLOC_ARENA_MAX=2

echo "Starting DBFD..."
# Check for headless flag or default to headless if no display
if [ -z "$DISPLAY" ]; then
    echo "No display detected, forcing headless mode."
    python3 main.py --headless
else
    python3 main.py "$@"
fi
