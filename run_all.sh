#!/usr/bin/env bash
# run_all.sh
# Convenience script to:
# 1) create a venv
# 2) install requirements
# 3) parse thecapicornist.txt into data/ files
# 4) run training (train.py)
# 5) launch the chatbot (chatbot.py)
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh
#
# Note: This script is for Unix-like systems (Linux / macOS). If you're on Windows,
# run the commands manually or adapt the script.

set -e

PYTHON=${PYTHON:-python3}
VENV_DIR=.venv_chatbot

echo "1) Creating virtualenv in ${VENV_DIR} (if missing)..."
if [ ! -d "${VENV_DIR}" ]; then
  ${PYTHON} -m venv ${VENV_DIR}
fi

echo "2) Activating venv and installing requirements..."
# shellcheck disable=SC1091
source ${VENV_DIR}/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "3) Parsing thecapicornist.txt -> data/galaxydev13.txt + data/daydreaming_val_76222.txt"
python3 split_logs.py --infile thecapicornist.txt --outdir data

echo "4) Training models (this may take a moment)..."
python3 train.py --bot1 data/galaxydev13.txt --bot2 data/daydreaming_val_76222.txt --out models

echo "5) Launching chatbot (interactive). Use /bot1 or /bot2 to switch personas, /quit to exit."
python3 chatbot.py --models models

# deactivate venv automatically when finished
deactivate || true