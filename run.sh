#!/bin/bash
#
# run.sh: Automatically install, set up, and run PCBTagent
#
# This script is intended to provide a one-click reproduction experience
# for paper reviewers.
# It will:
# 0. Check if the .env file and API keys are configured
# 1. Create an isolated Conda environment
# 2. Install all Python dependencies
# 3. Run the src/main.py script to process the sample data
#
# Assumptions:
# - Conda (Anaconda / Miniconda) is installed.

# --- 1. Configuration ---
# Define the environment name
ENV_NAME="pcbocr_env"
# Define the Python version (a stable, common version is recommended)
PYTHON_VERSION="3.12"
# Define the .env file path
ENV_FILE=".env"

echo "[INFO] Starting PCBTagent automated reproduction script..."

# --- 0. Check .env File and API Key ---
echo "[0/4] Checking .env configuration ($ENV_FILE)..."

if [ ! -f "$ENV_FILE" ]; then
    echo "[ERROR] Configuration file '$ENV_FILE' not found."
    echo "Please ensure '$ENV_FILE' exists in the project root and contains a valid API key."
    exit 1
fi

# Read LLM_PROVIDER from .env file (and remove potential \r characters)
PROVIDER=$(grep -E "^LLM_PROVIDER=" "$ENV_FILE" | cut -d'=' -f2 | tr -d '[:space:]')

KEY_FILLED=false
if [ "$PROVIDER" == "gpt" ]; then
    # Check if OPENAI_API_KEY exists and is non-empty (simple check for sk-)
    if grep -q -E "^OPENAI_API_KEY=sk-" "$ENV_FILE"; then
        KEY_FILLED=true
    fi
    if [ "$KEY_FILLED" = false ]; then
        echo "[ERROR] LLM_PROVIDER (in $ENV_FILE) is set to 'gpt', but 'OPENAI_API_KEY' is missing or invalid."
        echo "Please set OPENAI_API_KEY in '$ENV_FILE' (Format: OPENAI_API_KEY=sk-...)"
        exit 1
    fi
    echo "[INFO] Detected LLM_PROVIDER=gpt. OPENAI_API_KEY found."

elif [ "$PROVIDER" == "deepseek" ]; then
    # Check if DEEPSEEK_API_KEY exists and is non-empty (simple check for sk-)
    if grep -q -E "^DEEPSEEK_API_KEY=sk-" "$ENV_FILE"; then
        KEY_FILLED=true
    fi
    if [ "$KEY_FILLED" = false ]; then
        echo "[ERROR] LLM_PROVIDER (in $ENV_FILE) is set to 'deepseek', but 'DEEPSEEK_API_KEY' is missing or invalid."
        echo "Please set DEEPSEEK_API_KEY in '$ENV_FILE' (Format: DEEPSEEK_API_KEY=sk-...)"
        exit 1
    fi
    echo "[INFO] Detected LLM_PROVIDER=deepseek. DEEPSEEK_API_KEY found."

else
    echo "[WARN] LLM_PROVIDER ('$PROVIDER') in .env is not 'gpt' or 'deepseek'."
    echo "Script will continue, but the Python script (src/main.py) may fail if API access is required."
fi


# --- 1. Create Conda Environment ---
echo "[1/4] Creating Conda environment '$ENV_NAME' (using Python $PYTHON_VERSION)..."
# '-y' flag automatically confirms all prompts
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Check if environment creation was successful
if [ $? -ne 0 ]; then
    echo "[ERROR] Conda environment creation failed. Please check your Conda installation."
    exit 1
fi

# --- 2. Install Dependencies ---
echo "[2/4] Activating environment and installing dependencies (from requirements.txt)..."
# 'conda run' is the recommended way to execute commands in a specific environment within scripts
conda run -n $ENV_NAME pip install -r requirements.txt

# Check if dependency installation was successful
if [ $? -ne 0 ]; then
    echo "[ERROR] Dependency installation failed. Please check 'requirements.txt'."
    exit 1
fi

# --- 3. Run Main Script ---
echo "[3/4] Running main script (src/main.py)..."
# This command executes main.py within the $ENV_NAME environment
# The script will automatically load configurations from config.py and .env
conda run -n $ENV_NAME python src/main.py

# Check if Python script execution was successful
if [ $? -ne 0 ]; then
    echo "[ERROR] Python script execution failed. Please check 'logs.log' for details."
    exit 1
fi

# --- 4. Finish ---
echo "[4/4] Script execution finished."
echo "=== [SUCCESS] ==="
echo "Output files are located in the OUTPUT_PATH specified in your config.py or .env."
echo "To run manually again, use: 'conda activate $ENV_NAME' then 'python src/main.py --input data/agent_input_text --output data/knowledgebase_output'"