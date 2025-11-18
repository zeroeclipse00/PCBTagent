@echo off
setlocal
REM
REM run.bat: Automatically install, set up, and run PCBTagent
REM
REM This script is intended to provide a one-click reproduction experience
REM for paper reviewers.
REM It will:
REM 0. Check if the .env file and API keys are configured
REM 1. Create an isolated Conda environment
REM 2. Install all Python dependencies
REM 3. Run the src/main.py script to process the sample data
REM
REM Assumptions:
REM - Conda (Anaconda / Miniconda) is installed.

REM --- 1. Configuration ---
REM Define the environment name
set "ENV_NAME=pcbocr_env"
REM Define the Python version (a stable, common version is recommended)
set "PYTHON_VERSION=3.12"
REM Define the .env file path
set "ENV_FILE=.env"

echo [INFO] Starting PCBTagent automated reproduction script...

REM --- 0. Check .env File and API Key ---
echo [0/4] Checking .env configuration (%ENV_FILE%)...

if not exist "%ENV_FILE%" (
    echo [ERROR] Configuration file '%ENV_FILE%' not found.
    echo Please ensure '%ENV_FILE%' exists in the project root and contains a valid API key.
    exit /b 1
)

REM Read LLM_PROVIDER from .env file
REM We use 'findstr' to find the line, and 'for /f' to parse it.
REM "tokens=1,* delims==" splits the line "KEY=VALUE" into "KEY" (%%i) and "VALUE" (%%j)
set "PROVIDER="
for /f "tokens=1,* delims==" %%i in ('findstr /B "LLM_PROVIDER=" "%ENV_FILE%"') do (
    set "PROVIDER=%%j"
)

REM Clean up potential whitespace (approximates 'tr -d [:space:]')
if defined PROVIDER (
    REM Remove all spaces
    set "PROVIDER=%PROVIDER: =%"
    REM This 'for' loop trick removes potential trailing carriage returns (\r)
    for /f "delims=" %%k in ("%PROVIDER%") do set "PROVIDER=%%k"
)

set "KEY_FILLED=false"
if /I "%PROVIDER%" == "gpt" (
    REM Check if OPENAI_API_KEY exists and is non-empty (simple check for sk-)
    REM 'findstr' sets ERRORLEVEL to 0 if found, 1 if not found. '> nul' hides output.
    findstr /R /C:"^OPENAI_API_KEY=sk-" "%ENV_FILE%" > nul
    if %ERRORLEVEL% equ 0 (
        set "KEY_FILLED=true"
    )
    if "%KEY_FILLED%" == "false" (
        echo [ERROR] LLM_PROVIDER (in %ENV_FILE%) is set to 'gpt', but 'OPENAI_API_KEY' is missing or invalid.
        echo Please set OPENAI_API_KEY in '%ENV_FILE%' (Format: OPENAI_API_KEY=sk-...)
        exit /b 1
    )
    echo [INFO] Detected LLM_PROVIDER=gpt. OPENAI_API_KEY found.

) else if /I "%PROVIDER%" == "deepseek" (
    REM Check if DEEPSEEK_API_KEY exists and is non-empty (simple check for sk-)
    findstr /R /C:"^DEEPSEEK_API_KEY=sk-" "%ENV_FILE%" > nul
    if %ERRORLEVEL% equ 0 (
        set "KEY_FILLED=true"
    )
    if "%KEY_FILLED%" == "false" (
        echo [ERROR] LLM_PROVIDER (in %ENV_FILE%) is set to 'deepseek', but 'DEEPSEEK_API_KEY' is missing or invalid.
        echo Please set DEEPSEEK_API_KEY in '%ENV_FILE%' (Format: DEEPSEEK_API_KEY=sk-...)
        exit /b 1
    )
    echo [INFO] Detected LLM_PROVIDER=deepseek. DEEPSEEK_API_KEY found.

) else (
    echo [WARN] LLM_PROVIDER ('%PROVIDER%') in .env is not 'gpt' or 'deepseek'.
    echo Script will continue, but the Python script (src/main.py) may fail if API access is required.
)


REM --- 1. Create Conda Environment ---
echo [1/4] Creating Conda environment '%ENV_NAME%' (using Python %PYTHON_VERSION%)...
REM '-y' flag automatically confirms all prompts
conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y

REM Check if environment creation was successful
REM %ERRORLEVEL% neq 0 checks if the last command failed
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Conda environment creation failed. Please check your Conda installation.
    exit /b 1
)

REM --- 2. Install Dependencies ---
echo [2/4] Activating environment and installing dependencies (from requirements.txt)...
REM 'conda run' is the recommended way to execute commands in a specific environment within scripts
conda run -n %ENV_NAME% pip install -r requirements.txt

REM Check if dependency installation was successful
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Dependency installation failed. Please check 'requirements.txt'.
    exit /b 1
)

REM --- 3. Run Main Script ---
echo [3/4] Running main script (src/main.py)...
REM This command executes main.py within the %ENV_NAME% environment
REM The script will automatically load configurations from config.py and .env
conda run -n %ENV_NAME% python src/main.py

REM Check if Python script execution was successful
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python script execution failed. Please check 'logs.log' for details.
    exit /b 1
)

REM --- 4. Finish ---
echo [4/4] Script execution finished.
echo === [SUCCESS] ===
echo Output files are located in the OUTPUT_PATH specified in your config.py or .env.
echo To run manually again, use: "conda activate %ENV_NAME%" then "python src/main.py --input data/agent_input_text --output data/knowledgebase_output"

REM 'endlocal' is implicitly called here, or use 'exit /b 0' for success
exit /b 0