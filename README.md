# PCBOCR: A Knowledge-Driven Multi-Agent Framework for High-Precision Text Recognition in Complex PCB Schematics
This project is a correction tool that uses LLM to perform OCR on PCB schematics. It adopts a lightweight RAG strategy, which corrects the OCR results by providing domain knowledge and correction rules to the model.

## File Structure
```
 PCBTagent/
 │
 ├── data/
 │   ├── 
 │ 
 ├── resources/
 │   ├── knowledge_base_v2.json
 │   └── sampled_gts_unique_700_long_300_short.txt
 │
 ├── src/
 │   ├── utils/
 │   │   ├── logging_setup.py
 │   │   └── parser.py
 │   │
 │   ├── config.py
 │   ├── llm_clients.py
 │   ├── main.py
 │   ├── pipeline.py
 │   └── prompting.py
 │
 ├── .env.example
 ├── .gitignore
 ├── LICENSE
 ├── README.md
```

## 

## 

## Complete data link








# PCBTagent: [在这里填入你的论文副标题，例如：一个用于修正 PCB 丝印 OCR 的 LLM Agent]

This repository provides the code implementation for the paper:
**"[在这里填入你的论文标题]"**
(Submission ID: [在这里填入你的投稿ID])

This agent is designed to correct OCR errors in PCB (Printed Circuit Board) silk-screen text. It reads text files containing ground truth and OCR predictions, processes them using a Large Language Model (LLM), and outputs the corrected text.

## Prerequisites

-   **Conda**: The environment setup relies on Anaconda or Miniconda.
-   **API Key**: This project requires an LLM API key (e.g., OpenAI) to function. **We have already included a pre-configured `.env` file with a valid API key for review purposes.**
-   **Data**: Sample input data is included in the `[填写你的输入路径, e.g., data/input]` directory.

## Quickstart: One-Click Run

We provide an automated script (`run.sh`) to set up the environment, install dependencies, and run the complete pipeline on the sample data.

**On Linux/macOS:**

1.  Give the script execution permissions:
    ```bash
    chmod +x run.sh
    ```

2.  Execute the script:
    ```bash
    ./run.sh
    ```

The script will automatically:
1.  Create a Conda environment named `pcbt_env`.
2.  Install all required packages from `requirements.txt`.
3.  Run `src/main.py`, which processes files from the configured `INPUT_PATH`.
4.  Results will be saved in the configured `OUTPUT_PATH` (e.g., `data/output`).

## Manual Installation (Alternative)

If you prefer to set up the environment manually:

1.  **Create and activate Conda environment:**
    ```bash
    conda create -n pcbt_env python=3.10 -y
    conda activate pcbt_env
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the script:**
    (Ensure the `.env` file is present in the root directory)
    ```bash
    python src/main.py
    ```

## Project Structure

-   `src/main.py`: Main entry point. Handles file/folder processing.
-   `src/pipeline.py`: Core logic for processing batches of text.
-   `src/llm_clients.py`: Manages API calls to different LLM providers.
-   `src/utils/parser.py`: Handles parsing the `gt||ocr conf` format and rebuilding output lines.
-   `src/config.py`: Contains configuration variables.
-   `.env`: (Pre-configured) Contains environment-specific variables like API keys.
-   `requirements.txt`: List of Python dependencies.
-   `[data/input/ (示例)]`: Sample input `.txt` files.
-   `[data/output/ (示例)]`: Directory for corrected outputs.