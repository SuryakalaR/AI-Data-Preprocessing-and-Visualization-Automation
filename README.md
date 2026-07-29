# AI-Powered Data Cleaner and Dashboard 📊🤖

This project is a Streamlit web application that combines a local Large Language Model (LLM) via Ollama with an always-on interactive dashboard. Upload a CSV and instantly explore it visually — then optionally let the AI summarize the dataset and generate custom preprocessing code for it.

## ✨ Features

- **⬆️ CSV Upload** — simple drag-and-drop upload interface.
- **📊 Instant Interactive Dashboard** — renders immediately on upload, no LLM required:
  - KPI cards: row count, column count, missing cells, duplicate rows
  - **Distributions tab** — pick any numeric column, view histogram + box plot overlay
  - **Categorical Breakdown tab** — pick a categorical column, adjustable "Top N" bar chart
  - **Correlation tab** — interactive Plotly heatmap for numeric columns
  - **Data Explorer tab** — filter rows by any column (multiselect for text, range slider for numbers) and browse the live-filtered table
- **🤖 On-Demand AI Summary** — generates a plain-language description and key insights about the dataset's structure, missing values, and data types, run only when you click the button (so you're never stuck waiting on it just to see your data).
- **🧹 On-Demand AI Preprocessing**:
  - Generates Python code dynamically based on your dataset's actual columns and types
  - Handles missing values (numeric → median, categorical → mode)
  - Drops low-variance or high-missing columns
  - Handles outliers via IQR
  - Label-encodes categorical columns
  - **Import safety check** — generated code is scanned for disallowed/hallucinated imports (e.g. a nonexistent `sklearn.metrics.variance_score`) *before* execution, and safely falls back to the raw dataset instead of crashing
- **👀 Side-by-Side Comparison** — original vs. cleaned dataset tables
- **💻 Code Transparency** — view the exact Python code the LLM generated for preprocessing, in a collapsible panel
- **🚀 Local LLM Powered** — uses Ollama to run models like `llama3.2:3b` locally, keeping your data private
- **⚡ Fast by design** — LLM responses are cached, and the dashboard/data state persist across interactions via `st.session_state`, so clicking a tab or dragging a slider never re-triggers a slow model call

## ⚙️ How it Works

1. **Upload** — you upload a CSV via the Streamlit interface.
2. **Dashboard (instant)** — the app reads the CSV into a Pandas DataFrame and immediately renders the interactive dashboard on the raw data — no waiting on any AI step.
3. **AI Summary (on demand)** — clicking "🤖 Run AI Summary & Preprocessing" sends dataset metadata (columns, dtypes, missing values, shape) to the local Ollama endpoint, which returns a written summary and insights.
4. **AI Preprocessing (on demand)** — a second prompt asks the LLM to write preprocessing code tailored to your dataset's actual columns.
5. **Safety Check** — before running that code, it's scanned for disallowed imports. If it tries to import something outside an approved allowlist, preprocessing is skipped and the raw dataset is used instead — no crash, no silent corruption.
6. **Execute** — the generated code runs in a shared namespace (so helper functions the LLM defines can call each other correctly), and the cleaned result is read back from disk to guarantee consistency.
7. **Dashboard Refresh** — the same dashboard now reflects the cleaned dataset instead of the raw one.

## 🛠️ Technology Stack

- **Frontend:** Streamlit
- **Backend/Logic:** Python
- **Data Handling:** Pandas, NumPy
- **Machine Learning/Preprocessing:** Scikit-learn
- **Visualization:** Plotly (interactive charts, not static images)
- **LLM Interaction:** Requests library, Ollama
- **LLM:** Configured for `llama3.2:3b` (or any other Ollama-compatible model)

## 📋 Prerequisites

- **Python:** Version 3.8 or higher
- **Ollama:** Must be installed and running locally — download from [ollama.ai](https://ollama.ai)
- **LLM Model:** Pull the model used by the script:
  ```bash
  ollama pull llama3.2:3b
  ```
  (Or edit `MODEL_NAME` in the script to use a different model you have pulled.)
- **pip:** Python package installer

## 🚀 Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SuryakalaR/AI-Data-Preprocessing-and-Visualization-Automation.git
   cd AI-Data-Preprocessing-and-Visualization-Automation
   ```

2. **Create `requirements.txt`** (if not already present) with:
   ```txt
   streamlit
   pandas
   numpy
   plotly
   scikit-learn
   requests
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure Ollama is running** — start the Ollama app/service and confirm it's reachable, usually at `http://localhost:11434`:
   ```bash
   curl http://localhost:11434/api/tags
   ```

## ▶️ Running the Application

```bash
streamlit run AI_Data_Cleaning.py
```

The app should automatically open in your browser at `http://localhost:8501`.

## ⚙️ Configuration

Constants you might want to adjust at the top of `AI_Data_Cleaning.py`:

| Constant | Default | Description |
|---|---|---|
| `OLLAMA_API_URL` | `http://localhost:11434/api/generate` | URL for your running Ollama instance |
| `MODEL_NAME` | `llama3.2:3b` | Ollama model tag to use — must already be pulled |
| `TEMP_DIR` | `temp` | Directory for uploaded/cleaned datasets and generated code. Auto-created; **not** committed to git (see `.gitignore`) |
| `ALLOWED_IMPORT_MODULES` | pandas, numpy, os, re, sklearn, sklearn.impute, sklearn.preprocessing | Whitelist of modules the AI-generated preprocessing code is allowed to import |



