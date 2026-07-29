import pandas as pd
import numpy as np
import requests
import re
import os
import uuid
from pathlib import Path

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"
TEMP_DIR = "temp"

# Modules the generated preprocessing code is allowed to import.
# Anything outside this list gets rejected BEFORE exec(), instead of
# blowing up mid-run with a hallucinated import like `variance_score`.
ALLOWED_IMPORT_MODULES = {
    "pandas", "numpy", "os", "re",
    "sklearn", "sklearn.impute", "sklearn.preprocessing",
}

os.makedirs(TEMP_DIR, exist_ok=True)

st.set_page_config(page_title="Dataset Preprocessing & Dashboard", layout="wide")
st.title("Dataset Preprocessing and Visualization")
st.write("Upload a CSV dataset to preprocess, summarize, and explore it in an interactive dashboard.")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")


# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------
def generate_summary_prompt(df):
    summary = {
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "shape": df.shape,
    }
    return (
        "You are a professional data analyst. Based on the summary of the dataset provided below, "
        "follow a step-by-step process to understand what the dataset is about and extract meaningful insights.\n\n"
        "First, identify the structure of the dataset from the number of rows, columns, and data types.\n"
        "Then, look at the missing values and identify where data quality issues might exist.\n"
        "Analyze column names and infer what kind of information the dataset contains.\n"
        "Think through each column, whether it's categorical or numeric, and what role it might play.\n"
        "Reflect on any visible patterns or points of interest.\n"
        "Use this thinking to write:\n\n"
        "*Description*: A short paragraph explaining what the dataset is likely about.\n"
        "*Insights*: At least 8 to 10 insightful observations based strictly on the provided summary.\n\n"
        "Be factual. Do not generate synthetic examples or code. Avoid hallucinations.\n\n"
        f"Dataset Summary:\n"
        f"Columns: {summary['columns']}\n"
        f"Missing Values: {summary['missing_values']}\n"
        f"Data Types: {summary['dtypes']}\n"
        f"Shape: {summary['shape']}\n\n"
    )


def generate_preprocessing_prompt(df, dataset_path, cleaned_path):
    summary = {
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
        "numeric_columns": df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
    }
    allowed = ", ".join(sorted(ALLOWED_IMPORT_MODULES))
    return (
        "You are a skilled Python data preprocessing expert.\n"
        "Your task is to generate robust and production-safe Python code that performs preprocessing "
        "on any given tabular dataset.\n\n"
        f"Dataset Path: {dataset_path}\nOutput Path: {cleaned_path}\n\n"
        "Dataset Overview:\n"
        f"Missing Values: {summary['missing_values']}\n"
        f"Data Types: {summary['dtypes']}\n"
        f"Categorical Columns: {summary['categorical_columns']}\n"
        f"Numeric Columns: {summary['numeric_columns']}\n\n"
        "Instructions:\n"
        "- Load the dataset from the given path into a DataFrame.\n"
        "- Dynamically identify numeric and categorical columns using select_dtypes.\n"
        "- Drop columns with more than 80% missing values or columns with >=90% identical values.\n"
        "- Impute missing numeric columns with median using SimpleImputer.\n"
        "- Impute missing categorical columns with mode using SimpleImputer.\n"
        "- For categorical columns, use a single instance of LabelEncoder, applied within a loop.\n"
        "- Handle outliers using IQR (set outliers to NaN, then re-impute).\n"
        "- Ensure all column operations include checks for existence and avoid hardcoding column names.\n"
        "- Save the cleaned dataset to the provided output path.\n\n"
        "Constraints:\n"
        f"- ONLY import from these modules, nothing else: {allowed}.\n"
        "- Do NOT import anything from sklearn.metrics or sklearn.model_selection.\n"
        "- Do not hardcode column names.\n"
        "- Do not use non-standard or made-up functions. Only use real, well-known sklearn/pandas/numpy APIs.\n"
        "- Ensure the code is fully executable in a Python script.\n\n"
        "Your Output:\n"
        "- Return only clean and complete Python code. No markdown, comments, or explanations.\n"
    )


# ----------------------------------------------------------------------------
# LLM helpers
# ----------------------------------------------------------------------------
def clean_llm_response(response):
    """Strip markdown code fences so the result can be safely exec()'d."""
    if not response:
        return None
    cleaned = response.strip()
    cleaned = re.sub(r'^```(?:python)?\s*\n?', '', cleaned)
    cleaned = re.sub(r'\n?```\s*$', '', cleaned)
    return cleaned.strip()


@st.cache_data(show_spinner=False)
def get_llama_response(prompt):
    # Cached: identical prompts (e.g. re-running on the same dataset after a
    # widget click elsewhere in the app) return instantly instead of
    # re-hitting Ollama every single Streamlit rerun.
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=300
        )
        if response.status_code == 200:
            return response.json().get("response")
        st.warning(f"⚠️ Ollama returned status {response.status_code}: {response.text}")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"⚠️ Could not connect to Ollama at {OLLAMA_API_URL}. Is `ollama serve` running?")
        return None
    except requests.exceptions.Timeout:
        st.error("⚠️ Ollama request timed out. Try a smaller model or increase the timeout.")
        return None
    except Exception as e:
        st.error(f"⚠️ Ollama Error: {e}")
        return None


def find_disallowed_imports(code):
    """
    Scan generated code for `import X` / `from X import Y` statements and
    return any module names that aren't in ALLOWED_IMPORT_MODULES.
    This catches hallucinated imports (e.g. sklearn.metrics.variance_score)
    BEFORE exec() runs, instead of failing mid-script.
    """
    bad = []
    for match in re.finditer(r'^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))', code, re.MULTILINE):
        module = match.group(1) or match.group(2)
        root = module.split('.')[0]
        full = module
        if root not in ALLOWED_IMPORT_MODULES and full not in ALLOWED_IMPORT_MODULES:
            bad.append(module)
    return bad


def apply_generated_code(df, code, dataset_path, cleaned_path):
    if not code:
        return df

    disallowed = find_disallowed_imports(code)
    if disallowed:
        st.error(
            "⚠️ Generated preprocessing code tried to import unsupported/hallucinated "
            f"module(s): {', '.join(disallowed)}. Skipping preprocessing and using the "
            "raw uploaded dataset instead."
        )
        return df

    try:
        exec_namespace = {
            "df": df.copy(), "pd": pd, "np": np, "os": os,
            "SimpleImputer": SimpleImputer,
            "LabelEncoder": LabelEncoder,
            "StandardScaler": StandardScaler,
            "dataset_path": dataset_path, "cleaned_path": cleaned_path,
            "__builtins__": __builtins__,
        }
        with open(os.path.join(TEMP_DIR, "preprocessing_code.py"), "w") as f:
            f.write(code)
        # IMPORTANT: pass the SAME dict as both globals and locals.
        # exec(code, globals_dict, locals_dict) with two DIFFERENT dicts
        # makes Python treat the code like a class body: top-level `def`s
        # land in locals_dict, but when one of those functions calls
        # another (e.g. clean_data() calling load_data()), it looks the
        # name up in its __globals__ (the separate, near-empty globals
        # dict) and fails with "name 'load_data' is not defined" — even
        # though the function is clearly defined right there.
        # Using one shared dict makes it behave like a normal module.
        exec(code, exec_namespace)

        # The generated code's own top-level `df` variable may never get
        # reassigned (e.g. if cleaning happens inside a function that only
        # saves to disk locally). What IS reliable is that the code writes
        # to cleaned_path. Prefer reading that back if it exists and is
        # non-empty; otherwise fall back to whatever is in the namespace.
        if os.path.exists(cleaned_path) and os.path.getsize(cleaned_path) > 0:
            try:
                return pd.read_csv(cleaned_path)
            except Exception:
                pass
        return exec_namespace.get("df", df)
    except SyntaxError as e:
        st.error(f"⚠️ SyntaxError in preprocessing code: {e}. Using raw dataset instead.")
        return df
    except ImportError as e:
        st.error(f"⚠️ ImportError in preprocessing code: {e}. Using raw dataset instead.")
        return df
    except KeyError as e:
        st.error(f"⚠️ KeyError in preprocessing code: {e}. Check dataset columns. Using raw dataset instead.")
        return df
    except Exception as e:
        st.error(f"⚠️ Error in preprocessing code: {e}. Using raw dataset instead.")
        return df


# ----------------------------------------------------------------------------
# Interactive dashboard (no LLM involved — deterministic and always works)
# ----------------------------------------------------------------------------
def render_dashboard(df):
    st.subheader("📊 Interactive Dashboard")

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # ---- KPI row ----
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows", f"{df.shape[0]:,}")
    k2.metric("Columns", f"{df.shape[1]:,}")
    k3.metric("Missing cells", f"{int(df.isnull().sum().sum()):,}")
    k4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

    tab_dist, tab_cat, tab_corr, tab_explore = st.tabs(
        ["📈 Distributions", "🧩 Categorical Breakdown", "🔥 Correlation", "🔎 Data Explorer"]
    )

    # ---- Distributions tab ----
    with tab_dist:
        if numeric_cols:
            col1, col2 = st.columns([1, 3])
            with col1:
                selected_num = st.selectbox("Numeric column", numeric_cols, key="dist_col")
                show_kde = st.checkbox("Show as smoothed curve", value=True)
            with col2:
                if show_kde:
                    fig = px.histogram(
                        df, x=selected_num, marginal="box",
                        nbins=40, opacity=0.85,
                        title=f"Distribution of {selected_num}"
                    )
                else:
                    fig = px.histogram(df, x=selected_num, nbins=40, title=f"Distribution of {selected_num}")
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns available to plot.")

    # ---- Categorical tab ----
    with tab_cat:
        if categorical_cols:
            col1, col2 = st.columns([1, 3])
            with col1:
                selected_cat = st.selectbox("Categorical column", categorical_cols, key="cat_col")
                top_n = st.slider("Top N categories", min_value=5, max_value=30, value=10)
            with col2:
                counts = df[selected_cat].value_counts().nlargest(top_n).reset_index()
                counts.columns = [selected_cat, "count"]
                fig = px.bar(
                    counts, x="count", y=selected_cat, orientation="h",
                    title=f"Top {top_n} values in {selected_cat}",
                    color="count", color_continuous_scale="Blues"
                )
                fig.update_layout(height=450, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical columns available to plot.")

    # ---- Correlation tab ----
    with tab_corr:
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.columns,
                colorscale="RdBu", zmid=0, text=np.round(corr.values, 2),
                texttemplate="%{text}"
            ))
            fig.update_layout(title="Correlation Heatmap", height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for a correlation heatmap.")

    # ---- Data explorer tab ----
    with tab_explore:
        filter_cols = st.multiselect("Filter by column(s)", df.columns.tolist())
        filtered = df.copy()
        for col in filter_cols:
            if df[col].dtype == object:
                options = sorted(df[col].dropna().unique().tolist())
                chosen = st.multiselect(f"Values for {col}", options, key=f"filter_{col}")
                if chosen:
                    filtered = filtered[filtered[col].isin(chosen)]
            else:
                min_v, max_v = float(df[col].min()), float(df[col].max())
                rng = st.slider(f"Range for {col}", min_v, max_v, (min_v, max_v), key=f"range_{col}")
                filtered = filtered[(filtered[col] >= rng[0]) & (filtered[col] <= rng[1])]
        st.write(f"Showing {filtered.shape[0]:,} of {df.shape[0]:,} rows")
        st.dataframe(filtered, height=400, use_container_width=True)


# ----------------------------------------------------------------------------
# Main flow
# ----------------------------------------------------------------------------
if uploaded_file is not None:
    try:
        # Generate the file_id and re-save to disk only ONCE per uploaded
        # file, not on every widget interaction / Streamlit rerun. Without
        # this, every tab click or slider drag in the dashboard re-writes
        # the CSV to disk and generates a fresh UUID for no reason.
        if "file_id" not in st.session_state or st.session_state.get("uploaded_name") != uploaded_file.name:
            st.session_state["file_id"] = str(uuid.uuid4())
            st.session_state["uploaded_name"] = uploaded_file.name
            st.session_state.pop("df_cleaned", None)  # reset cleaned state for new file
            st.session_state.pop("ai_ran", None)
            st.session_state.pop("summary_response", None)
            st.session_state.pop("preprocessing_code", None)

        file_id = st.session_state["file_id"]
        dataset_path = os.path.join(TEMP_DIR, f"dataset_{file_id}.csv")
        cleaned_path = os.path.join(TEMP_DIR, f"cleaned_{file_id}.csv")

        if "df_raw" not in st.session_state:
            df = pd.read_csv(uploaded_file)
            df.to_csv(dataset_path, index=False)
            st.session_state["df_raw"] = df
        df = st.session_state["df_raw"]
        st.success("✅ Dataset uploaded successfully!")

        # ---- Dashboard renders IMMEDIATELY on the raw data ----
        # No LLM call is needed for this, so the user isn't stuck waiting
        # 30-120+ seconds staring at a blank page before seeing anything.
        st.session_state.setdefault("df_cleaned", df)
        render_dashboard(st.session_state["df_cleaned"])

        st.divider()

        # ---- AI summary + preprocessing run on demand, not automatically ----
        # This is the part that actually calls the local LLM and is slow.
        # Gating it behind a button means the user sees the dashboard first,
        # and only pays the LLM latency cost if they actually want it.
        run_ai = st.button("🤖 Run AI Summary & Preprocessing")

        if run_ai:
            with st.spinner("Asking the local model for a dataset summary..."):
                summary_response = get_llama_response(generate_summary_prompt(df))
            st.session_state["summary_response"] = summary_response

            with st.spinner("Asking the local model to write preprocessing code..."):
                preprocessing_code_raw = get_llama_response(
                    generate_preprocessing_prompt(df, dataset_path, cleaned_path)
                )
            if preprocessing_code_raw:
                preprocessing_code = clean_llm_response(preprocessing_code_raw)
                df_cleaned = apply_generated_code(df, preprocessing_code, dataset_path, cleaned_path)
            else:
                preprocessing_code = None
                df_cleaned = df.copy()

            st.session_state["preprocessing_code"] = preprocessing_code
            df_cleaned.to_csv(cleaned_path, index=False)
            st.session_state["df_cleaned"] = df_cleaned
            st.session_state["ai_ran"] = True
            st.rerun()

        # Everything below reads from session_state instead of local
        # variables tied to the button click, so it stays visible on every
        # rerun afterward (e.g. clicking a dashboard tab or slider) instead
        # of vanishing the instant run_ai goes back to False.
        if st.session_state.get("ai_ran"):
            st.subheader("Dataset Summary")
            if st.session_state.get("summary_response"):
                st.text_area(
                    "Summary and Insights",
                    st.session_state["summary_response"],
                    height=200,
                    key="summary_display",
                )
            else:
                st.error("⚠️ Failed to generate dataset summary.")

            st.subheader("Preprocessing")
            if st.session_state.get("preprocessing_code"):
                with st.expander("View generated preprocessing code", expanded=False):
                    st.code(st.session_state["preprocessing_code"], language="python")
                st.success(f"✅ Cleaned dataset saved as {cleaned_path}. Dashboard above reflects the cleaned data — scroll up.")
            else:
                st.error("⚠️ Failed to generate preprocessing code. Using raw dataset instead.")

        st.subheader("Original vs Cleaned Dataset")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Original Dataset**")
            st.dataframe(df, height=300, use_container_width=True)
        with col2:
            st.write("**Cleaned Dataset**")
            st.dataframe(st.session_state["df_cleaned"], height=300, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Error processing dataset: {e}")

else:
    st.info("Please upload a CSV file to begin.")
