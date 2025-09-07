import streamlit as st
import pandas as pd
import os
import io
import matplotlib.pyplot as plt
import tempfile
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS

# This import is necessary for the app to function
try:
    from advanced_rag import (
        run_rag_workflow,
        llm,
        embedding_model,
        faiss_vector_store,
        compiled_graph,
        GraphState,
        visualization_generation_node,
        aggregate_analyses_node_v2,
        initialize_components
    )
except ImportError as e:
    st.error(f"Error importing advanced_rag.py: {e}. Please ensure it exists and is in the correct path.")
    st.stop()

# --- API Key Loading ---
# Load environment variables from a .env file for local development
load_dotenv()

try:
    groq_api_key = os.getenv("GROQ_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not all([groq_api_key, google_api_key, hf_api_key]):
        raise ValueError("One or more required environment variables are missing.")

    # Initialize components in advanced_rag.py with the loaded keys
    initialize_components(groq_api_key, google_api_key, hf_api_key)
    print("API keys loaded and components initialized successfully.")
except ValueError as e:
    st.error(f"Error: {e}. Please ensure all required environment variables are set.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during API key loading: {e}")
    st.stop()

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="RAG-Powered Data Analyzer")

col1, col2 = st.columns([3, 3])
with col1:
    st.image("assets/cute.png", width=330)
with col2:
    st.title("RAG-Powered Data Analyzer")
    st.markdown("Upload a CSV file, visualize the data, and ask questions using an LLM.")

# --- File Uploader and Data Display ---

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if 'file_info' not in st.session_state:
    st.session_state.file_info = None
if 'file_path' not in st.session_state:
    st.session_state.file_path = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'generated_plots' not in st.session_state:
    st.session_state.generated_plots = []
if 'visualization_analyses' not in st.session_state:
    st.session_state.visualization_analyses = []
if 'faiss_vector_store' not in st.session_state:
    st.session_state.faiss_vector_store = faiss_vector_store
if 'selected_plot_indices' not in st.session_state:
    st.session_state.selected_plot_indices = []

if uploaded_file is not None:
    current_file_info = (uploaded_file.name, uploaded_file.size)
    if st.session_state.file_info != current_file_info:
        st.session_state.file_info = current_file_info
        st.session_state.file_path = None
        st.session_state.df = None
        st.session_state.generated_plots = []
        st.session_state.messages = []
        st.session_state.visualization_analyses = []
        st.session_state.selected_plot_indices = []

        if embedding_model:
            try:
                st.session_state.faiss_vector_store = FAISS.from_texts(["initial load"], embedding_model)
                print("FAISS index reset for new file.")
            except Exception as e:
                print(f"Error resetting FAISS index: {e}")
                st.session_state.faiss_vector_store = None

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            st.session_state.file_path = tmp_file.name

        st.success(f"File '{uploaded_file.name}' uploaded successfully. Loading data...")

        try:
            st.session_state.df = pd.read_csv(st.session_state.file_path)
            st.session_state.df.index = st.session_state.df.index + 1

            st.subheader("Data Preview")
            st.dataframe(st.session_state.df.head())

            if not st.session_state.generated_plots or not st.session_state.visualization_analyses:
                st.info("Generating and analyzing visualizations...")
                if llm is None or embedding_model is None:
                    st.warning("LLM or Embedding model not initialized. Visualization analysis and querying may fail.")
                else:
                    try:
                        viz_state = GraphState(df=st.session_state.df, generated_plots=[], visualization_analyses=[],
                                               user_query="", initial_answer=None, reflection_output=None,
                                               file_path=st.session_state.file_path)
                        viz_result = visualization_generation_node(viz_state)
                        st.session_state.generated_plots = viz_result.get("generated_plots", [])
                        st.success(f"Generated {len(st.session_state.generated_plots)} visualizations.")

                        analysis_state = GraphState(df=st.session_state.df,
                                                    generated_plots=st.session_state.generated_plots,
                                                    visualization_analyses=[], user_query="", initial_answer=None,
                                                    reflection_output=None, file_path=st.session_state.file_path)
                        analysis_result = aggregate_analyses_node_v2(analysis_state)
                        st.session_state.visualization_analyses = analysis_result.get("visualization_analyses", [])
                        st.success(f"Generated {len(st.session_state.visualization_analyses)} visualization analyses.")

                    except Exception as e:
                        st.error(f"Error generating or analyzing visualizations: {e}")
                        st.session_state.generated_plots = []
                        st.session_state.visualization_analyses = []

        except Exception as e:
            st.error(f"Error loading or displaying data: {e}")
            st.session_state.df = None
            st.session_state.file_path = None
            st.session_state.generated_plots = []
            st.session_state.messages = []
            st.session_state.visualization_analyses = []
            st.session_state.selected_plot_indices = []

# --- Display Plots and Selection ---
if st.session_state.generated_plots:
    st.sidebar.subheader("Visualizations")
    st.sidebar.markdown("Select plots to include in analysis context:")
    current_selected_indices = []
    for i, plot_data in enumerate(st.session_state.generated_plots):
        checkbox_key = f"plot_checkbox_{i}"
        fig = plot_data['figure']
        metadata = plot_data.get('metadata', {})
        plot_title = metadata.get('title') or f"Plot {i + 1}"
        container = st.sidebar.container()
        with container:
            selection_indicator = "✅ " if i in st.session_state.selected_plot_indices else "⬜ "
            st.sidebar.markdown(f"{selection_indicator} **{plot_title}**")
            is_selected = st.checkbox(
                f"Include in context",
                value=i in st.session_state.selected_plot_indices,
                key=checkbox_key
            )
            if is_selected:
                current_selected_indices.append(i)
            try:
                st.pyplot(fig)
            except Exception as e:
                st.sidebar.warning(f"Could not display plot {i + 1}: {e}")
            finally:
                plt.close(fig)
    st.session_state.selected_plot_indices = current_selected_indices
    st.sidebar.markdown(f"Selected plots for analysis: {len(st.session_state.selected_plot_indices)}")
elif st.session_state.file_path is None:
    st.info("Please upload a CSV file to get started.")

# --- Chat Interface ---
st.subheader("Ask Questions About the Data")

if 'messages' not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Enter your question here...")
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    if st.session_state.file_path is None or st.session_state.df is None:
        response = "Please upload a CSV file first."
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
    else:
        with st.spinner("Thinking..."):
            try:
                if llm is None or embedding_model is None:
                    response_content = "Error: Core components (LLM or Embedding Model) are not initialized. Please set your API keys."
                    st.session_state.messages.append({"role": "assistant", "content": response_content})
                    with st.chat_message("assistant"):
                        st.markdown(response_content)
                else:
                    selected_analyses = [
                        st.session_state.visualization_analyses[i]
                        for i in st.session_state.selected_plot_indices
                        if 0 <= i < len(st.session_state.visualization_analyses)
                    ]
                    context_feedback = ""
                    if not st.session_state.visualization_analyses:
                        context_feedback = "No visualization analyses were generated. The AI will answer based on general knowledge."
                        print("No visualization analyses available to provide context for the query.")
                    elif not selected_analyses:
                        context_feedback = "No specific plots were selected. The AI is attempting retrieval from all generated visualization analyses."
                    else:
                        selected_analyses_titles = [
                            st.session_state.generated_plots[i]['metadata'].get('title', f"Plot {i + 1}")
                            for i in st.session_state.selected_plot_indices
                            if 0 <= i < len(st.session_state.generated_plots)
                        ]
                        if selected_analyses_titles:
                            context_feedback = f"Using analyses from the following selected plots as context: {', '.join(selected_analyses_titles)}"
                        else:
                            context_feedback = "Using analyses from selected plots as context (titles not available)."
                    with st.chat_message("assistant"):
                        st.info(context_feedback)
                    workflow_result = run_rag_workflow(st.session_state.file_path, user_query, selected_analyses)
                    ai_response = workflow_result.get("initial_answer", "Could not generate an answer.")
                    reflection = workflow_result.get("reflection_output", "No reflection performed.")
                    response_content = f"**AI Response:**\n{ai_response}\n\n**Reflection:**\n{reflection}"
                    st.session_state.messages.append({"role": "assistant", "content": response_content})
                    with st.chat_message("assistant"):
                        st.markdown(response_content)
            except Exception as e:
                error_message = f"An error occurred during workflow execution: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                with st.chat_message("assistant"):
                    st.markdown(error_message)
