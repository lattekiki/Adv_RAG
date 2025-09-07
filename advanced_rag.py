# advanced_rag.py
# This script defines the core LangGraph RAG workflow with multimodal capabilities.
# It is designed to be imported by a Streamlit application (e.g., app.py).

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import warnings
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
import google.generativeai as genai
from PIL import Image
import numpy as np

# Import necessary components from LangChain
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# Suppress the LangChain deprecation warning for HuggingFaceEmbeddings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Global Component Variables ---
# These will be initialized by the `initialize_components` function.
llm = None
multimodal_llm = None
embedding_model = None
faiss_vector_store = None


# Function to initialize all core components using provided API keys
def initialize_components(groq_api_key: str, google_api_key: str, hf_api_key: str):
    """Initializes all core components using provided API keys."""
    global llm, multimodal_llm, embedding_model, faiss_vector_store

    # Reset existing components to handle re-initialization
    llm, multimodal_llm, embedding_model, faiss_vector_store = None, None, None, None

    try:
        if groq_api_key:
            llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")
            print("Groq model initialized.")
        else:
            print("GROQ_API_KEY is missing. Skipping Groq initialization.")
    except Exception as e:
        print(f"Error initializing Groq model: {e}")
        llm = None

    try:
        if google_api_key:
            genai.configure(api_key=google_api_key)
            multimodal_llm = genai.GenerativeModel('gemini-1.5-flash-latest')
            print("Google multimodal model initialized.")
        else:
            print("GOOGLE_API_KEY is missing. Skipping Gemini initialization.")
    except Exception as e:
        print(f"Error initializing Google multimodal model: {e}")
        multimodal_llm = None

    try:
        if hf_api_key:
            embedding_model = HuggingFaceEndpointEmbeddings(
                model="sentence-transformers/all-MiniLM-L6-v2",
                huggingfacehub_api_token=hf_api_key
            )
            print("Embedding model initialized.")
        else:
            print("HUGGINGFACEHUB_API_TOKEN is missing. Skipping embedding model initialization.")
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        embedding_model = None

    if embedding_model:
        try:
            faiss_vector_store = FAISS.from_texts(["initial load"], embedding_model)
            print("FAISS vector store initialized.")
        except Exception as e:
            print(f"Error initializing FAISS vector store: {e}")
            faiss_vector_store = None

    print("All components initialized.")


# --- LangGraph State Definition ---
class GraphState(TypedDict):
    """Represents the state of our LangGraph workflow."""
    df: Any
    generated_plots: List[Dict[str, Any]]
    visualization_analyses: List[str]
    user_query: str
    initial_answer: str
    reflection_output: str
    file_path: str


# --- LangGraph Node Functions ---
def data_loading_node(state: GraphState) -> GraphState:
    print("---Executing Data Loading Node---")
    file_path = state.get('file_path')
    if not file_path or not os.path.exists(file_path):
        return {"df": None, "initial_answer": "Error: Invalid file path provided."}
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return {"df": df}
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return {"df": None, "initial_answer": f"Error loading data: {e}"}


def visualization_generation_node(state: GraphState) -> GraphState:
    print("---Executing Visualization Generation Node---")
    df = state.get('df')
    if df is None:
        return {"generated_plots": []}

    generated_plots = []
    numerical_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    if len(numerical_cols) >= 1:
        col = numerical_cols[0]
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f'Distribution of {col}')
        stats = df[col].describe().to_dict()
        skewness = df[col].skew()
        generated_plots.append({
            'figure': fig,
            'metadata': {'title': f'Distribution of {col}', 'stats': stats, 'skewness': skewness}
        })

    if len(categorical_cols) >= 1:
        col = categorical_cols[0]
        fig, ax = plt.subplots()
        sns.countplot(y=df[col], ax=ax)
        ax.set_title(f'Counts of {col}')
        value_counts = df[col].value_counts().to_dict()
        generated_plots.append({
            'figure': fig,
            'metadata': {'title': f'Counts of {col}', 'value_counts': value_counts}
        })

    if len(numerical_cols) >= 2:
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[numerical_cols[0]], y=df[numerical_cols[1]], ax=ax)
        ax.set_title(f'Relationship between {numerical_cols[0]} and {numerical_cols[1]}')
        corr = df[[numerical_cols[0], numerical_cols[1]]].corr().loc[numerical_cols[0], numerical_cols[1]]
        generated_plots.append({
            'figure': fig,
            'metadata': {'title': f'Relationship between {numerical_cols[0]} and {numerical_cols[1]}',
                         'correlation': corr}
        })

    print(f"Generated {len(generated_plots)} visualizations with metadata.")
    return {"generated_plots": generated_plots}


def analyze_visualization_multimodal(plot_data: Dict[str, Any]) -> str:
    """Analyzes a Matplotlib figure using a multimodal LLM, incorporating statistical metadata."""
    global multimodal_llm
    if multimodal_llm is None:
        return "Error: Multimodal LLM is not initialized for analysis."

    try:
        fig = plot_data['figure']
        metadata = plot_data.get('metadata', {})
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        image = Image.open(buffer)

        metadata_text = "\n".join(
            [f"- {key.replace('_', ' ').capitalize()}: {value}" for key, value in metadata.items() if key != 'title'])

        prompt_text = (
            "Analyze the following data visualization. "
            f"The title of the plot is '{metadata.get('title', 'Untitled')}'. "
            "Here are some key statistics and information about the data shown: "
            f"{metadata_text} "
            "Please describe any notable trends, patterns, "
            "key insights, or observations you see in the visualization, "
            "and **reference the provided statistical data** to support your analysis. "
            "What story does this chart tell about the data?"
        )
        response = multimodal_llm.generate_content([prompt_text, image])
        if hasattr(response, 'text'):
            return response.text
        else:
            return "No text content in the Gemini response."
    except Exception as e:
        return f"An error occurred during multimodal analysis: {e}"


def aggregate_analyses_node_v2(state: GraphState) -> GraphState:
    print("---Executing Aggregate Analyses Node (Multimodal)---")
    generated_plots_with_metadata = state.get('generated_plots')
    if not generated_plots_with_metadata:
        print("No plots available for analysis.")
        return {"visualization_analyses": []}

    visualization_analyses = []
    for i, plot_data in enumerate(generated_plots_with_metadata):
        print(f"Analyzing plot {i + 1}/{len(generated_plots_with_metadata)} using multimodal LLM...")
        analysis = analyze_visualization_multimodal(plot_data)

        if analysis and "Error:" not in analysis:
            plot_title = plot_data['metadata']['title']
            visualization_analyses.append(f"Analysis of '{plot_title}':\n{analysis}")
        else:
            print(f"Multimodal analysis for plot {i + 1} failed. {analysis}")

    print(f"Generated {len(visualization_analyses)} visualization analyses.")

    global faiss_vector_store, embedding_model
    if faiss_vector_store and embedding_model and visualization_analyses:
        print("Adding analyses to FAISS...")
        try:
            faiss_vector_store = FAISS.from_texts(visualization_analyses, embedding_model)
            print(f"Added {len(visualization_analyses)} analyses to FAISS.")
        except Exception as e:
            print(f"Error adding analyses to FAISS: {e}")
    elif not (faiss_vector_store and embedding_model):
        print("FAISS or Embedding model not initialized. Skipping FAISS update.")

    return {"visualization_analyses": visualization_analyses}


def user_query_processing_node_refined_v2(state: GraphState) -> GraphState:
    print("---Executing User Query Processing Node (Refined V2)---")
    user_query = state.get('user_query')
    selected_analyses = state.get('visualization_analyses', [])
    reflection_output = state.get('reflection_output')
    if not user_query:
        return {"initial_answer": "No query received."}

    context = ""
    context_source = ""

    if selected_analyses:
        print("Using selected visualization analyses from state as context.")
        context = "\n\n".join(selected_analyses)
        context_source = "analyses of the *selected* visualizations"
    elif faiss_vector_store and user_query:
        print("No selected analyses in state. Attempting FAISS retrieval from full index.")
        try:
            if faiss_vector_store.index.ntotal > 0:
                docs = faiss_vector_store.similarity_search(user_query, k=min(3, faiss_vector_store.index.ntotal))
                context = "\n\n".join([doc.page_content for doc in docs])
                context_source = "analyses of the visualizations (retrieved based on your query)"
            else:
                context_source = "general knowledge (FAISS index is empty)"
        except Exception as e:
            print(f"Error during FAISS retrieval: {e}")
            context_source = "general knowledge (FAISS retrieval failed)"
    else:
        context_source = "general knowledge (no relevant context available)"

    if reflection_output and "revised and improved answer" in reflection_output.lower():
        prompt_text = f"""You are an AI assistant that can answer questions about data based on provided {context_source} and a critical reflection on a previous attempt to answer the query.
Here are the {context_source}:
{context}
Here is a critical reflection on a previous attempt to answer the user's query "{user_query}":
{reflection_output}
Based on the above information AND the critical reflection, provide the final, best possible answer to the user's query.
"""
    else:
        prompt_text = f"""You are an AI assistant that can answer questions about data based on provided {context_source}.
Here are the {context_source}:
{context}
Based on the above analyses, answer the following question:
{user_query}
Provide a clear, concise, and accurate answer. Reference specific data points, trends, or observations from the analyses to support your answer.
"""
    global llm
    if llm:
        try:
            response = llm.invoke(prompt_text)
            llm_answer = response.content
            return {"initial_answer": llm_answer}
        except Exception as e:
            return {"initial_answer": f"Error: Unable to generate response. {e}"}
    return {"initial_answer": "Error: LLM is not initialized."}


def reflection_node_v2(state: GraphState) -> GraphState:
    print("---Executing Reflection Node---")
    user_query = state.get('user_query')
    initial_answer = state.get('initial_answer')
    if not initial_answer or "Error:" in initial_answer:
        return {"reflection_output": "Skipped reflection due to missing or error-prone initial answer."}

    reflection_prompt = f"""You are a helpful assistant that critically evaluates an initial answer to a user's data query.
User query: "{user_query}"
Initial answer: "{initial_answer}"
Your task is to:
1. Critique the initial answer.
2. Suggest improvements for a revised and improved answer.
3. Focus on whether the answer is accurate, comprehensive, and directly addresses the user's question using the provided context.
4. Provide a concluding statement that summarizes whether the answer is sufficient or needs improvement.
"""
    global llm
    if llm:
        try:
            response = llm.invoke(reflection_prompt)
            reflection_text = response.content
            return {"reflection_output": reflection_text}
        except Exception as e:
            return {"reflection_output": f"Error: LLM not initialized for reflection. {e}"}
    return {"reflection_output": "Error: LLM not initialized for reflection."}


# --- LangGraph Conditional Routing ---
def route_after_reflection(state: GraphState) -> str:
    print("---Executing Router After Reflection---")
    reflection_output = state.get('reflection_output')
    if reflection_output and (
            "revised and improved answer" in reflection_output.lower() or "needs improvement" in reflection_output.lower() or "shortcomings" in reflection_output.lower()):
        return "user_query_processing"
    else:
        return END


# --- LangGraph Setup ---
graph = StateGraph(GraphState)
graph.add_node("data_loading", data_loading_node)
graph.add_node("visualization_generation", visualization_generation_node)
graph.add_node("aggregate_analyses", aggregate_analyses_node_v2)
graph.add_node("user_query_processing", user_query_processing_node_refined_v2)
graph.add_node("reflection", reflection_node_v2)
graph.set_entry_point("data_loading")
graph.add_edge("data_loading", "visualization_generation")
graph.add_edge("visualization_generation", "aggregate_analyses")
graph.add_edge("aggregate_analyses", "user_query_processing")
graph.add_edge("user_query_processing", "reflection")
graph.add_conditional_edges("reflection", route_after_reflection,
                            {"user_query_processing": "user_query_processing", "__end__": END})
compiled_graph = graph.compile()
print("LangGraph has been set up and compiled.")


# --- Main Workflow Execution Function ---
def run_rag_workflow(file_path: str, user_query: str, selected_analyses: List[str]):
    """Runs the full RAG workflow using the compiled LangGraph."""
    print(f"\nStarting RAG workflow for query: {user_query}")
    if not os.path.exists(file_path):
        return {"initial_answer": f"Error: File not found at {file_path}"}

    global llm, multimodal_llm, embedding_model
    if llm is None or multimodal_llm is None or embedding_model is None:
        return {"initial_answer": "Error: Core LLM/Embedding components are not initialized. Please set your API keys."}

    initial_state = {
        "file_path": file_path,
        "user_query": user_query,
        "df": None,
        "generated_plots": [],
        "visualization_analyses": selected_analyses,
        "initial_answer": None,
        "reflection_output": None,
    }

    try:
        final_state = compiled_graph.invoke(initial_state)
        print("\n--- Workflow Completed ---")
        return {
            "initial_answer": final_state.get("initial_answer", "No answer generated."),
            "reflection_output": final_state.get("reflection_output", "No reflection performed."),
            "generated_plots": [item['figure'] for item in final_state.get("generated_plots", [])],
            "df": final_state.get("df", None)
        }
    except Exception as e:
        import traceback
        return {"initial_answer": f"An internal error occurred: {e}", "reflection_output": traceback.format_exc()}
