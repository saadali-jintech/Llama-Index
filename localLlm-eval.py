import os
import logging
import sys
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    Response,
)
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    AnswerRelevancyEvaluator,
    EvaluationResult,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Configure logging
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG) # Uncomment for detailed logs
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# --- Configuration ---
PDF_FILE_PATH = os.path.join("data", "sample.pdf")
LLM_MODEL_NAME = "llama3.1"
EMBEDDING_MODEL_NAME = "nomic-embed-text" # Or "mxbai-embed-large", etc.
OLLAMA_BASE_URL = "http://localhost:11434"
REQUEST_TIMEOUT = 120.0 # Increase if needed for larger models/slower hardware
TOP_K = 2 # Number of source nodes to retrieve

# --- Assumption ---
# Ensure Ollama is running locally and the models
# (LLM_MODEL_NAME and EMBEDDING_MODEL_NAME) are available.
# Example: `ollama run llama3.1` and `ollama pull nomic-embed-text`

print("--- Setup Complete ---")

# === Load Data ===
print(f"Loading data from {PDF_FILE_PATH}...")
# Check if the data directory and PDF file exist
if not os.path.exists("data"):
    print("Error: 'data' directory not found.")
    sys.exit(1)
if not os.path.exists(PDF_FILE_PATH):
    print(f"Error: '{PDF_FILE_PATH}' not found.")
    sys.exit(1)

# SimpleDirectoryReader supports various file types including PDF [19, 20, 21]
# It requires 'pypdf' to be installed for PDF parsing.
try:
    reader = SimpleDirectoryReader(input_files=)
    documents = reader.load_data()
    print(f"Successfully loaded {len(documents)} document(s).")
except ImportError:
    print("Error: 'pypdf' is required to read PDF files. Please install it: pip install pypdf")
    sys.exit(1)
except Exception as e:
    print(f"Error loading PDF: {e}")
    sys.exit(1)

print("--- Data Loading Complete ---")

# === Configure Local Models via LlamaIndex Settings ===
print("Configuring LlamaIndex to use local Ollama models...")
# Configure the LLM (llama3.1 served by Ollama)
# [1, 2, 3, 4, 16, 22]
Settings.llm = Ollama(
    model=LLM_MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    request_timeout=REQUEST_TIMEOUT,
    # Optional: Add other parameters like temperature if needed
)

# Configure the Embedding Model (e.g., nomic-embed-text served by Ollama)
# [5, 6, 17, 18, 23]
Settings.embed_model = OllamaEmbedding(
    model_name=EMBEDDING_MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    # Optional: Add ollama_additional_kwargs if needed
)
print(f"Using LLM: {LLM_MODEL_NAME} and Embed Model: {EMBEDDING_MODEL_NAME} via Ollama.")
print("--- Model Configuration Complete ---")

# === Create Index ===
print("Creating vector store index...")
# The VectorStoreIndex embeds document chunks using the configured embed_model
# and stores them in a simple in-memory vector store by default.
# [4, 5, 23, 24]
index = VectorStoreIndex.from_documents(
    documents,
    # ServiceContext can also be used here, but Settings provides global config
    # service_context=ServiceContext.from_defaults(llm=Settings.llm, embed_model=Settings.embed_model)
)
print("--- Index Creation Complete ---")

# === Create Query Engine ===
print("Creating query engine...")
# The query engine uses the configured LLM for response synthesis.
# retrieve_nodes=True is implicitly handled by.query() returning a Response object.
# similarity_top_k controls how many relevant chunks are retrieved.
# [4, 5, 24, 25]
query_engine = index.as_query_engine(
    similarity_top_k=TOP_K,
    # Set streaming=False for easier handling in evaluation
    streaming=False,
)
print(f"Query engine created with similarity_top_k={TOP_K}.")
print("--- Query Engine Creation Complete ---")

# === Define Query ===
# IMPORTANT: Replace this with a query relevant to your 'sample.pdf' content!
query = "What is the main topic discussed in the document?"
print(f"\nSample Query: {query}")

# === Execute Query ===
print("Executing query...")
# Query the engine to get response and source nodes (context)
# The Response object contains both the text response and the source_nodes used.
response = query_engine.query(query)
print("--- Query Execution Complete ---")

# Print Response and Context
print("\nResponse:")
print(response) # Prints the textual response

print(f"\nRetrieved Source Nodes (Context used for response, top-{TOP_K}):")
for i, node in enumerate(response.source_nodes):
    print(f"--- Node {i+1} (Score: {node.score:.4f}) ---")
    # Truncate node text for display brevity
    node_text_preview = node.get_content(metadata_mode="all").strip()[:300] + "..."
    print(node_text_preview)
    # print(f"Metadata: {node.metadata}") # Uncomment to see metadata like file_path, page_label

# === Setup Evaluation ===
print("\nSetting up evaluators...")
# Use the same LLM ('llama3.1') for evaluation.
# Note: Using the same LLM for generation and evaluation can introduce bias.
# Consider using a different, potentially stronger, judge LLM for evaluation in production.

# Faithfulness checks if the response is supported by the retrieved context (source nodes)
# [25, 26]
faithfulness_evaluator = FaithfulnessEvaluator(llm=Settings.llm)
print("FaithfulnessEvaluator initialized.")

# AnswerRelevancy checks if the response is relevant to the query
# [26, 27, 28]
relevancy_evaluator = AnswerRelevancyEvaluator(llm=Settings.llm)
print("AnswerRelevancyEvaluator initialized.")

print("--- Evaluator Setup Complete ---")

# === Perform Evaluation ===
print("\nPerforming evaluation...")

# 1. Faithfulness Evaluation
print("Evaluating Faithfulness...")
faithfulness_result: EvaluationResult = faithfulness_evaluator.evaluate_response(
    query=query, response=response
)
print(f"Faithfulness Result: Passing={faithfulness_result.passing}")
if faithfulness_result.feedback:
    print(f"Faithfulness Feedback: {faithfulness_result.feedback}")

# 2. Answer Relevancy Evaluation
print("\nEvaluating Answer Relevancy...")
relevancy_result: EvaluationResult = relevancy_evaluator.evaluate_response(
    query=query, response=response
)
print(f"Answer Relevancy Result: Passing={relevancy_result.passing}")
if relevancy_result.feedback:
    print(f"Answer Relevancy Feedback: {relevancy_result.feedback}")
