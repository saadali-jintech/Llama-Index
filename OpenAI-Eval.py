import os
import sys
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Response
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, AnswerRelevancyEvaluator, EvaluationResult

# --- 1. Setup and Configuration ---
print("--- 1. Setup and Configuration ---")
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY environment variable not set.")
    print("Please set it using: export OPENAI_API_KEY='YOUR_API_KEY' (Linux/macOS) or set OPENAI_API_KEY=YOUR_API_KEY (Windows)")
    sys.exit(1)

PDF_FILE_PATH = "data/sample_eval_report.pdf"
os.makedirs('data', exist_ok=True)

# --- 2. Load Data from PDF ---
print("\n--- 2. Load Data from PDF ---")
try:
    reader = SimpleDirectoryReader(input_files=[PDF_FILE_PATH])
    documents = reader.load_data()
    if not documents:
        print(f"Error: No documents were loaded from {PDF_FILE_PATH}.")
        sys.exit(1)
    print(f"Successfully loaded {len(documents)} document(s) from {PDF_FILE_PATH}")
    print(f"Content Snippet: {documents[0].text[:150]}...")
    print(f"Metadata: {documents[0].metadata}")
except Exception as e:
    print(f"Error loading PDF: {e}")
    sys.exit(1)

# --- 3. Create Vector Store Index ---
print("\n--- 3. Create Vector Store Index ---")
try:
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    print("VectorStoreIndex created successfully.")
except Exception as e:
    print(f"Error creating index: {e}")
    sys.exit(1)

# --- 4. Query the Index ---
print("\n--- 4. Query the Index ---")
query_str = "What is the purpose of Faithfulness evaluation in RAG systems?"
print(f"Query: {query_str}")
response = None
try:
    query_engine = index.as_query_engine()
    response = query_engine.query(query_str)

    print("\nGenerated Response:")
    print(response.response if response else "No response generated.")

    print("\nSource Nodes Retrieved:")
    if response and response.source_nodes:
        for i, source_node in enumerate(response.source_nodes):
            print(f"  Node {i+1} (Score: {source_node.score:.4f}): {source_node.node.get_content()[:100]}...")
    else:
        print("  No source nodes retrieved or response object is invalid.")

except Exception as e:
    print(f"Error querying index: {e}")
    if response is None:
        sys.exit(1)

# --- 5. Configure Evaluators ---
print("\n--- 5. Configure Evaluators ---")
faithfulness_evaluator = None
answer_relevancy_evaluator = None
try:
    eval_llm = OpenAI(model="gpt-3.5-turbo", temperature=0.0)
    faithfulness_evaluator = FaithfulnessEvaluator(llm=eval_llm)
    print("FaithfulnessEvaluator initialized.")
    answer_relevancy_evaluator = AnswerRelevancyEvaluator(llm=eval_llm)
    print("AnswerRelevancyEvaluator initialized.")
except Exception as e:
    print(f"Error initializing evaluators: {e}")
    print("Skipping evaluation due to initialization error.")

# --- 6. Execute Evaluation (if response and evaluators are valid) ---
print("\n--- 6. Execute Evaluation ---")
faithfulness_result = None
answer_relevancy_result = None

if response and faithfulness_evaluator:
    print("Running Faithfulness Evaluation...")
    try:
        faithfulness_result = faithfulness_evaluator.evaluate_response(response=response)
        print("Faithfulness Evaluation Complete.")
    except Exception as e:
        print(f"Error during Faithfulness evaluation: {e}")

if response and answer_relevancy_evaluator:
    print("\nRunning Answer Relevancy Evaluation...")
    try:
        answer_relevancy_result = answer_relevancy_evaluator.evaluate_response(query=query_str, response=response)
        print("Answer Relevancy Evaluation Complete.")
    except Exception as e:
        print(f"Error during Answer Relevancy evaluation: {e}")

# --- 7. Display Evaluation Results ---
print("\n--- 7. Display Evaluation Results ---")
print(f"Original Query: {query_str}")
print(f"Retrieved Response: {response.response if response else 'N/A'}")

if faithfulness_result:
    print("\nFaithfulness Result:")
    print(f"  Passing: {faithfulness_result.passing}")
    print(f"  Feedback: {faithfulness_result.feedback}")
elif faithfulness_evaluator:
    print("\nFaithfulness Result: Evaluation failed or was skipped.")

if answer_relevancy_result:
    print("\nAnswer Relevancy Result:")
    print(f"  Passing: {answer_relevancy_result.passing}")
    print(f"  Score: {answer_relevancy_result.score}")
    print(f"  Feedback: {answer_relevancy_result.feedback}")
elif answer_relevancy_evaluator:
    print("\nAnswer Relevancy Result: Evaluation failed or was skipped.")
