import streamlit as st
from llama_index.llms.ollama import Ollama
from llama_index.core.evaluation import CorrectnessEvaluator, FaithfulnessEvaluator, RelevancyEvaluator
import pandas as pd

# Initialize the Ollama Llama 3.1 model
ollama_model = Ollama(
    system_prompt="You are a helpful assistant. Answer the question to the best of your ability. If you don't know the answer, say 'I am not sure about that.'",
    model="llama3.1",
)

# Initialize evaluators
correctness_evaluator = CorrectnessEvaluator(llm=ollama_model)
faithfulness_evaluator = FaithfulnessEvaluator(llm=ollama_model)
relevancy_evaluator = RelevancyEvaluator(llm=ollama_model)

# Function to return dynamic context
def return_context(query):
    return "France, located in Western Europe, is a country known for its rich history, world-renowned cuisine, and global cultural influence. Its capital, Paris, is famous for landmarks like the Eiffel Tower and the Louvre Museum. With a population of around 67 million and French as the official language, France operates as a semi-presidential republic. It has played a key role in global events such as the French Revolution and both World Wars. France is also a leader in fashion, art, and philosophy, and is one of the top tourist destinations in the world. The country boasts a strong economy, excelling in industries such as aerospace, luxury goods, agriculture, and tourism, and is a founding member of the European Union and other major international organizations."

# Evaluate function
def evaluate_queries(queries):
    results = []

    for query in queries:
        context = return_context(query)
        context_list = [context]
        prompt = f"Query: {query}\nContext: {context}\nResponse:"
        response = ollama_model.complete(prompt)
        response_text = response.text if isinstance(response.text, str) else str(response.text)

        correctness_result = correctness_evaluator.evaluate(query, response_text)
        faithfulness_result = faithfulness_evaluator.evaluate(query, response_text, context_list)
        relevancy_result = relevancy_evaluator.evaluate(query, response_text, context_list)

        results.append({
            'Query': query,
            'Response': response_text,
            'Correctness passing': correctness_result.passing,
            'Correctness score': correctness_result.score,
            'Faithfulness passing': faithfulness_result.passing,
            'Faithfulness score': faithfulness_result.score,
            'Relevancy passing': relevancy_result.passing,
            'Relevancy score': relevancy_result.score,
        })

    return results

# Sample queries
queries = [
    "What is capital of France",
    "Do we need air to breathe?",
    "What are the top tourist attractions in France?"
]

result = evaluate_queries(queries)


df = pd.DataFrame(result)

for idx, row in df.iterrows():
    print(f"\n--- Evaluation Result {idx + 1} ---")
    print(f"Query               : {row['Query']}")
    print(f"Response            : {row['Response']}")
    print(f"Correctness Passing : {row['Correctness passing']}")
    print(f"Correctness Score   : {row['Correctness score']}")
    print(f"Faithfulness Passing: {row['Faithfulness passing']}")
    print(f"Faithfulness Score  : {row['Faithfulness score']}")
    print(f"Relevancy Passing   : {row['Relevancy passing']}")
    print(f"Relevancy Score     : {row['Relevancy score']}")

