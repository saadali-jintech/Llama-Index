import streamlit as st
import nest_asyncio
nest_asyncio.apply()

from llama_index.llms.ollama import Ollama
from llama_index.core.evaluation.faithfulness import FaithfulnessEvaluator
from llama_index.core.evaluation.answer_relevancy import AnswerRelevancyEvaluator
from llama_index.core.evaluation.correctness import CorrectnessEvaluator
from llama_index.core.base.response.schema import Response

# Helper to fix scoring inconsistencies
def safe_score(score, feedback=None):
    if score is not None and score > 0:
        return score
    if feedback:
        if "2/2" in feedback:
            return 1.0
        elif "1/2" in feedback:
            return 0.5
        elif "0/2" in feedback:
            return 0.0
    return 0.0  # default fallback

# Initialize LLM
llm = Ollama(model="llama3.1")

# Streamlit UI setup
st.set_page_config(page_title="LLM Query & Evaluation", layout="centered")
st.title("🧠 LLM Response Evaluator")

# User query input
query = st.text_input("Enter your query:", value="What is the capital of France?")

# Run the LLM and evaluators
if st.button("Run Query"):
    with st.spinner("Querying the LLM..."):
        raw_response = llm.complete(query)
        response_text = raw_response.text.strip()

        # Wrap in Response object for evaluators
        wrapped_response = Response(response=response_text)

        # Show response to user
        st.subheader("📝 LLM Response")
        st.write(response_text)

    with st.spinner("Evaluating response..."):
        try:
            # Initialize evaluators
            faithfulness_evaluator = FaithfulnessEvaluator(llm=llm, raise_error=True)   
            relevancy_evaluator = AnswerRelevancyEvaluator(llm=llm)
            correctness_evaluator = CorrectnessEvaluator(llm=llm)

            # Evaluate
            reference_answer = "Paris"
            faithfulness_result = faithfulness_evaluator.evaluate_response(
                query=query, response=wrapped_response, reference=reference_answer
            )
            relevancy_result = relevancy_evaluator.evaluate_response(
                query=query, response=wrapped_response, reference=reference_answer
            )

            correctness_result = correctness_evaluator.evaluate_response(
                query=query, response=wrapped_response, reference=None
            )

            # Normalize correctness (handle both /4 and already-normalized cases)
            raw_correctness_score = correctness_result.score or 0
            correctness_score = raw_correctness_score / 4 if raw_correctness_score > 1 else raw_correctness_score

            # Apply fallback logic for faithfulness and relevancy
            faithfulness_score = safe_score(faithfulness_result.score, faithfulness_result.feedback)
            relevancy_score = safe_score(relevancy_result.score, relevancy_result.feedback)

            # Show evaluation
            st.subheader("📊 Evaluation Metrics")

            st.markdown(f"**Faithfulness Score:** {faithfulness_score:.2f}")
            st.markdown(f"> {faithfulness_result.feedback or 'No feedback provided.'}")

            st.markdown(f"**Relevancy Score:** {relevancy_score:.2f}")
            st.markdown(f"> {relevancy_result.feedback or 'No feedback provided.'}")

            st.markdown(f"**Correctness Score:** {correctness_score:.2f}")
            st.markdown(f"> {correctness_result.feedback or 'No feedback provided.'}")

        except Exception as e:
            st.error(f"Evaluation failed: {e}")
