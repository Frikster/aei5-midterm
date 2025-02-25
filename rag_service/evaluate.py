from ragas import evaluate
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    FactualCorrectness,
    ResponseRelevancy,
    ContextEntityRecall,
    NoiseSensitivity
)
from ragas import RunConfig
from datasets import Dataset
from langchain_anthropic import ChatAnthropic
from ragas.llms import LangchainLLMWrapper
# from langsmith import Client
# from langsmith.evaluation import LangChainStringEvaluator, evaluate
import json
from pathlib import Path
from typing import Dict
from ragas.metrics import SingleTurnMetric
from typing import Set
from dataclasses import dataclass, field
from ragas.metrics.base import MetricType
import re
from ragas.metrics import AspectCritic
# Try relative import first
try:
    from .config import get_anthropic_api_key, LLM_MODEL
    from .rag_pipeline import GrantSummaryPipeline
# Fall back to absolute import
except ImportError:
    from config import get_anthropic_api_key, LLM_MODEL
    from rag_pipeline import GrantSummaryPipeline


@dataclass
class SummaryStructureMetric(SingleTurnMetric):
    """Evaluates if the summary follows the two-sentence structure and starts with 'This project'"""
    
    _required_columns: Dict[MetricType, Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"response"}
        }
    )
    
    def init(self, run_config: RunConfig):
        """Initialize the metric with run configuration"""
        pass  # No initialization needed for this metric
    
    async def _single_turn_ascore(self, sample, callbacks=None) -> float:
        """Score a single sample asynchronously"""
        response = sample.response
        
        # Check if starts with "This project"
        starts_correctly = response.strip().startswith("This project")
        
        # Count sentences
        sentences = response.split('.')
        sentence_count = sum(1 for s in sentences if s.strip())
        correct_length = sentence_count == 2
        
        # Return score (1.0 if both conditions met, 0.5 if one met, 0.0 if none met)
        if starts_correctly and correct_length:
            return 1.0
        elif starts_correctly or correct_length:
            return 0.5
        return 0.0

@dataclass
class StyleGuidelineMetric(SingleTurnMetric):
    """Evaluates if the summary follows style guidelines including active voice and length"""
    
    _required_columns: Dict[MetricType, Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"response"}
        }
    )
    
    def init(self, run_config: RunConfig):
        """Initialize the metric with run configuration"""
        pass  # No initialization needed for this metric
    
    async def _single_turn_ascore(self, sample, callbacks=None) -> float:
        """Score a single sample asynchronously"""
        response = sample.response
        score = 1.0
        
        # Check length (between 30 and 100 words)
        word_count = len(response.split())
        if word_count < 30 or word_count > 100:
            score -= 0.3
        
        # Check for passive voice indicators
        passive_indicators = [
            r'\bis\s+\w+ed\b',
            r'\bwas\s+\w+ed\b',
            r'\bare\s+\w+ed\b',
            r'\bwere\s+\w+ed\b',
            r'\bbeing\s+\w+ed\b',
            r'\bbeen\s+\w+ed\b',
        ]
        
        for pattern in passive_indicators:
            if re.search(pattern, response, re.IGNORECASE):
                score -= 0.1
                break
        
        return max(0.0, score)  # Ensure score doesn't go below 0


def load_evaluation_dataset(rag_pipeline: GrantSummaryPipeline, skip_privacy=False) -> Dataset:
    """Load and format the evaluation dataset for RAGAS"""
    try:
        with open("evaluation_dataset-golden.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        with open("evaluation_dataset.json", "r") as f:
            data = json.load(f)
    
    # Format data for RAGAS evaluation
    eval_data = {
        "question": [],  # In our case, this is always the same summary request
        "context": [],   # Original grant text
        "answer": [],    # Generated summary
        "ground_truth": [], # Golden summary
        "retrieved_contexts": []  # Required by RAGAS metrics
    }
    
    for example in data:
        if "Insufficient context" not in example["golden_summary"]:
            eval_data["question"].append("Summarize this grant application in two sentences.")
            # Full context is the original text
            eval_data["context"].append(example["original_text"])
            # Retrieved contexts should be what the RAG system actually used
            eval_data["retrieved_contexts"].append(rag_pipeline.get_relevant_chunks(example["grant_id"]))
            # Answer should be the generated summary
            eval_data["answer"].append(rag_pipeline.run_agent(context=example["original_text"], skip_privacy=skip_privacy))
            # Ground truth is our golden summary
            eval_data["ground_truth"].append(example["golden_summary"])
    
    return Dataset.from_dict(eval_data)

def evaluate_ragas(rag_pipeline: GrantSummaryPipeline, skip_privacy= False):
    """Run RAGAS evaluation on our summaries"""
    dataset = load_evaluation_dataset(rag_pipeline, skip_privacy)
    
    # Initialize evaluator LLM as in example
    evaluator_llm = LangchainLLMWrapper(
        ChatAnthropic(api_key=get_anthropic_api_key(), model=LLM_MODEL)
    )
    
    # Configure timeout as in example
    custom_run_config = RunConfig(timeout=360)
    
    # you can init the metric with the evaluator llm
    institution_leak = AspectCritic(
        name="institution_leak",
        definition="""Does the response contain any institution names?""",
        llm=evaluator_llm,
    )
    name_leak = AspectCritic(
        name="name_leak",
        definition="""Does the response contain any people's names?""",
        llm=evaluator_llm,
    )
    date_leak = AspectCritic(
        name="date_leak",
        definition="""Does the response contain any specific dates?""",
        llm=evaluator_llm,
    )
    address_leak = AspectCritic(
        name="address_leak",
        definition="""Does the response contain specific location details or addresses?""",
        llm=evaluator_llm,
    )
    other_leak = AspectCritic(
        name="other_leak",
        definition="""Does the response contain information (NOT including names, addresses, dates, or institution names) you judge to be highly confidential that appears to have accidentally been included in the response""",
        llm=evaluator_llm,
    )
    
    # Define metrics following example notebook exactly
    metrics = [
        LLMContextRecall(),
        Faithfulness(),
        FactualCorrectness(),
        ResponseRelevancy(),
        ContextEntityRecall(),
        NoiseSensitivity(),
        # custom metrics
        institution_leak,
        name_leak,
        date_leak,
        address_leak,
        other_leak,
        SummaryStructureMetric(),
        StyleGuidelineMetric()
    ]
    
    # Run evaluation with same parameters as example
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        run_config=custom_run_config
    )
    return results

# def create_langsmith_dataset(ragas_dataset: Dataset) -> str:
#     """Convert RAGAS dataset to LangSmith format and return dataset ID"""
#     client = Client()
    
#     # Create a new dataset
#     dataset_name = f"Grant Summary Evaluation {uuid4().hex[:8]}"
#     langsmith_dataset = client.create_dataset(
#         dataset_name=dataset_name,
#         description="Grant summary evaluation dataset"
#     )
    
#     # Add examples to the dataset
#     for example in ragas_dataset:
#         client.create_example(
#             inputs={
#                 "question": example["question"],
#                 "context": example["context"]
#             },
#             outputs={
#                 "answer": example["answer"]
#             },
#             metadata={
#                 "ground_truth": example["ground_truth"],
#                 "retrieved_contexts": example["retrieved_contexts"]
#             },
#             dataset_id=langsmith_dataset.id
#         )
    
#     return dataset_name


# def evaluate_custom(pipeline) -> Dict[str, float]:
#     """Run custom evaluations on summaries"""
#     dataset = load_evaluation_dataset()
#     dataset_name = create_langsmith_dataset(dataset)
#     evaluator_llm = ChatAnthropic(
#         api_key=get_anthropic_api_key(),
#         model=LLM_MODEL
#     )
    
#     # Initialize custom evaluators
#     custom_evaluators = [
#         LangChainStringEvaluator(
#             "criteria",
#             config={
#                 "criteria": {
#                     "privacy": (
#                         "Does this summary avoid revealing private information like:"
#                         "\n- Institution names"
#                         "\n- People names"
#                         "\n- Specific dates"
#                         "\n- Location details"
#                     )
#                 },
#                 "llm": evaluator_llm
#             }
#         ),
#         LangChainStringEvaluator(
#             "criteria",
#             config={
#                 "criteria": {
#                     "first_sentence": (
#                         "Does the first sentence begin with 'This project', state the goal,"
#                         " and include 2-3 methodological approaches?"
#                     ),
#                     "second_sentence": (
#                         "Does the second sentence list concrete outputs with quantitative"
#                         " metrics and organizational achievements?"
#                     )
#                 },
#                 "llm": evaluator_llm
#             }
#         ),
#         LangChainStringEvaluator(
#             "criteria",
#             config={
#                 "criteria": {
#                     "style": (
#                         "Does this summary follow style guidelines:"
#                         "\n- Uses active voice"
#                         "\n- Avoids jargon unless field-standard"
#                         "\n- Keeps to ~100 words"
#                         "\n- Focuses on measurable outcomes"
#                     )
#                 },
#                 "llm": evaluator_llm
#             }
#         ),
#         LangChainStringEvaluator(
#             "criteria",
#             config={
#                 "criteria": {
#                     "no_comments": (
#                         "Is this summary free of LLM comments/notes like 'Note:', 'I think',"
#                         " or text in brackets/parentheses?"
#                     )
#                 },
#                 "llm": evaluator_llm
#             }
#         )
#     ]
    
#     # Run custom evaluations using LangSmith evaluate
#     print("Evaluating on Custom Metrics")
#     custom_results = evaluate(
#         pipeline.run_agent,  # The function to evaluate
#         data=dataset_name,     # Your evaluation dataset
#         evaluators=custom_evaluators,
#         metadata={"revision_id": "default_chain"}
#     )
#     return custom_results

def main():
    results = evaluate_ragas()
    
    # Convert results to pandas DataFrame
    results_df = results.to_pandas()
    
    # Save results using pandas to_json
    output_path = Path("ragas_evaluation_results.json")
    results_df.to_json(output_path, orient="records", indent=2)
    print(f"\nSaved detailed results to {output_path}")
    
    # Print summary scores
    print("\nRAGAS Evaluation Results:")
    print("------------------------")
    for metric_name, metric_value in results._repr_dict.items():
        print(f"{metric_name}: {metric_value:.3f}")

if __name__ == "__main__":
    main() 