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
# Fall back to absolute import
except ImportError:
    from config import get_anthropic_api_key, LLM_MODEL


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


def load_evaluation_dataset():
    """Load and format the evaluation dataset for RAGAS"""
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
            
            # Original text becomes both context and retrieved_contexts
            # since we're evaluating golden summaries that were generated from full context
            eval_data["context"].append(example["original_text"])
            eval_data["retrieved_contexts"].append([example["original_text"]])  # List of contexts used
            
            eval_data["answer"].append(example["golden_summary"])
            eval_data["ground_truth"].append(example["golden_summary"])
    
    # Convert to HuggingFace dataset
    return Dataset.from_dict(eval_data)

def evaluate_ragas():
    """Run RAGAS evaluation on our summaries"""
    # Load dataset
    dataset = load_evaluation_dataset()
    
    # Initialize evaluator LLM as in example
    evaluator_llm = LangchainLLMWrapper(
        ChatAnthropic(api_key=get_anthropic_api_key(), model=LLM_MODEL)
    )
    
    # Configure timeout as in example
    custom_run_config = RunConfig(timeout=360)
    
    # you can init the metric with the evaluator llm
    privacy_binary = AspectCritic(
        name="privacy_binary",
        definition="""Is the summary private? Check if any of the following confidential information exists:
        1. Institution names
        2. People's names or other proper nouns
        3. Specific dates
        4. Location details
        5. Any other information you judge to be highly confidential that appears to have accidentally been included in the summary
        
        Return 'no' if ANY confidential information is found, 'yes' if the summary properly anonymizes all details.""",
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
        privacy_binary,
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