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
import json
from pathlib import Path
# Try relative import first
try:
    from .config import ANTHROPIC_API_KEY, LLM_MODEL
# Fall back to absolute import
except ImportError:
    from config import ANTHROPIC_API_KEY, LLM_MODEL

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

def evaluate_summaries():
    """Run RAGAS evaluation on our summaries"""
    # Load dataset
    dataset = load_evaluation_dataset()
    
    # Initialize evaluator LLM as in example
    evaluator_llm = LangchainLLMWrapper(
            ChatAnthropic(api_key=ANTHROPIC_API_KEY, model=LLM_MODEL)
        )
    
    # Configure timeout as in example
    custom_run_config = RunConfig(timeout=360)
    
    # Define metrics following example notebook exactly
    metrics = [
        LLMContextRecall(),
        Faithfulness(),
        FactualCorrectness(),
        ResponseRelevancy(),
        ContextEntityRecall(),
        NoiseSensitivity()
    ]
    
    # Run evaluation with same parameters as example
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        run_config=custom_run_config
    )
    return results

def main():
    results = evaluate_summaries()
    
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