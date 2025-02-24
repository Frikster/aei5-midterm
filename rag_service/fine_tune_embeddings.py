from typing import List
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer, InputExample
from torch.utils.data import DataLoader
from tqdm import tqdm
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from datasets import Dataset
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, ResponseRelevancy, ContextEntityRecall, NoiseSensitivity
from ragas import evaluate
import wandb

# Try relative imports first
try:
    from .synthetic_data import EvaluationExample
    from .rag_pipeline import GrantSummaryPipeline
# Fall back to absolute imports
except ImportError:
    from synthetic_data import EvaluationExample
    from rag_pipeline import GrantSummaryPipeline

class EmbeddingTrainer:
    def __init__(self, base_model: str = "Snowflake/snowflake-arctic-embed-l"):
        self.model = SentenceTransformer(base_model)
        
    def prepare_training_data(self, eval_examples: List[EvaluationExample]) -> List[InputExample]:
        """Convert our evaluation examples into training pairs
        
        Note: we don't split into train/test/val sets because:
        1. We have a small, curated dataset of real grants
        2. We evaluate performance using RAGAS metrics directly
        3. We're fine-tuning for a very specific task (grant summaries only)
        """
        training_examples = []
        
        for example in tqdm(eval_examples, desc="Preparing training examples"):
            if "Insufficient context" not in example.golden_summary:
                # Create document-summary pairs
                # We want the embeddings of the original text to be closer to its summary
                training_examples.append(
                    InputExample(
                        texts=[
                            example.original_text,
                            example.golden_summary
                        ]
                    )
                )
        
        print(f"Created {len(training_examples)} training examples")
        return training_examples

    def create_evaluator(self, evaluation_examples):
        """Create evaluator for tracking embedding performance during training"""
        
        # Create corpus and queries from evaluation examples
        eval_corpus = {}  # Original grant texts
        eval_queries = {}  # Golden summaries
        relevant_docs = {}  # Maps each summary to its original text
        
        for i, example in enumerate(evaluation_examples):
            if "Insufficient context" not in example.golden_summary:
                # Original text becomes corpus document
                doc_id = f"doc_{i}"
                eval_corpus[doc_id] = example.original_text
                
                # Golden summary becomes query
                query_id = f"query_{i}"
                eval_queries[query_id] = example.golden_summary
                
                # Map summary to its source text
                relevant_docs[query_id] = [doc_id]
        
        # Create evaluator
        return InformationRetrievalEvaluator(
            eval_queries,
            eval_corpus, 
            relevant_docs,
            name='grant-retrieval-eval'
        )

    def train(self, training_examples: List[InputExample], output_path: str, evaluator) -> SentenceTransformer:
        """Fine-tune the embedding model"""
        train_dataloader = DataLoader(
            training_examples,
            shuffle=True,
            batch_size=10  # Small batch size since we have limited data
        )
        
        # Sets which vector dimensions are most to least important
        matryoshka_dimensions = [768, 512, 256, 128, 64]
        
        # Use contrastive loss to pull embeddings of related texts (grant and its summary) closer together
        # while pushing embeddings of unrelated texts (grant and other summaries) further apart.
        # This helps the model learn semantic similarity between grants and their summaries.
        inner_train_loss = MultipleNegativesRankingLoss(self.model)
        
        # Wrap with MatryoshkaLoss for hierarchical embeddings
        train_loss = MatryoshkaLoss(
            self.model, 
            inner_train_loss, 
            matryoshka_dims=matryoshka_dimensions
        )
        
        # Update model.fit to use evaluator
        EPOCHS = 3
        warmup_steps = int(len(train_dataloader) * EPOCHS * 0.1)
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=EPOCHS,
            warmup_steps=warmup_steps,
            output_path=output_path,
            show_progress_bar=True,
            evaluator=evaluator,
            evaluation_steps=50
        )
        return self.model
    
    def push_to_hf(self):
        hf_username = "Frikster42"
        self.model.push_to_hub(f"{hf_username}/grant-summary-embeddings-v1")

def main():
    wandb.init(mode="disabled")
    
    # Load existing evaluation dataset
    with Path("evaluation_dataset.json").open() as f:
        eval_data = [EvaluationExample(**example) for example in json.load(f)]
    print(f"Loaded {len(eval_data)} evaluation examples")

    # Train model
    trainer = EmbeddingTrainer()
    training_examples = trainer.prepare_training_data(eval_data)
    
    # Create evaluator for tracking during training
    evaluator = trainer.create_evaluator(eval_data)
    
    # Train with evaluator
    fine_tuned_model = trainer.train(
        training_examples, 
        "fine_tuned_embeddings",
        evaluator=evaluator
    )
    
    pipeline = GrantSummaryPipeline()
    
    # Run RAG Pipeline using OpenAI embeddings
    print("Running Anthropic model RAG pipeline...")
    base_openai_dataset = []
    for example in tqdm(eval_data, desc="Processing base model examples"):
        if "Insufficient context" not in example.golden_summary:
            response = pipeline.generate_summary(grant_id=example.grant_id)
            base_openai_dataset.append({
                "question": "Question not used yet", # TODO
                "context": example.original_text,
                "answer": response,
                "ground_truth": example.golden_summary,
                "retrieved_contexts": [example.original_text]  # For now, using full context
            })
    
    # Run Snowflake base model RAG pipeline before fine-tuning
    print("Running Snowflake base model RAG pipeline...")
    pipeline.embeddings = trainer.model
    base_snowflake_dataset = []
    for example in tqdm(eval_data, desc="Processing snowflake base model examples"):
        if "Insufficient context" not in example.golden_summary:
            response = pipeline.generate_summary(grant_id=example.grant_id)
            base_snowflake_dataset.append({
                "question": "Question not used yet", # TODO
                "context": example.original_text,
                "answer": response,
                "ground_truth": example.golden_summary,
                "retrieved_contexts": [example.original_text]  # For now, using full context
            })
    
    # Run fine-tuned model RAG pipeline
    print("\nRunning Snowflake fine-tuned model RAG pipeline...")
    pipeline.embeddings = fine_tuned_model
    fine_tuned_dataset = []
    for example in tqdm(eval_data, desc="Processing fine-tuned model examples"):
        if "Insufficient context" not in example.golden_summary:
            response = pipeline.generate_summary(grant_id=example.grant_id)
            fine_tuned_dataset.append({
                "question": "Question not used yet", # TODO
                "context": example.original_text,
                "answer": response,
                "ground_truth": example.golden_summary,
                "retrieved_contexts": [example.original_text]  # For now, using full context
            })
    
    
    
    # Convert to RAGAS evaluation datasets
    base_anthropic_evaluation_dataset = Dataset.from_list(base_openai_dataset)
    base_snowflake_evaluation_dataset = Dataset.from_list(base_snowflake_dataset)
    fine_tuned_evaluation_dataset = Dataset.from_list(fine_tuned_dataset)
    
    # Run RAGAS evaluations
    evaluator_llm = LangchainLLMWrapper(pipeline.llm)
    
    base_anthropic_results = evaluate(
        dataset=base_anthropic_evaluation_dataset,
        metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness(), 
                ResponseRelevancy(), ContextEntityRecall(), NoiseSensitivity()],
        llm=evaluator_llm
    )
    base_snowflake_results = evaluate(
        dataset=base_snowflake_evaluation_dataset,
        metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness(), 
                ResponseRelevancy(), ContextEntityRecall(), NoiseSensitivity()],
        llm=evaluator_llm
    )
    fine_tuned_results = evaluate(
        dataset=fine_tuned_evaluation_dataset,
        metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness(), 
                ResponseRelevancy(), ContextEntityRecall(), NoiseSensitivity()],
        llm=evaluator_llm
    )
    
    # Compare results
    print("\nRAGAS Metric Comparison:")
    print("------------------------")
    for metric in base_anthropic_results._repr_dict:
        anthropic = base_anthropic_results._repr_dict[metric]
        snowflake = base_snowflake_results._repr_dict[metric]
        fine = fine_tuned_results._repr_dict[metric]
        print(f"{metric}:")
        print(f"  Base Anthropic: {anthropic:.3f}")
        print(f"  Base Snowflake: {snowflake:.3f}")
        print(f"  Fine-tuned:     {fine:.3f}")
        print(f"  Snowflake {'better' if snowflake > anthropic else 'worse'} than Anthropic by: {abs((snowflake-anthropic)/anthropic)*100:.1f}%")
        print(f"  Improvement vs Anthropic: {((fine-anthropic)/anthropic)*100:.1f}%")
        print(f"  Improvement vs Snowflake: {((fine-snowflake)/snowflake)*100:.1f}%")

if __name__ == "__main__":
    main()