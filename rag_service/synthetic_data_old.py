from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer, MultiHopAbstractQuerySynthesizer, MultiHopSpecificQuerySynthesizer
from ragas.testset.transforms import default_transforms, apply_transforms

# from ragas.testset.synthesizers.single_hop import (
#     SingleHopQuerySynthesizer,
# )
# from ragas.testset.synthesizers.multi_hop import (
#     MultiHopQuerySynthesizer
# )

from config import ANTHROPIC_API_KEY, LLM_MODEL, EMBEDDING_MODEL
from data_loader import load_grants_from_airtable
from typing import List, Dict
import json
from tqdm import tqdm
from pathlib import Path
import random

class GrantTestsetGenerator:
    def __init__(self):
        # Initialize models
        self.generator_llm = LangchainLLMWrapper(
            ChatAnthropic(api_key=ANTHROPIC_API_KEY, model=LLM_MODEL)
        )
        self.generator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model=EMBEDDING_MODEL)
        )
        
        # Initialize knowledge graph
        self.kg = KnowledgeGraph()

    # TODO: remove since not needed?
    def build_knowledge_graph(self, documents):
        """Build knowledge graph from grant documents"""
        print("Creating knowledge graph from grants...")
        
        for doc in documents:
            # Add each grant as a chunk node
            # Include grant_id in properties to maintain document boundaries
            self.kg.nodes.append(
                Node(
                    type=NodeType.DOCUMENT,
                    properties={
                        "page_content": doc.page_content,
                        "document_metadata": doc.metadata,
                        "grant_id": doc.metadata["grant_id"]
                    }
                )
            )
        print(f"Added {len(self.kg.nodes)} nodes to knowledge graph")

        # Apply default transformations while preserving grant boundaries
        apply_transforms(self.kg, default_transforms(
            documents=documents,
            llm=self.generator_llm,
            embedding_model=self.generator_embeddings))

    def generate_test_data(self, documents, num_examples=20):
        """Generate synthetic test data"""
        print(f"Generating {num_examples} synthetic test examples...")

        # Configure generator for single-grant focus
        generator = TestsetGenerator(
            llm=self.generator_llm,
            embedding_model=self.generator_embeddings,
            # No need for knowledge graph - just use documents directly
        )

        # Use only SingleHopSpecificQuerySynthesizer since we're testing
        # specific information extraction from individual grants
        query_distribution = [
            (SingleHopSpecificQuerySynthesizer(llm=self.generator_llm), 1.0),
        ]

        # Generate test dataset
        testset = generator.generate_with_langchain_docs(
            documents=documents,
            testset_size=num_examples,
            query_distribution=query_distribution
        )
        
        return testset

def main():
    # Load grants
    print("Loading documents from Airtable...")
    documents = load_grants_from_airtable()
    print(f"Loaded {len(documents)} documents from Airtable")
    
    # Sample 50 grants for testing
    sample_docs = random.sample(documents, 3)
    
    # Generate synthetic data
    generator = GrantTestsetGenerator()
    dataset = generator.generate_test_data(sample_docs, num_examples=3)
    
    # Save dataset
    dataset.save("grant_synthetic_dataset.json")
    print("Saved synthetic dataset")

if __name__ == "__main__":
    main()