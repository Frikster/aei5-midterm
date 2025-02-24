from typing import List, Dict
import json
from pathlib import Path
from dataclasses import dataclass
from langchain.schema import Document
# Try relative imports first
try:
    from .rag_pipeline import GrantSummaryPipeline
    from .data_loader import load_grants_from_airtable
# Fall back to absolute imports
except ImportError:
    from rag_pipeline import GrantSummaryPipeline
    from data_loader import load_grants_from_airtable

@dataclass
class EvaluationExample:
    grant_id: str
    original_text: str
    golden_summary: str
    privacy_elements: List[str]  # List of sensitive elements that should be removed

class GrantEvaluationGenerator:
    def __init__(self):
        # Initialize the RAG pipeline to use its prompt
        self.rag_pipeline = GrantSummaryPipeline()
        self.llm = self.rag_pipeline.llm  # Use the same LLM instance
        
    def create_evaluation_dataset(self, grants: List[Dict], num_examples: int = 15) -> List[EvaluationExample]:
        """
        Create evaluation dataset by:
        1. Sampling diverse grants
        2. Generating golden summaries
        3. Identifying privacy elements
        """
        # Sample grants ensuring diversity
        selected_grants = self._sample_diverse_grants(grants, num_examples)
        
        evaluation_examples = []
        for grant in selected_grants:
            # Generate golden summary using same prompt as RAG pipeline
            golden_summary = self._generate_golden_summary(grant)
            
            # Identify privacy elements that should be removed
            privacy_elements = self._identify_privacy_elements(grant)
            
            example = EvaluationExample(
                grant_id=grant.metadata["grant_id"],
                original_text=grant.page_content,
                golden_summary=golden_summary,
                privacy_elements=privacy_elements
            )
            evaluation_examples.append(example)
            
        return evaluation_examples

    def _sample_diverse_grants(self, grants: List[Document], num_examples: int = 15) -> List[Document]:
        """
        Sample grants ensuring diversity across funds and other characteristics
        """
        # Group grants by fund
        fund_groups = {
            "LTFF": [],
            "EAIF": [],
            "AWF": []
        }
        
        for grant in grants:
            fund = grant.metadata.get("fund_evaluating")  # Access through metadata
            if fund in fund_groups:
                fund_groups[fund].append(grant)
        
        # Calculate samples per fund (roughly equal distribution)
        samples_per_fund = num_examples // len(fund_groups)
        remainder = num_examples % len(fund_groups)
        
        selected_grants = []
        for fund, fund_grants in fund_groups.items():
            # Add remainder samples to first funds
            n_samples = samples_per_fund + (1 if remainder > 0 else 0)
            remainder = max(0, remainder - 1)
            
            # Sort by length to ensure diversity in grant complexity
            fund_grants.sort(key=lambda x: len(x.page_content))
            
            # Take equal samples from start, middle, and end to get diversity in length
            third = len(fund_grants) // 3
            samples = (
                fund_grants[:third][:n_samples//3] +  # Short grants
                fund_grants[third:2*third][:(n_samples//3)] +  # Medium grants
                fund_grants[2*third:][:n_samples - (2*(n_samples//3))]  # Long grants
            )
            selected_grants.extend(samples)
        
        return selected_grants

    def _generate_golden_summary(self, grant: Document) -> str:
        """
        Generate golden summary using the RAG pipeline's prompt and LLM
        but bypassing retrieval since we have the full document
        """        
        # Use the pipeline's generate_summary method directly
        # We pass None as grant_id since we're providing context directly
        summary = self.rag_pipeline.generate_summary(
            grant_id=None,  # Not needed when providing context directly
            context=grant.page_content  # Pass the context directly
        )
        
        return summary

    def _identify_privacy_elements(self, grant: Document) -> List[str]:
        """
        Identify privacy-sensitive elements that should be removed from summaries
        """
        privacy_elements = []
        text = grant.page_content
        
        # Look for common patterns that indicate private information
        patterns = [
            # Organizations
            r'(?:University|Institute|Foundation|Lab|Center) of [A-Z][a-zA-Z\s]+',
            r'[A-Z][a-zA-Z\s]+ (?:University|Institute|Foundation|Lab|Center)',
            # Names with titles
            r'(?:Dr\.|Prof\.|Professor|Mr\.|Ms\.|Mrs\.) [A-Z][a-zA-Z\s]+',
            # Dates
            r'\b(?:19|20)\d{2}\b',  # Years
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b',
            # Locations
            r'[A-Z][a-zA-Z\s]+(?:University|College)',
            r'[A-Z][a-zA-Z\s]+(?:Institute|Center|Laboratory)'
        ]
        
        import re
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            privacy_elements.extend(match.group() for match in matches)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(privacy_elements))

def main():
    # Load grants from Airtable
    grants = load_grants_from_airtable()
    
    # Create evaluation dataset
    generator = GrantEvaluationGenerator()
    evaluation_examples = generator.create_evaluation_dataset(grants)
    
    # Save dataset
    output_path = Path("evaluation_dataset.json")
    with output_path.open("w") as f:
        json.dump(
            [vars(example) for example in evaluation_examples],
            f,
            indent=2
        )
    print(f"Saved {len(evaluation_examples)} evaluation examples to {output_path}")

if __name__ == "__main__":
    main()
