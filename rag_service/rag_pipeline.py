from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from tqdm import tqdm
from pathlib import Path
from rag_service.data_loader import load_grants_from_airtable

from rag_service.config import (
    ANTHROPIC_API_KEY, 
    LLM_MODEL, 
    EMBEDDING_MODEL,
)

class GrantSummaryPipeline:
    def __init__(self, persist_dir: str = "vector_store", force_disable_check_same_thread: bool = False):
        # Initialize models
        self.llm = ChatAnthropic(
            model=LLM_MODEL,
            anthropic_api_key=ANTHROPIC_API_KEY
        )
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        
        # Initialize vector store with persistence
        self.persist_dir = Path(persist_dir)
        client = QdrantClient(
            path=str(self.persist_dir),
            force_disable_check_same_thread=force_disable_check_same_thread
        )
        
        # Check if collection exists before creating
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if "grants" not in collection_names:
            print(f"Creating new vector store in {persist_dir}")
            client.create_collection(
                collection_name="grants",
                vectors_config=VectorParams(
                    #  size=1536 because:
                    #  - OpenAI's text-embedding-3-small outputs 1536-dimensional vectors
                    #  - Must match embedding model's output dimension
                    size=1536,
                    # distance=Distance.COSINE because:
                    # - Cosine similarity is standard for text embeddings
                    # - Works well with normalized vectors
                    # - Other options include:
                    #   * Distance.DOT - Dot product (faster but less accurate)
                    #   * Distance.EUCLID - Euclidean (better for non-normalized vectors)
                    distance=Distance.COSINE,
                ),
            )
        else:
            print(f"Loading existing vector store from {persist_dir}")
            
        self.vectorstore = QdrantVectorStore(
            client=client,
            embedding=self.embeddings,
            collection_name="grants"
        )
        
        # Define text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            # Chunk size of 1000 chosen because:
            # - Large enough to maintain context for grant summaries
            # - Small enough to get specific retrievals
            # - Balances token limits with Claude's 100k context window
            chunk_size=1000,
            
            # Overlap of 200 ensures:
            # - Key information isn't lost at chunk boundaries
            # - Important context isn't split between chunks
            # - ~20% overlap is a common best practice for RAG
            chunk_overlap=200
        )
        
        # Update prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI that generates summaries of research grants.
            Use the provided context to generate a two-sentence summary following this format:

            FIRST SENTENCE - Research Goal:
            - Begin with "This project"
            - State the high-level goal
            - Include 2-3 key methodological approaches
            - Focus on concrete deliverables
            - Maintain technical precision
            - Avoid institution names or specific people

            SECOND SENTENCE - Track Record:
            - List concrete research outputs
            - Include publication venues where relevant
            - Add quantitative metrics of success
            - Remove identifying details while keeping specific achievements
            - Include leadership/organizational achievements if relevant
            - Use objective metrics where possible

            Style Guidelines:
            - Use active voice
            - Avoid jargon unless field-standard
            - Keep to ~100 words total
            - Remove all proper nouns, dates, and institutions
            
            If you cannot generate a good summary from the context, respond with: "Insufficient context to generate summary."
            """),
            ("user", "Context: {context}\n\nGenerate a summary of this grant:"),
        ])

    def load_grants(self):
        """Load grants from Airtable and index them in the vector store"""
        documents = load_grants_from_airtable()
        
        # Split documents into chunks with progress bar
        print("Splitting documents into chunks...")
        chunks = []
        for doc in tqdm(documents, desc="Processing documents"):
            # chunks not documents are added to vector store
            doc_chunks = self.text_splitter.split_documents([doc])
            chunks.extend(doc_chunks)
        print(f"Created {len(chunks)} chunks")
        
        # Add chunks to vector store in batches
        if chunks:
            print("Adding chunks to vector store...")
            # Process in batches of 100 to avoid overwhelming the API
            batch_size = 100
            for i in tqdm(range(0, len(chunks), batch_size), desc="Adding to vector store"):
                batch = chunks[i:i + batch_size]
                self.vectorstore.add_documents(batch)
            print("Added all chunks to vector store")
        
    def generate_summary(self, grant_id: str | None = None, context: str | None = None) -> str:
        """Generate a summary for a specific grant
        
        Args:
            grant_id: ID of the grant to summarize. Required if context not provided.
            context: Direct context to use instead of retrieval. Optional.
        """
        if grant_id is None and context is None:
            raise ValueError("Either grant_id or context must be provided")
        
        if context is not None:
            # Use provided context directly
            chain = self.prompt | self.llm
            response = chain.invoke({"context": context})
            return response.content
        
        # Original retrieval logic for grant_id
        retriever = self.vectorstore.as_retriever(
            # "similarity" search finds closest vector matches
            # Other options include:
            # - "mmr": Maximum Marginal Relevance (balances relevance & diversity)
            # - "similarity_score_threshold": Only returns results above score
            # We use similarity because grant summaries need precise context matching
            search_type="similarity",
            search_kwargs={
                "k": float("inf"),  # Return all chunks with the grant_id
                "filter": models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.grant_id",
                            match=models.MatchValue(value=grant_id)
                        )
                    ]
                )
            }
        )
        
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}  # TODO: question needs to be added to prompt
            | self.prompt 
            | self.llm
        )
        
        # TODO: followup questions wil need to repass entire conversation through chain.
        # TODO: question here does not matter at all yet
        # Generate summary by passing empty string as query
        # The retriever will still use the filter to get relevant chunks
        
        # since raw string the string will pass to everything in chain
        response = chain.invoke("Add more quantitative metrics")
        return response.content


def main():
    # Test the pipeline
    pipeline = GrantSummaryPipeline()
    
    # Check if collection is empty (not just if it exists)
    collection_info = pipeline.vectorstore.client.get_collection("grants")
    if collection_info.points_count == 0:
        print("Vector store is empty, loading grants...")
        pipeline.load_grants()
    else:
        print(f"Vector store contains {collection_info.points_count} points")
    
    # Test with a sample grant ID
    test_ids = ["recJYbsdzFWrtRioh", "reckdtWlsWmQ1QpcQ"]
    for grant_id in test_ids:
        print(f"\nGenerating summary for grant {grant_id}:")
        summary = pipeline.generate_summary(grant_id)
        print(summary)

if __name__ == "__main__":
    main()
