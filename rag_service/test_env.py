import sys
from anthropic import Anthropic
from openai import OpenAI
# Try relative import first
try:
    from .config import ANTHROPIC_API_KEY, OPENAI_API_KEY, EMBEDDING_MODEL, LLM_MODEL
# Fall back to absolute import
except ImportError:
    from config import ANTHROPIC_API_KEY, OPENAI_API_KEY, EMBEDDING_MODEL, LLM_MODEL

def test_environment():
    print("Testing Python environment and dependencies...")
    print(f"Python version: {sys.version}")
    
    print("\nTesting API connections...")
    
    # Test OpenAI connection
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        embedding = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input="Test message"
        )
        print("✓ OpenAI connection successful")
    except Exception as e:
        print(f"✗ OpenAI connection failed: {str(e)}")
        
    # Test Anthropic connection
    try:
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model=LLM_MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": "Say hello"}]
        )
        print("✓ Anthropic connection successful")
    except Exception as e:
        print(f"✗ Anthropic connection failed: {str(e)}")

if __name__ == "__main__":
    test_environment() 