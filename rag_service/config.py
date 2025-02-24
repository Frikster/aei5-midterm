import os
from dotenv import load_dotenv
import getpass

# Load environment variables from .env file
load_dotenv()

# Required API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Model configurations
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI's embedding model (Anthropic doesn't offer their own)
LLM_MODEL = "claude-3-5-sonnet-latest"     # Claude model for summary generation

# Check for required environment variables
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

# Add Airtable configuration
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_GRANTS_BASE_ID = os.getenv("AIRTABLE_GRANTS_BASE_ID")
AIRTABLE_GRANTS_TABLE_ID = os.getenv("AIRTABLE_GRANTS_TABLE_ID")
AIRTABLE_GRANTS_PUBLISHED_PUBLIC_VIEW = os.getenv("AIRTABLE_GRANTS_PUBLISHED_PUBLIC_VIEW_NO_ANONYMOUS")

# Airtable Fund IDs
FUND_ID_TO_FUND_STRING = {
    os.getenv("AIRTABLE_LTFF_ID"): "LTFF",
    os.getenv("AIRTABLE_EAIF_ID"): "EAIF",
    os.getenv("AIRTABLE_AWF_ID"): "AWF"
}

# Interactive API key entry (jupyter notebook)
# os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter Your OpenAI API Key: ")
# os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter Your Anthropic API Key: ")
# os.environ["AIRTABLE_API_KEY"] = getpass.getpass("Enter Your Airtable API Key: ")
# os.environ["AIRTABLE_GRANTS_BASE_ID"] = getpass.getpass("Enter Your Airtable Grants Base ID: ")
# os.environ["AIRTABLE_GRANTS_TABLE_ID"] = getpass.getpass("Enter Your Airtable Grants Table ID: ")
# os.environ["AIRTABLE_GRANTS_PUBLISHED_PUBLIC_VIEW"] = getpass.getpass("Enter Your Airtable Grants Published Public View: ")
# os.environ["AIRTABLE_LTFF_ID"] = getpass.getpass("Enter Your Airtable LTFF ID: ")
# os.environ["AIRTABLE_EAIF_ID"] = getpass.getpass("Enter Your Airtable EAIF ID: ")
# os.environ["AIRTABLE_AWF_ID"] = getpass.getpass("Enter Your Airtable AWF ID: ")