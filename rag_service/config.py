import os
from dotenv import load_dotenv
import getpass

# Load environment variables from .env file
load_dotenv()

# Model configurations (these are fine as constants)
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "claude-3-5-sonnet-latest"

def get_required_env_var(var_name: str) -> str:
    """Get required environment variable or raise error."""
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"{var_name} not found in environment variables")
    return value

# API key getters
def get_openai_api_key(): return get_required_env_var("OPENAI_API_KEY")
def get_anthropic_api_key(): return get_required_env_var("ANTHROPIC_API_KEY")
def get_airtable_api_key(): return get_required_env_var("AIRTABLE_API_KEY")
def get_langchain_api_key(): return get_required_env_var("LANGCHAIN_API_KEY")

# Airtable config getters
def get_airtable_grants_base_id(): return get_required_env_var("AIRTABLE_GRANTS_BASE_ID")
def get_airtable_grants_table_id(): return get_required_env_var("AIRTABLE_GRANTS_TABLE_ID")
def get_airtable_grants_published_view(): return get_required_env_var("AIRTABLE_GRANTS_PUBLISHED_PUBLIC_VIEW_NO_ANONYMOUS")

def get_fund_id_mapping():
    return {
        os.getenv("AIRTABLE_LTFF_ID"): "LTFF",
        os.getenv("AIRTABLE_EAIF_ID"): "EAIF",
        os.getenv("AIRTABLE_AWF_ID"): "AWF"
    }

# os.environ["LANGCHAIN_TRACING_V2"] = "true"

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