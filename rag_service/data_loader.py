from langchain_community.document_loaders import AirtableLoader
# Try relative import first
try:
    from .config import (
        get_airtable_api_key,
        get_airtable_grants_base_id,
        get_airtable_grants_table_id,
        get_airtable_grants_published_view,
        get_fund_id_mapping
    )
# Fall back to absolute import
except ImportError:
    from config import (
        get_airtable_api_key,
        get_airtable_grants_base_id,
        get_airtable_grants_table_id,
        get_airtable_grants_published_view,
        get_fund_id_mapping
    )
from tqdm import tqdm

def load_grants_from_airtable():
    """Load grants from Airtable and add grant_id to metadata"""
    loader = AirtableLoader(
        api_token=get_airtable_api_key(),
        base_id=get_airtable_grants_base_id(),
        table_id=get_airtable_grants_table_id(),
        view=get_airtable_grants_published_view(),
        fields=[
            "Fund Evaluating",
            "Grant Title",
            "Project short description",
            "Project summary",
            "Project goals",
            "Track record"
        ]
    )
    
    # Load documents with progress bar
    print("Loading documents from Airtable...")
    documents = []
    for record in tqdm(loader.lazy_load(), desc="Loading records"):
        # Extract record ID from the raw record data
        record_data = eval(record.page_content)  # Safely convert string to dict
        record_id = record_data.get('id')
        if not record_id:
            raise ValueError(f"No ID found in record data: {record_data}")
        record.metadata['grant_id'] = record_id
        fund_evaluating_list = record_data['fields'].get('Fund Evaluating')
        if not fund_evaluating_list:
            raise ValueError(f"No Fund Evaluating found in record data: {record_data}")
        fund_evaluating = get_fund_id_mapping()[fund_evaluating_list[0]]
        record.metadata['fund_evaluating'] = fund_evaluating
        documents.append(record)
    print(f"Loaded {len(documents)} documents from Airtable")
    
    return documents 