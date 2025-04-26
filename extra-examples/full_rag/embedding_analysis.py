import os
import pandas as pd
from nomic import atlas, AtlasDataset
from supabase import create_client, Client
from dotenv import load_dotenv
from typing import Dict, List
import logging
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

def fetch_data(supabase: Client) -> List[Dict]:
    """Fetch data from Supabase using a stored procedure."""
    try:
        response = supabase.rpc('get_pages_with_keywords', {'limit_param': 100000}).execute()
        return response.data
    except Exception as e:
        logger.error(f"Failed to fetch data from Supabase: {e}")
        raise

def create_embeddings(data: List[Dict]):
    """Create embeddings using Nomic."""
    try:
        # Initialize Nomic client
        api_key = os.getenv("NOMIC_API_KEY")
        if not api_key:
            raise ValueError("Missing NOMIC_API_KEY in .env file")
        
        # Convert to DataFrame and extract netloc from URLs
        df = pd.DataFrame(data)
        df = df[["url", "keyword", "definition"]]
        df['url'] = df['url'].apply(lambda x: urlparse(x).netloc if pd.notna(x) else x)
        
        logger.info(f"Original DataFrame size: {len(df)}")

        definitions = atlas.map_data(
            df, 
            indexed_field="definition", 
            identifier="Documentation Definitions v2", 
            topic_model=True
            )
            
        return definitions
        
    except Exception as e:
        logger.error(f"Failed to create embeddings: {e}")
        raise

def main():
    """Main execution function."""
    try:
        # Fetch data
        logger.info("Fetching data from Supabase...")
        data = fetch_data(supabase)

        for keys, values in data[1].items():
            print(keys)
            
        # Create embeddings
        logger.info("Creating embeddings with Nomic...")
        datasets = create_embeddings(data)
        

        logger.info("Analysis complete! Check the Atlas UI for visualizations.")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main() 
