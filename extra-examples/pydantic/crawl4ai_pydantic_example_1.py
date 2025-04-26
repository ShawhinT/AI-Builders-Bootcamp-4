import os
import asyncio
from pydantic import BaseModel, Field
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai import LLMConfig

# Define Pydantic schema for data extraction
class Product(BaseModel):
    name: str = Field(..., description="Product name from the webpage")
    price: str = Field(..., description="Current price including currency")
    rating: float = Field(None, description="User rating between 1-5 stars")
    features: list[str] = Field(..., description="Key product features")

# Configure LLM extraction strategy with proper LLMConfig
llm_strategy = LLMExtractionStrategy(
    llm_config=LLMConfig(
        provider="openai/gpt-4o",
        api_token=os.getenv("OPENAI_API_KEY")
    ),
    schema=Product.schema(),
    extraction_type="schema",
    instruction="Extract product details from the e-commerce page",
    chunk_token_threshold=2048,
    verbose=True
)

# Set up crawler configuration with BrowserConfig
browser_config = BrowserConfig(
    headless=True,
    verbose=True,
    extra_args=["--disable-gpu", "--no-sandbox"]
)

# Set up crawler configuration with CrawlerRunConfig
crawl_config = CrawlerRunConfig(
    extraction_strategy=llm_strategy,
    cache_mode=CacheMode.BYPASS,
    word_count_threshold=100
)

async def main():
    async with AsyncWebCrawler(config=browser_config) as crawler:  # Added browser config
        result = await crawler.arun(
            url="https://www.amazon.com/alm/storefront/?almBrandId=VUZHIFdob2xlIEZvb2Rz",
            config=crawl_config
        )
        
        if result.success:
            try:
                # Validate extraction with Pydantic
                product_data = Product.parse_raw(result.extracted_content)
                print(f"Extracted product: {product_data.json(indent=2)}")
            except Exception as e:
                print(f"Validation error: {str(e)}")
                print(f"Raw content: {result.extracted_content}")
        else:
            print("Extraction failed:", result.error_message)

if __name__ == "__main__":
    asyncio.run(main())

'''
Example Output:

[
    {
        "name": "Annie's Frozen Pizza Poppers, Three Cheese, Snacks, 6.8 oz, 15 ct",
        "price": "$4.59",
        "rating": null,
        "features": [
            "Three Cheese",
            "Snacks",
            "6.8 oz",
            "15 ct"
        ],
        "error": false
    },
    ...
]
'''