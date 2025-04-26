from dotenv import load_dotenv
from llama_cloud_services import LlamaExtract
from llama_cloud.core.api_error import ApiError
from llama_cloud import ExtractConfig
from pydantic import BaseModel, Field
from typing import List
import json

# Load environment variables (ensure LLAMA_CLOUD_API_KEY is set in your .env file)
load_dotenv(override=True)

# Path to the PDF file
lm317_pdf = "./data/lm317.pdf"

# Initialize the LlamaExtract client
llama_extract = LlamaExtract(
    project_id="your project id",
    organization_id="your organization id",
)


# Define the data model
class VoltageRange(BaseModel):
    min_voltage: float = Field(..., description="Minimum voltage in volts")
    max_voltage: float = Field(..., description="Maximum voltage in volts")
    unit: str = Field("V", description="Voltage unit")

class PinConfiguration(BaseModel):
    pin_count: int = Field(..., description="Number of pins")
    layout: str = Field(..., description="Detailed pin layout description")

class LM317Spec(BaseModel):
    component_name: str = Field(..., description="Name of the component")
    component_type: str = Field(..., description="Component variation (e.g., LM317, LM317F)")
    output_voltage: VoltageRange = Field(..., description="Output voltage range specification")
    dropout_voltage: float = Field(..., description="Dropout voltage in volts")
    max_current: float = Field(..., description="Maximum current rating in amperes")
    input_voltage: VoltageRange = Field(..., description="Input voltage range specification")
    pin_configuration: PinConfiguration = Field(..., description="Pin configuration details")
    features: List[str] = Field([], description="List of additional technical features")

class LM317Schema(BaseModel):
    specs: List[LM317Spec] = Field(..., description="List of extracted LM317 technical specifications")


# Create the extraction agent
try:
    existing_agent = llama_extract.get_agent(name="lm317-datasheet")
    if existing_agent:
        llama_extract.delete_agent(existing_agent.id)
except ApiError as e:
    if e.status_code == 404:
        pass
    else:
        raise

# Define the extraction configuration
extract_config = ExtractConfig(
    extraction_mode="BALANCED",
)

# Create the extraction agent
agent = llama_extract.create_agent(
    name="lm317-datasheet", data_schema=LM317Schema, config=extract_config
)

# Extract structured technical specifications from the datasheet
lm317_extract = agent.extract(lm317_pdf)

# Display the extraction results
print("\nExtracted LM317 Specifications:")
print(json.dumps(lm317_extract.data, indent=2))