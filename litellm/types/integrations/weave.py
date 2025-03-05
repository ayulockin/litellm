from pydantic import BaseModel
from typing import Dict, Any


class WeaveCredentials(BaseModel):
    wandb_api_key: str
    wandb_entity: str
    weave_project: str


class WeaveLogData(BaseModel):
    standard_logging_data: Dict[str, Any]
    call_type: str
    error_str: str
    error_information: Dict[str, str]
    response: Dict[str, Any]
