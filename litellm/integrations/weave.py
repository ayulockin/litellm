from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
import os
import json
import time
import uuid
import requests
from typing import Optional, Dict, Any, List, Union

from litellm.types.integrations.weave import (
    WeaveCredentials,
    WeaveLogData
    # TODO: more types
)


class WeaveLogger(CustomLogger):
    def __init__(
        self,
        wandb_api_key: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        weave_project: Optional[str] = None,
        weave_base_url: str = "https://trace.wandb.ai/",
        **kwargs,
    ):
        self.credentials = self._get_credentials_from_env(
            wandb_api_key=wandb_api_key,
            wandb_entity=wandb_entity,
            weave_project=weave_project,
        )

        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        super().__init__(**kwargs)

    def _get_credentials_from_env(
            self,
            wandb_api_key: Optional[str] = None,
            wandb_entity: Optional[str] = None,
            weave_project: Optional[str] = None,
    ) -> WeaveCredentials:
        _credentials_api_key = wandb_api_key or os.getenv("WANDB_API_KEY")
        if _credentials_api_key is None:
            raise Exception("W&B API Key is not set. Visit https://wandb.ai/authorize to get one.")

        # Note: If entity is not set, the traces will be logged to the user's default entity.
        # The default entity can be set in the W&B UI Settings page or it can be set via the WANDB_ENTITY environment variable.
        _credentials_entity = wandb_entity or os.getenv("WANDB_ENTITY")

        _credentials_project = weave_project or os.getenv("WANDB_PROJECT")
        if _credentials_project is None:
            raise Exception("Please provide a project name using the `weave_project` parameter or the `WANDB_PROJECT` environment variable.")
        
        return WeaveCredentials(
            wandb_api_key=_credentials_api_key,
            wandb_entity=_credentials_entity,
            weave_project=_credentials_project,
        )
    
    def _prepare_data(self, kwargs):
        """Prepare the data to be logged to Weave"""
        standard_logging_data = kwargs.get("standard_logging_data")

        # Remove tokens usage because it will be logged using `call/end` endpoint.
        def _remove_keys_with_tokens(data: dict) -> dict:
            return {k: v for k, v in data.items() if 'token' not in k}

        standard_logging_data = _remove_keys_with_tokens(standard_logging_data)

        # Call type will inform the display name of the operation in Weave.
        call_type = standard_logging_data.get("call_type")
        _ = standard_logging_data.pop("stream", None)

        # Remove timing keys because they will be logged as payloads to `call/start` and `call/end`.
        def _remove_timing_keys(data: dict) -> dict:
            timing_keys = ["startTime", "endTime", "completionStartTime", "response_time"]
            return {k: v for k, v in data.items() if k not in timing_keys}

        standard_logging_data = _remove_timing_keys(standard_logging_data)

        # Metadata will be logged as a payload to `call/start` but we redact sensitive keys
        # and keys with no value. This is to avoid cluttering the UI with unnecessary data.
        metadata = standard_logging_data.pop("metadata", None)

        def _redact_and_clean_metadata(metadata: dict) -> dict:
            if not metadata:
                return {}

            # Redact keys with _api_ in them
            redacted_metadata = {k: v for k, v in metadata.items() if "_api_" not in k}

            # Remove keys with None values
            cleaned_metadata = {k: v for k, v in redacted_metadata.items() if v is not None}

            # If applied_guardrails is empty, remove it
            if len(cleaned_metadata.get("applied_guardrails", [])) == 0:
                cleaned_metadata.pop("applied_guardrails", None)

            return cleaned_metadata

        metadata = _redact_and_clean_metadata(metadata)
        standard_logging_data["metadata"] = metadata

        # We remove the `hidden_params` key.
        _ = standard_logging_data.pop("hidden_params", None)

        # Handle model_map_information
        model_map_information = standard_logging_data.pop("model_map_information", None)

        def _clean_model_map_information(model_map_info: dict) -> dict:
            if not model_map_info:
                return {}

            # Remove keys with None values
            cleaned_model_map_info = {k: v for k, v in model_map_info.items() if v is not None}

            # Remove nested keys with None values in model_map_value
            if "model_map_value" in cleaned_model_map_info:
                # Remove keys with None values
                cleaned_model_map_info["model_map_value"] = {
                    k: v for k, v in cleaned_model_map_info["model_map_value"].items() if v is not None
                }

                # Remove keys with substring "support"
                cleaned_model_map_info["model_map_value"] = {
                    k: v for k, v in cleaned_model_map_info["model_map_value"].items() if "support" not in k
                }

            return cleaned_model_map_info

        model_map_information = _clean_model_map_information(model_map_information)
        standard_logging_data["model_map_information"] = model_map_information

        error_str = standard_logging_data.pop("error_str", None)
        error_information = standard_logging_data.pop("error_information", None)
        # Don't know know to handle these.
        _ = standard_logging_data.pop("response_cost_failure_debug_info", None)
        _ = standard_logging_data.pop("guardrail_information", None)

        # The response will be logged as a payload to `call/end`.
        llm_response = standard_logging_data.get("response", None)

        return WeaveLogData(
            standard_logging_data=standard_logging_data,
            call_type=call_type,
            error_str=error_str,
            error_information=error_information,
            response=llm_response,
        )


    def _prepare_call_start_payload(self, log_data, start_time_str):
        """Prepare the payload to be logged to Weave as a call start"""
        payload = {
            "start": {
                "project_id": f"{self.credentials.wandb_entity}/{self.credentials.weave_project}" if self.credentials.wandb_entity else self.credentials.weave_project,
                "op_name": f"litellm.{log_data.call_type}",
                "display_name": f"litellm.{log_data.call_type}",
                "started_at": start_time_str,
                "attributes": {},
                "inputs": log_data.standard_logging_data,
            }
        }
        return json.dumps(payload)


    def _prepare_call_end_payload(self, log_data, end_time_str):
        """Prepare the payload to be logged to Weave as a call end"""
        pass
    
    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        log_data = self._prepare_data(kwargs)
        start_time_str = start_time.isoformat()
        end_time_str = end_time.isoformat()

        self._log_call_start(log_data, start_time_str)
        self._log_call_end(log_data, end_time_str)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        log_data = self._prepare_data(kwargs)
        start_time_str = start_time.isoformat()
        end_time_str = end_time.isoformat()

        self._log_call_start(log_data, start_time_str)
        self._log_call_end(log_data, end_time_str)
