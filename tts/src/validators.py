"""Parameter validators for TTS requests."""

from typing import Optional
from .models.messages import TTSRequestParams
from .config import ModelConfig
from .errors import ValidationError


class ValidationResult:
    """Validation result."""

    def __init__(self):
        self.is_valid = True
        self.errors = {}

    def add_error(self, field: str, message: str):
        """Add a validation error."""
        self.is_valid = False
        self.errors[field] = message

    @property
    def error_message(self) -> str:
        """Get formatted error message."""
        return "Validation failed: " + ", ".join(
            f"{k}: {v}" for k, v in self.errors.items()
        )


class TTSRequestValidator:
    """Validator for TTS request parameters."""

    def __init__(self, config: ModelConfig):
        self.config = config

    def validate(self, params: dict) -> ValidationResult:
        """
        Validate TTS request parameters.

        Args:
            params: Parameters dictionary to validate

        Returns:
            ValidationResult with validation status and errors
        """
        result = ValidationResult()

        # Validate text
        if "text" not in params:
            result.add_error("text", "is required")
        elif not isinstance(params["text"], str):
            result.add_error("text", "must be a string")
        elif len(params["text"].strip()) == 0:
            result.add_error("text", "cannot be empty")
        elif len(params["text"]) > self.config.max_text_length:
            result.add_error(
                "text",
                f"too long (max {self.config.max_text_length} characters)"
            )

        # Validate mode
        if "mode" in params and params["mode"] not in ("streaming", "non_streaming"):
            result.add_error("mode", "must be 'streaming' or 'non_streaming'")

        # Validate cfg_value
        if "cfg_value" in params:
            cfg_value = params["cfg_value"]
            min_val, max_val = self.config.cfg_value_range
            if not isinstance(cfg_value, (int, float)):
                result.add_error("cfg_value", "must be a number")
            elif cfg_value < min_val or cfg_value > max_val:
                result.add_error(
                    "cfg_value",
                    f"must be between {min_val} and {max_val}"
                )

        # Validate inference_timesteps
        if "inference_timesteps" in params:
            steps = params["inference_timesteps"]
            min_val, max_val = self.config.inference_timesteps_range
            if not isinstance(steps, int):
                result.add_error("inference_timesteps", "must be an integer")
            elif steps < min_val or steps > max_val:
                result.add_error(
                    "inference_timesteps",
                    f"must be between {min_val} and {max_val}"
                )

        # Validate boolean parameters
        for param in ("normalize", "denoise", "retry_badcase"):
            if param in params and not isinstance(params[param], bool):
                result.add_error(param, "must be a boolean")

        # Validate retry_badcase_max_times
        if "retry_badcase_max_times" in params:
            max_times = params["retry_badcase_max_times"]
            min_val, max_val = self.config.retry_badcase_max_times_range
            if not isinstance(max_times, int):
                result.add_error("retry_badcase_max_times", "must be an integer")
            elif max_times < min_val or max_times > max_val:
                result.add_error(
                    "retry_badcase_max_times",
                    f"must be between {min_val} and {max_val}"
                )

        # Validate retry_badcase_ratio_threshold
        if "retry_badcase_ratio_threshold" in params:
            threshold = params["retry_badcase_ratio_threshold"]
            min_val, max_val = self.config.retry_badcase_ratio_threshold_range
            if not isinstance(threshold, (int, float)):
                result.add_error("retry_badcase_ratio_threshold", "must be a number")
            elif threshold < min_val or threshold > max_val:
                result.add_error(
                    "retry_badcase_ratio_threshold",
                    f"must be between {min_val} and {max_val}"
                )

        # Validate prompt_wav_url
        if "prompt_wav_url" in params and params["prompt_wav_url"] is not None:
            if not isinstance(params["prompt_wav_url"], str):
                result.add_error("prompt_wav_url", "must be a string")

        # Validate prompt_text
        if "prompt_text" in params and params["prompt_text"] is not None:
            if not isinstance(params["prompt_text"], str):
                result.add_error("prompt_text", "must be a string")

        return result
