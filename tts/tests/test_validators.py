"""Tests for parameter validators."""

import pytest
from src.validators import TTSRequestValidator, ValidationResult
from src.config import ModelConfig


@pytest.fixture
def model_config():
    """Create a test model configuration."""
    return ModelConfig()


@pytest.fixture
def validator(model_config):
    """Create a validator instance."""
    return TTSRequestValidator(model_config)


class TestTTSRequestValidator:
    """Tests for TTSRequestValidator."""

    def test_validate_valid_params(self, validator):
        """Test validation of valid parameters."""
        params = {
            "text": "Hello, world!",
            "mode": "streaming"
        }
        result = validator.validate(params)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_missing_text(self, validator):
        """Test validation fails when text is missing."""
        params = {"mode": "streaming"}
        result = validator.validate(params)
        assert not result.is_valid
        assert "text" in result.errors
        assert "required" in result.errors["text"]

    def test_validate_empty_text(self, validator):
        """Test validation fails when text is empty."""
        params = {
            "text": "   ",
            "mode": "streaming"
        }
        result = validator.validate(params)
        assert not result.is_valid
        assert "text" in result.errors

    def test_validate_text_too_long(self, validator, model_config):
        """Test validation fails when text exceeds max length."""
        params = {
            "text": "a" * (model_config.max_text_length + 1),
            "mode": "streaming"
        }
        result = validator.validate(params)
        assert not result.is_valid
        assert "text" in result.errors
        assert "too long" in result.errors["text"]

    def test_validate_invalid_mode(self, validator):
        """Test validation fails with invalid mode."""
        params = {
            "text": "Hello",
            "mode": "invalid_mode"
        }
        result = validator.validate(params)
        assert not result.is_valid
        assert "mode" in result.errors

    def test_validate_cfg_value_out_of_range(self, validator):
        """Test validation fails when cfg_value is out of range."""
        params = {
            "text": "Hello",
            "cfg_value": 100.0
        }
        result = validator.validate(params)
        assert not result.is_valid
        assert "cfg_value" in result.errors

    def test_validate_inference_timesteps_invalid(self, validator):
        """Test validation fails when inference_timesteps is invalid."""
        params = {
            "text": "Hello",
            "inference_timesteps": 100
        }
        result = validator.validate(params)
        assert not result.is_valid
        assert "inference_timesteps" in result.errors

    def test_validate_boolean_params(self, validator):
        """Test validation of boolean parameters."""
        params = {
            "text": "Hello",
            "normalize": "true"  # Should be boolean
        }
        result = validator.validate(params)
        assert not result.is_valid
        assert "normalize" in result.errors

    def test_validate_valid_optional_params(self, validator):
        """Test validation with valid optional parameters."""
        params = {
            "text": "Hello",
            "cfg_value": 2.5,
            "inference_timesteps": 15,
            "normalize": True,
            "denoise": False
        }
        result = validator.validate(params)
        assert result.is_valid

    def test_validate_with_prompt_wav_url(self, validator):
        """Test validation with prompt_wav_url."""
        params = {
            "text": "Hello",
            "prompt_wav_url": "https://example.com/audio.wav"
        }
        result = validator.validate(params)
        assert result.is_valid

    def test_validate_invalid_prompt_wav_url(self, validator):
        """Test validation fails with invalid prompt_wav_url type."""
        params = {
            "text": "Hello",
            "prompt_wav_url": 123  # Should be string or None
        }
        result = validator.validate(params)
        assert not result.is_valid
        assert "prompt_wav_url" in result.errors
