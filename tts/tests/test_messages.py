"""Tests for message models."""

import pytest
from pydantic import ValidationError
from src.models.messages import (
    TTSRequest,
    TTSRequestParams,
    CancelRequest,
    PingMessage,
    ErrorDetail,
    ErrorMessage,
)


class TestTTSRequestParams:
    """Tests for TTSRequestParams model."""

    def test_valid_params(self):
        """Test valid parameters."""
        params = TTSRequestParams(
            text="Hello, world!",
            mode="streaming"
        )
        assert params.text == "Hello, world!"
        assert params.mode == "streaming"
        assert params.cfg_value == 2.0
        assert params.inference_timesteps == 10

    def test_default_values(self):
        """Test default parameter values."""
        params = TTSRequestParams(text="Hello")
        assert params.mode == "streaming"
        assert params.cfg_value == 2.0
        assert params.inference_timesteps == 10
        assert params.normalize is False
        assert params.retry_badcase is True

    def test_empty_text_validation(self):
        """Test that empty text fails validation."""
        with pytest.raises(ValidationError):
            TTSRequestParams(text="   ")

    def test_text_stripping(self):
        """Test that text is stripped."""
        params = TTSRequestParams(text="  Hello  ")
        assert params.text == "Hello"

    def test_cfg_value_range(self):
        """Test cfg_value range validation."""
        with pytest.raises(ValidationError):
            TTSRequestParams(text="Hello", cfg_value=0.05)

        with pytest.raises(ValidationError):
            TTSRequestParams(text="Hello", cfg_value=15.0)

    def test_inference_timesteps_range(self):
        """Test inference_timesteps range validation."""
        with pytest.raises(ValidationError):
            TTSRequestParams(text="Hello", inference_timesteps=0)

        with pytest.raises(ValidationError):
            TTSRequestParams(text="Hello", inference_timesteps=100)

    def test_retry_badcase_max_times_range(self):
        """Test retry_badcase_max_times range validation."""
        with pytest.raises(ValidationError):
            TTSRequestParams(text="Hello", retry_badcase_max_times=15)

    def test_retry_badcase_ratio_threshold_range(self):
        """Test retry_badcase_ratio_threshold range validation."""
        with pytest.raises(ValidationError):
            TTSRequestParams(text="Hello", retry_badcase_ratio_threshold=0.5)


class TestTTSRequest:
    """Tests for TTSRequest model."""

    def test_valid_request(self):
        """Test valid TTS request."""
        request = TTSRequest(
            type="tts_request",
            request_id="test-123",
            params={"text": "Hello", "mode": "streaming"}
        )
        assert request.type == "tts_request"
        assert request.request_id == "test-123"
        assert isinstance(request.params, TTSRequestParams)


class TestCancelRequest:
    """Tests for CancelRequest model."""

    def test_valid_cancel(self):
        """Test valid cancel request."""
        request = CancelRequest(
            type="cancel",
            request_id="test-123"
        )
        assert request.type == "cancel"
        assert request.request_id == "test-123"


class TestPingMessage:
    """Tests for PingMessage model."""

    def test_valid_ping(self):
        """Test valid ping message."""
        ping = PingMessage(type="ping", timestamp=1234567890)
        assert ping.type == "ping"
        assert ping.timestamp == 1234567890

    def test_default_timestamp(self):
        """Test default timestamp is 0."""
        ping = PingMessage(type="ping")
        assert ping.timestamp == 0


class TestErrorMessage:
    """Tests for ErrorMessage model."""

    def test_valid_error(self):
        """Test valid error message."""
        error = ErrorMessage(
            type="error",
            request_id="test-123",
            error=ErrorDetail(
                code="INVALID_PARAMS",
                message="Invalid parameters"
            )
        )
        assert error.type == "error"
        assert error.request_id == "test-123"
        assert error.error.code == "INVALID_PARAMS"

    def test_error_with_details(self):
        """Test error message with details."""
        error = ErrorMessage(
            type="error",
            error=ErrorDetail(
                code="INVALID_PARAMS",
                message="Validation failed",
                details={"field": "text", "reason": "required"}
            )
        )
        assert error.error.details["field"] == "text"
