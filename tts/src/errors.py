"""Error definitions for VoxCPM TTS Server."""


class VoxCPMError(Exception):
    """Base error class for VoxCPM TTS Server."""

    code: str = "INTERNAL_ERROR"
    http_status: int = 500
    message: str = "An internal error occurred"

    def __init__(self, message: str = None, details: dict = None):
        self.message = message or self.message
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict:
        """Convert error to dictionary format."""
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details
        }


class ValidationError(VoxCPMError):
    """Parameter validation error."""

    code = "INVALID_PARAMS"
    http_status = 400
    message = "Invalid parameters"

    def __init__(self, field: str = None, reason: str = None, errors: dict = None):
        self.field = field
        self.reason = reason
        self.errors = errors or {}
        details = {}
        if field:
            details["field"] = field
        if reason:
            details["reason"] = reason
        if errors:
            details["errors"] = errors
        super().__init__(message=f"Validation failed: {field} - {reason}" if field and reason else "Validation failed", details=details)


class TextTooLongError(ValidationError):
    """Text length exceeds maximum."""

    code = "TEXT_TOO_LONG"
    http_status = 400
    message = "Text length exceeds maximum allowed"

    def __init__(self, length: int, max_length: int):
        super().__init__(
            field="text",
            reason=f"Text length ({length}) exceeds maximum ({max_length})"
        )
        self.details = {"length": length, "max_length": max_length}


class UnsupportedFormatError(VoxCPMError):
    """Unsupported format error."""

    code = "UNSUPPORTED_FORMAT"
    http_status = 400
    message = "Unsupported format"


class ModelNotLoadedError(VoxCPMError):
    """Model not loaded error."""

    code = "MODEL_NOT_LOADED"
    http_status = 503
    message = "Model is not loaded"


class GenerationFailedError(VoxCPMError):
    """TTS generation failed error."""

    code = "GENERATION_FAILED"
    http_status = 500
    message = "Failed to generate audio"


class TimeoutError(VoxCPMError):
    """Request timeout error."""

    code = "TIMEOUT"
    http_status = 504
    message = "Request timed out"


class RateLimitError(VoxCPMError):
    """Rate limit exceeded error."""

    code = "RATE_LIMITED"
    http_status = 429
    message = "Rate limit exceeded"

    def __init__(self, limit: int = None, window: int = None):
        details = {}
        if limit:
            details["limit"] = limit
        if window:
            details["window"] = f"{window}s"
        super().__init__(details=details)


class InvalidJSONError(VoxCPMError):
    """Invalid JSON error."""

    code = "INVALID_JSON"
    http_status = 400
    message = "Invalid JSON format"


class UnknownMessageTypeError(VoxCPMError):
    """Unknown message type error."""

    code = "UNKNOWN_MESSAGE_TYPE"
    http_status = 400
    message = "Unknown message type"


class RequestNotFoundError(VoxCPMError):
    """Request not found error."""

    code = "REQUEST_NOT_FOUND"
    http_status = 404
    message = "Request not found"
