from typing import Optional, Dict, Any

class ByakuException(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}

class ModelUnavailableException(ByakuException):
    def __init__(self, model_name: str, hint: str):
        super().__init__(
            code="MODEL_UNAVAILABLE",
            message=f"Model '{model_name}' is not available or weights are missing. {hint}",
            status_code=503
        )

class ValidationException(ByakuException):
    def __init__(self, message: str, code: str = "VALIDATION_ERROR", status_code: int = 400):
        super().__init__(code=code, message=message, status_code=status_code)

class NotFoundException(ByakuException):
    def __init__(self, resource: str, identifier: str):
        super().__init__(
            code="NOT_FOUND",
            message=f"{resource} '{identifier}' was not found.",
            status_code=404
        )

class GoneException(ByakuException):
    def __init__(self, message: str = "Resource has expired and is no longer available."):
        super().__init__(
            code="RESOURCE_EXPIRED",
            message=message,
            status_code=410
        )
