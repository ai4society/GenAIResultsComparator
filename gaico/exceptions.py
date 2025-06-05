class LLMMetricsError(Exception):
    """Base exception for LLM Metrics library"""


class InvalidInputError(LLMMetricsError):
    """Raised when input is invalid"""


class MetricNotImplementedError(LLMMetricsError):
    """Raised when a metric is not implemented"""
