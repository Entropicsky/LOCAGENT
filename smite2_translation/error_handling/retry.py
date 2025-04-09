import time
import random
import logging
from typing import Callable, Any, Optional, Type
from functools import wraps

# Assuming ErrorHandler and enums are importable
try:
    from .error_handler import ErrorHandler, ErrorSeverity, ErrorCategory
except ImportError:
    # Fallback if running script directly or structure differs
    # This might need adjustment
    ErrorHandler = None
    ErrorSeverity = None
    ErrorCategory = None
    print("Warning: Could not import ErrorHandler for retry decorator", file=sys.stderr)

logger = logging.getLogger(__name__)

def retry_with_exponential_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    allowed_exceptions: Optional[tuple[Type[Exception], ...]] = (Exception,),
    error_handler: Optional[ErrorHandler] = None,
    error_category: Optional[ErrorCategory] = ErrorCategory.API, # Default category for retries
    error_severity: ErrorSeverity = ErrorSeverity.MEDIUM
):
    """Decorator factory for retrying a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay in seconds.
        jitter: Whether to add random jitter to the delay.
        allowed_exceptions: Tuple of exception types to catch and retry on.
        error_handler: An instance of ErrorHandler to log retry attempts/failures.
        error_category: The ErrorCategory to use when logging.
        error_severity: The ErrorSeverity to use when logging the final failure.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            retries = 0
            delay = base_delay
            while True:
                try:
                    return func(*args, **kwargs)
                except allowed_exceptions as e:
                    retries += 1
                    if retries > max_retries:
                        error_message = f"Function {func.__name__} failed after {max_retries} retries."
                        if error_handler:
                            error_handler.log_error(
                                error_message,
                                category=error_category,
                                severity=error_severity, # Use specified severity for final failure
                                exception=e,
                                details={"args": args, "kwargs": kwargs, "retries": retries}
                            )
                        else:
                            logger.error(error_message, exc_info=e)
                        raise e # Re-raise the exception after logging
                    
                    current_delay = min(delay, max_delay)
                    if jitter:
                        current_delay = random.uniform(current_delay / 2, current_delay * 1.5)
                    
                    warning_message = (
                        f"Function {func.__name__} failed with {type(e).__name__}. "
                        f"Retrying in {current_delay:.2f} seconds... (Attempt {retries}/{max_retries})"
                    )
                    if error_handler:
                        # Log retries typically as WARNING or INFO
                        # Use MEDIUM as the equivalent warning level based on enum definition
                        log_sev = ErrorSeverity.MEDIUM if error_severity.value <= ErrorSeverity.MEDIUM.value else ErrorSeverity.INFO
                        error_handler.log_error(
                            warning_message,
                            category=error_category,
                            severity=log_sev,
                            exception=e, # Include exception in retry log
                            details={"args": args, "kwargs": kwargs, "retries": retries, "delay": current_delay}
                        )
                    else:
                        logger.warning(warning_message)
                    
                    time.sleep(current_delay)
                    delay *= 2 # Exponential backoff
        return wrapper
    return decorator

# Example Usage:
#
# if ErrorHandler:
#     error_handler_instance = ErrorHandler()
# else:
#     error_handler_instance = None
#
# @retry_with_exponential_backoff(max_retries=3, error_handler=error_handler_instance)
# def potentially_flaky_api_call(param):
#     print(f"Making API call with {param}...")
#     if random.random() < 0.6: # Simulate 60% failure rate
#         raise ConnectionError("Failed to connect to API")
#     print("API call successful!")
#     return {"result": "success", "param": param}
#
# if __name__ == '__main__':
#     # Setup logging if running standalone for testing
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     try:
#         result = potentially_flaky_api_call("test_param")
#         print(f"Final result: {result}")
#     except Exception as e:
#         print(f"Caught final exception: {e}")
#
#     if error_handler_instance:
#         print("--- Error Report ---")
#         print(json.dumps(error_handler_instance.generate_error_report(), indent=2)) 