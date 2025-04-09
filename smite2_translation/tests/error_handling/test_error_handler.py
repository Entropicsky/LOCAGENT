import pytest
import logging
import time
from smite2_translation.error_handling import ErrorHandler, ErrorSeverity, ErrorCategory, retry_with_exponential_backoff

@pytest.fixture
def handler():
    """Fixture to create a new ErrorHandler instance for each test."""
    return ErrorHandler()

def test_error_handler_initialization(handler):
    """Test that ErrorHandler initializes with an empty error list."""
    assert handler.errors == []

def test_log_error(handler, caplog):
    """Test logging a basic error."""
    caplog.set_level(logging.INFO)
    message = "Test error message"
    category = ErrorCategory.SYSTEM
    severity = ErrorSeverity.HIGH
    details = {"code": 123}

    error_record = handler.log_error(message, category, severity, details)

    assert len(handler.errors) == 1
    logged_error = handler.errors[0]
    assert logged_error["message"] == message
    assert logged_error["category"] == str(category)
    assert logged_error["severity"] == str(severity)
    assert logged_error["details"] == details
    assert "timestamp" in logged_error

    # Check log output
    assert f"[{category}] [{severity}] {message}" in caplog.text
    assert '"code": 123' in caplog.text

def test_log_error_with_exception(handler, caplog):
    """Test logging an error with an associated exception."""
    caplog.set_level(logging.WARNING)
    message = "Something went wrong"
    category = ErrorCategory.TRANSLATION
    severity = ErrorSeverity.MEDIUM
    exception = ValueError("Invalid value")

    error_record = handler.log_error(message, category, severity, exception=exception)

    assert len(handler.errors) == 1
    logged_error = handler.errors[0]
    assert logged_error["message"] == message
    assert logged_error["exception_type"] == "ValueError"
    assert logged_error["exception_message"] == "Invalid value"

    # Check log output (includes exception info)
    assert f"[{category}] [{severity}] {message}" in caplog.text
    assert "ValueError: Invalid value" in caplog.text # Exception info logged

def test_get_errors_filtering(handler):
    """Test filtering errors by severity and category."""
    handler.log_error("Error 1", ErrorCategory.SYSTEM, ErrorSeverity.HIGH)
    handler.log_error("Error 2", ErrorCategory.API, ErrorSeverity.MEDIUM)
    handler.log_error("Error 3", ErrorCategory.SYSTEM, ErrorSeverity.LOW)

    high_severity_errors = handler.get_errors(min_severity=ErrorSeverity.HIGH)
    assert len(high_severity_errors) == 1
    assert high_severity_errors[0]["message"] == "Error 1"

    medium_or_higher_errors = handler.get_errors(min_severity=ErrorSeverity.MEDIUM)
    assert len(medium_or_higher_errors) == 2

    system_errors = handler.get_errors(category=ErrorCategory.SYSTEM)
    assert len(system_errors) == 2

    system_high_errors = handler.get_errors(min_severity=ErrorSeverity.HIGH, category=ErrorCategory.SYSTEM)
    assert len(system_high_errors) == 1
    assert system_high_errors[0]["message"] == "Error 1"

def test_has_critical_errors(handler):
    """Test checking for critical errors."""
    assert not handler.has_critical_errors()
    handler.log_error("Non-critical error", ErrorCategory.SYSTEM, ErrorSeverity.HIGH)
    assert not handler.has_critical_errors()
    handler.log_error("Critical error!", ErrorCategory.API, ErrorSeverity.CRITICAL)
    assert handler.has_critical_errors()

def test_generate_error_report(handler):
    """Test generating the error summary report."""
    handler.log_error("Sys High", ErrorCategory.SYSTEM, ErrorSeverity.HIGH)
    handler.log_error("API Medium", ErrorCategory.API, ErrorSeverity.MEDIUM)
    handler.log_error("Sys Low", ErrorCategory.SYSTEM, ErrorSeverity.LOW)
    handler.log_error("API Critical", ErrorCategory.API, ErrorSeverity.CRITICAL)

    report = handler.generate_error_report()

    assert report["total_errors"] == 4
    assert report["errors_by_severity"][str(ErrorSeverity.CRITICAL)] == 1
    assert report["errors_by_severity"][str(ErrorSeverity.HIGH)] == 1
    assert report["errors_by_severity"][str(ErrorSeverity.MEDIUM)] == 1
    assert report["errors_by_severity"][str(ErrorSeverity.LOW)] == 1
    assert report["errors_by_category"][str(ErrorCategory.SYSTEM)] == 2
    assert report["errors_by_category"][str(ErrorCategory.API)] == 2
    assert report["critical_errors_present"] is True

def test_clear_errors(handler):
    """Test clearing the error list."""
    handler.log_error("Error 1", ErrorCategory.SYSTEM, ErrorSeverity.HIGH)
    assert len(handler.errors) == 1
    handler.clear_errors()
    assert len(handler.errors) == 0
    assert not handler.has_critical_errors()

# --- Tests for retry decorator ---

# Counter for flaky function calls
flaky_call_count = 0

@pytest.fixture
def reset_flaky_counter():
    global flaky_call_count
    flaky_call_count = 0

@pytest.fixture
def retry_handler():
    """Separate handler for retry tests."""
    return ErrorHandler()

def flaky_function(fail_times: int):
    """Function that fails a specified number of times before succeeding."""
    global flaky_call_count
    flaky_call_count += 1
    if flaky_call_count <= fail_times:
        raise ConnectionError(f"Failed on attempt {flaky_call_count}")
    return f"Success on attempt {flaky_call_count}"

def test_retry_success_within_limits(reset_flaky_counter, retry_handler, caplog):
    """Test retry that succeeds before max retries."""
    caplog.set_level(logging.WARNING)
    decorated_func = retry_with_exponential_backoff(
        max_retries=3,
        base_delay=0.01, # Short delay for testing
        error_handler=retry_handler,
        allowed_exceptions=(ConnectionError,)
    )(lambda: flaky_function(fail_times=2))

    result = decorated_func()
    assert result == "Success on attempt 3"
    assert flaky_call_count == 3
    # Check handler logs for retry warnings (should be MEDIUM severity)
    retry_logs = [e for e in retry_handler.errors if e["severity"] == str(ErrorSeverity.MEDIUM)]
    assert len(retry_logs) == 2 # Failed 2 times, retried twice
    assert "Retrying in" in retry_logs[0]["message"]
    assert "Attempt 1/3" in retry_logs[0]["message"]
    assert "Retrying in" in retry_logs[1]["message"]
    assert "Attempt 2/3" in retry_logs[1]["message"]
    # Check caplog as well (if handler wasn't used)
    assert "Retrying in" in caplog.text
    assert "Attempt 1/3" in caplog.text
    assert "Attempt 2/3" in caplog.text

def test_retry_failure_exceeds_limits(reset_flaky_counter, retry_handler, caplog):
    """Test retry that fails after exceeding max retries."""
    caplog.set_level(logging.WARNING)
    decorated_func = retry_with_exponential_backoff(
        max_retries=2,
        base_delay=0.01,
        error_handler=retry_handler,
        allowed_exceptions=(ConnectionError,),
        error_severity=ErrorSeverity.HIGH # Set final failure severity
    )(lambda: flaky_function(fail_times=3))

    with pytest.raises(ConnectionError, match="Failed on attempt 3"):
        decorated_func()

    assert flaky_call_count == 3
    # Check handler logs for retry warnings (should be MEDIUM severity)
    retry_logs = [e for e in retry_handler.errors if e["severity"] == str(ErrorSeverity.MEDIUM)]
    assert len(retry_logs) == 2
    # Check handler logs for final error
    final_error_logs = [e for e in retry_handler.errors if e["severity"] == str(ErrorSeverity.HIGH)]
    assert len(final_error_logs) == 1
    assert "failed after 2 retries" in final_error_logs[0]["message"]
    assert final_error_logs[0]["exception_type"] == "ConnectionError"

def test_retry_immediate_success(reset_flaky_counter, retry_handler):
    """Test retry when the function succeeds on the first try."""
    decorated_func = retry_with_exponential_backoff(
        max_retries=3, error_handler=retry_handler
    )(lambda: flaky_function(fail_times=0))

    result = decorated_func()
    assert result == "Success on attempt 1"
    assert flaky_call_count == 1
    assert len(retry_handler.errors) == 0 # No errors logged

def test_retry_wrong_exception_type(reset_flaky_counter, retry_handler):
    """Test that retry doesn't happen for unallowed exception types."""
    @retry_with_exponential_backoff(max_retries=3, allowed_exceptions=(ConnectionError,), error_handler=retry_handler)
    def func_raising_value_error():
        global flaky_call_count
        flaky_call_count += 1
        raise ValueError("Not a connection error")

    with pytest.raises(ValueError, match="Not a connection error"):
        func_raising_value_error()

    assert flaky_call_count == 1 # Should fail on first attempt
    # Error handler might log this if configured, but no retries occur
    # Depending on how the app uses it, the ValueError might be logged outside the retry

# Remove the invalid line below
# except Exception as e:
#     print(f"Caught final exception: {e}")
#
# if error_handler_instance:
#     print("--- Error Report ---")
#     print(json.dumps(error_handler_instance.generate_error_report(), indent=2)) 