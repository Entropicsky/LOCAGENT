from enum import Enum, auto
from typing import Dict, List, Any, Optional
import os
import datetime
import json
import logging

# Assuming config is importable
try:
    from smite2_translation import config
except ImportError:
    import config

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Enumeration of error severity levels."""
    CRITICAL = auto()  # System cannot continue, requires immediate attention
    HIGH = auto()      # Major functionality impacted, requires prompt attention
    MEDIUM = auto()    # Some functionality impacted, should be addressed soon
    LOW = auto()       # Minor issue, can be addressed in regular maintenance
    INFO = auto()      # Informational message, not an error

    def __str__(self):
        return self.name

class ErrorCategory(Enum):
    """Enumeration of error categories."""
    INPUT_DATA = auto()        # Issues with input CSV data
    RULESET = auto()           # Issues with ruleset files
    TRANSLATION = auto()       # Issues with translation process
    QUALITY_ASSESSMENT = auto() # Issues with quality assessment
    API = auto()               # Issues with OpenAI API calls
    SYSTEM = auto()            # General system issues
    INTEGRATION = auto()       # Issues with component integration
    CONFIGURATION = auto()     # Issues with setup/config
    FILE_IO = auto()           # Issues related to file input/output
    UNKNOWN = auto()           # Uncategorized errors

    def __str__(self):
        return self.name

class ErrorHandler:
    """Central error handling system for the translation process.

    Logs errors and keeps track of them for reporting.
    Based on Spec 9.3.1.
    """

    def __init__(self):
        """Initialize the error handler."""
        self.errors: List[Dict[str, Any]] = []
        logger.info("ErrorHandler initialized.")

    def log_error(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        recovery_action: Optional[str] = None
    ) -> Dict[str, Any]:
        """Logs an error and stores it.

        Args:
            message: The main error message.
            category: The category of the error.
            severity: The severity level of the error.
            details: Optional dictionary with context-specific details.
            exception: Optional exception object associated with the error.
            recovery_action: Optional description of any recovery action taken.

        Returns:
            A dictionary representing the logged error.
        """
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        error_record = {
            "timestamp": timestamp,
            "message": message,
            "category": str(category),
            "severity": str(severity),
            "details": details or {},
            "recovery_action": recovery_action
        }

        log_message = f"[{category}] [{severity}] {message}"
        if details:
            log_message += f" - Details: {json.dumps(details)}"
        if recovery_action:
            log_message += f" - Recovery: {recovery_action}"

        # Choose logging level based on severity
        log_level = logging.INFO
        if severity == ErrorSeverity.CRITICAL:
            log_level = logging.CRITICAL
        elif severity == ErrorSeverity.HIGH:
            log_level = logging.ERROR
        elif severity == ErrorSeverity.MEDIUM:
            log_level = logging.WARNING
        elif severity == ErrorSeverity.LOW:
            log_level = logging.INFO

        if exception:
            logger.log(log_level, log_message, exc_info=exception)
            error_record["exception_type"] = type(exception).__name__
            error_record["exception_message"] = str(exception)
        else:
            logger.log(log_level, log_message)

        self.errors.append(error_record)
        return error_record

    def get_errors(
        self,
        min_severity: Optional[ErrorSeverity] = None,
        category: Optional[ErrorCategory] = None
    ) -> List[Dict[str, Any]]:
        """Gets a filtered list of recorded errors.

        Args:
            min_severity: Minimum severity level to include.
            category: Specific category to filter by.

        Returns:
            A list of error dictionaries matching the criteria.
        """
        filtered_errors = self.errors
        if category:
            filtered_errors = [e for e in filtered_errors if e["category"] == str(category)]
        if min_severity:
            # Compare enum values directly - Assuming lower value means higher severity (e.g., CRITICAL=1)
            filtered_errors = [e for e in filtered_errors if ErrorSeverity[e["severity"]].value <= min_severity.value]

        return filtered_errors

    def has_critical_errors(self) -> bool:
        """Checks if any critical errors have been logged."""
        return any(e["severity"] == str(ErrorSeverity.CRITICAL) for e in self.errors)

    def generate_error_report(self) -> Dict[str, Any]:
        """Generates a summary report of all logged errors.

        Returns:
            A dictionary containing error counts by severity and category.
        """
        report = {
            "total_errors": len(self.errors),
            "errors_by_severity": {str(s): 0 for s in ErrorSeverity},
            "errors_by_category": {str(c): 0 for c in ErrorCategory},
            "critical_errors_present": self.has_critical_errors()
        }
        for error in self.errors:
            report["errors_by_severity"][error["severity"]] += 1
            report["errors_by_category"][error["category"]] += 1

        logger.info(f"Generated error report: {report}")
        return report

    def clear_errors(self):
        """Clears all recorded errors."""
        self.errors = []
        logger.info("Cleared all recorded errors.")

# Example instantiation (usually done once in the main application)
# error_handler_instance = ErrorHandler() 