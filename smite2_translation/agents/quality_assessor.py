"""
Quality Assessment Agent for the Smite 2 Translation System.

This module defines the QualityAssessmentAgent which evaluates translations
based on language-specific rulesets. It performs checks for formatting,
terminology consistency, and other quality criteria.
"""
import logging
from typing import Dict, Any, List

from smite2_translation.utils.qa_tools import check_formatting, check_terminology

logger = logging.getLogger(__name__)

class QualityAssessmentAgent:
    """
    Agent responsible for assessing the quality of translations based on
    predefined rulesets.
    
    The QualityAssessmentAgent evaluates translations against language-specific
    rules and glossaries to ensure formatting correctness, terminology consistency,
    and adherence to language-specific conventions.
    
    Quality issues are reported as structured data that can be used by the
    translation system to:
    1. Generate reports on translation quality
    2. Trigger auto-retry mechanisms for critical issues
    3. Provide specific feedback for manual review
    
    Current capabilities include:
    - Formatting checks (tags, placeholders, punctuation)
    - Terminology verification against glossaries
    - Detection of missing or altered content
    
    Issues are categorized by type and severity to allow for prioritized handling.
    """

    def __init__(self, rulesets: Dict[str, Dict[str, Any]]):
        """
        Initializes the QualityAssessmentAgent.

        Args:
            rulesets: A dictionary where keys are language codes (e.g., 'frFR')
                      and values are the corresponding ruleset dictionaries.
                      Each ruleset should contain rules and glossaries specific
                      to that language.
        """
        self.rulesets = rulesets
        logger.info(f"QualityAssessmentAgent initialized with rulesets for languages: {list(rulesets.keys())}")

    def assess_quality(
        self,
        source_text: str,
        target_text: str,
        target_language: str,
        record_id: str | None = None # Optional identifier for logging
    ) -> List[Dict[str, str]]:
        """
        Assesses the quality of a given translation against the ruleset
        for the specified target language.
        
        This method performs a series of quality checks on the translation:
        1. Formatting checks: Verifies that formatting elements (tags, placeholders)
           are preserved correctly
        2. Terminology checks: Ensures that glossary terms are translated consistently
        
        Each detected issue is reported with information about:
        - Type: The category of issue (FORMAT_ERROR, TERMINOLOGY_ERROR, etc.)
        - Error: A brief description of the issue
        - Details: Specific information about what's wrong and how to fix it
        - Severity: How critical the issue is (if applicable)
        
        Args:
            source_text: The original English source text.
            target_text: The translated text to assess.
            target_language: The target language code (e.g., 'frFR').
            record_id: An optional identifier for the text record (for logging).

        Returns:
            A list of dictionaries, where each dictionary represents a detected
            quality issue with keys for 'type', 'error', 'details', and optionally 'severity'.
            
        Example:
            [
                {
                    "type": "FORMAT_ERROR",
                    "error": "Missing placeholder",
                    "details": "Placeholder {COUNT} is missing from translation",
                    "severity": "CRITICAL"
                },
                {
                    "type": "TERMINOLOGY_ERROR",
                    "error": "Inconsistent translation",
                    "details": "Term 'Ability' should be translated as 'Fähigkeit'",
                    "severity": "HIGH"
                }
            ]
        """
        log_prefix = f"[Record {record_id}] " if record_id else ""
        logger.debug(f"{log_prefix}Starting quality assessment for language: {target_language}")

        all_errors: List[Dict[str, str]] = [] 

        ruleset = self.rulesets.get(target_language)
        if not ruleset:
            logger.warning(f"{log_prefix}No ruleset found for language '{target_language}'. Skipping assessment.")
            return all_errors # Return empty list if no rules apply
        
        # 1. Perform Formatting Checks
        try:
            formatting_errors = check_formatting(source_text, target_text, ruleset, target_language)
            if formatting_errors:
                logger.debug(f"{log_prefix}Found {len(formatting_errors)} formatting errors.")
                all_errors.extend(formatting_errors)
        except Exception as e:
            logger.error(f"{log_prefix}Error during formatting check: {e}", exc_info=True)
            all_errors.append({"type": "QA_ERROR", "error": "Exception during formatting check.", "details": str(e)})

        # 2. Perform Terminology Checks
        glossary = ruleset.get('glossary', {})
        if glossary:
            try:
                terminology_errors = check_terminology(source_text, target_text, glossary, target_language)
                if terminology_errors:
                    logger.debug(f"{log_prefix}Found {len(terminology_errors)} terminology errors.")
                    all_errors.extend(terminology_errors)
            except Exception as e:
                logger.error(f"{log_prefix}Error during terminology check: {e}", exc_info=True)
                all_errors.append({"type": "QA_ERROR", "error": "Exception during terminology check.", "details": str(e)})
        else:
            logger.debug(f"{log_prefix}No glossary found in ruleset for {target_language}. Skipping terminology check.")


        logger.info(f"{log_prefix}Quality assessment complete for language {target_language}. Found {len(all_errors)} total issues.")
        return all_errors 