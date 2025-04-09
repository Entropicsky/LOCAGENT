"""
Implements the Translation Agent using the OpenAI Agents SDK.

Handles translation tasks by configuring an Agent with language-specific instructions
and rulesets, then using a Runner for execution.
Based on Spec 9.2.2 and Agents SDK documentation.
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional

# Import Agents SDK components
from agents import Agent, Runner # Assuming these are the correct imports
import openai # Keep openai import for potential exceptions or underlying client needs

# Assuming error_handler and ruleset_manager are importable
try:
    from smite2_translation.error_handling import ErrorHandler, ErrorCategory, ErrorSeverity
    from smite2_translation.core.ruleset_manager import RulesetManager
except ImportError:
    # Define dummy classes if import fails
    print("!!! WARNING: Failed to import real ErrorHandler/RulesetManager. Using dummy classes for agent. !!!")
    class ErrorHandler:
        def log_error(self, *args, **kwargs): print(f"DUMMY LOG ERROR: {args} {kwargs}") # Print dummy logs
    class ErrorCategory:
        API = "API"
        CONFIGURATION = "CONFIGURATION"
        SYSTEM = "SYSTEM"
        CRITICAL = "CRITICAL"
        INPUT_DATA = "INPUT_DATA"
        RULESET = "RULESET"
        FILE_IO = "FILE_IO"
        UNKNOWN = "UNKNOWN"
    class ErrorSeverity:
        CRITICAL = "CRITICAL"
        ERROR = "ERROR"
        MEDIUM = "MEDIUM"
        HIGH = "HIGH"
        WARNING = "WARNING"
    class RulesetManager:
        def get_ruleset(self): return {}
        def get_supported_languages(self): return []

logger = logging.getLogger(__name__)

class TranslationAgent:
    """Manages translation tasks for a specific target language using the OpenAI Agents SDK."""

    # Base instructions - ruleset added dynamically
    BASE_AGENT_INSTRUCTIONS = """
You are a specialized translation assistant for the video game SMITE 2. Your goal is to provide high-quality, contextually accurate translations from English (enUS) to the target language: {target_language}.

Follow these instructions precisely:
1.  **Adhere to Rules:** Strictly follow the General Rules, Style Guide, and any other specific rules provided below.
2.  **Use Glossary:** Prioritize using the exact translations provided in the Glossary for listed terms.
3.  **Formatting:** Preserve all formatting tags (e.g., <tag>, {{placeholder}}) exactly as they appear in the source text.
4.  **Context:** Use the provided 'Context' field to inform the translation's tone and meaning.
5.  **Output Format:** Respond ONLY with the translated text for the given 'src_enUS' input. Do not include explanations, apologies, or any text other than the translation itself.

**TARGET LANGUAGE:** {target_language}

**RULESET:**
{ruleset_str}
"""

    def __init__(
        self,
        target_language: str,
        ruleset_manager: RulesetManager,
        error_handler: Optional[ErrorHandler] = None,
        model: str = "gpt-4-turbo-preview" # Or another suitable model
    ):
        """Initialize the TranslationAgent.

        Args:
            target_language: The target language code (e.g., 'frFR').
            ruleset_manager: An instance of RulesetManager containing loaded rules.
            error_handler: An instance of ErrorHandler for logging issues.
            model: The OpenAI model to use for the agent.
        """
        self.target_language = target_language
        self.ruleset_manager = ruleset_manager
        self.error_handler = error_handler or ErrorHandler()
        self.model = model
        # No persistent assistant_id needed with Agents SDK's Runner model
        # Check if OPENAI_API_KEY is set, as Runner likely relies on it implicitly
        if not os.getenv("OPENAI_API_KEY"):
             message = "OPENAI_API_KEY environment variable not set. Agent may not function."
             logger.error(message)
             # Log as warning or error? Depends if execution will fail later. Use HIGH for now.
             self.error_handler.log_error(message, ErrorCategory.CONFIGURATION, ErrorSeverity.HIGH)

        logger.info(f"TranslationAgent for {target_language} initialized with model {self.model}.")

    def _construct_prompt_rules(self) -> str:
        """Constructs the ruleset string for the agent instructions."""
        # This function remains largely the same, ensuring it pulls data correctly
        ruleset = self.ruleset_manager.get_ruleset()
        parts = []

        # Add general rules if they exist under that specific key
        general_rules = ruleset.get('general_rules', [])
        if general_rules and isinstance(general_rules, list):
            parts.append("**General Rules:**")
            parts.extend([f"- {rule}" for rule in general_rules])
            parts.append("\n")

        # Add other rule sections (like translation_rules, behavior, etc.)
        for key, value in ruleset.items():
             # Exclude keys already handled or not meant for the prompt
             if key not in ['glossary', 'general_rules', 'target_languages']:
                  if isinstance(value, list) and value:
                      section_title = key.replace('_', ' ').title()
                      parts.append(f"**{section_title}:**")
                      # Append each item in the list (assuming list of rule strings)
                      for item in value:
                           if isinstance(item, str):
                                parts.append(item) # Append rule/content string directly
                      parts.append("\n")

        # Format Glossary
        glossary = ruleset.get('glossary', {})
        if glossary and isinstance(glossary, dict):
            parts.append("**Glossary (Use these exact translations):**")
            # Simple Term: Definition format for prompt clarity
            # Sort glossary for consistent prompt order (optional)
            for term, definition in sorted(glossary.items()):
                parts.append(f"- {term}: {definition}")
            parts.append("\n")

        ruleset_str = "\n".join(parts).strip()
        if not ruleset_str:
            logger.warning(f"No rules or glossary found for language {self.target_language} to include in prompt.")
            return "(No specific rules or glossary provided for this language.)"
        
        # Basic check for potentially excessive length (adjust threshold as needed)
        # This is a rough estimate; actual token count depends on the model
        if len(ruleset_str) > 30000: # Example threshold, ~10k tokens estimate
            logger.warning(f"Generated ruleset string for {self.target_language} is very long ({len(ruleset_str)} chars). May exceed context limits.")
            # TODO: Implement summarization or truncation strategy if needed

        return ruleset_str

    # Removed _create_or_get_assistant - not needed with Agents SDK Runner
    # Removed _run_thread - Runner handles the execution loop

    def translate_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Translates a batch of records using a configured Agent and Runner.

        Args:
            batch: A list of dictionaries, each representing a row from the input CSV.
                   Expected keys: 'Record ID', 'src_enUS', 'Context'.

        Returns:
            A list of dictionaries with 'Record ID' and the translated text under
            a key like 'tgt_frFR' (based on self.target_language).
        """
        translations = []
        translation_key = f"tgt_{self.target_language}"

        # Construct the full ruleset string once for the batch
        try:
            ruleset_str = self._construct_prompt_rules()
        except Exception as e:
             message = f"Failed to construct ruleset string for {self.target_language}. Cannot proceed with batch."
             logger.error(message, exc_info=e)
             self.error_handler.log_error(message, ErrorCategory.SYSTEM, ErrorSeverity.CRITICAL, exception=e)
             return [] # Return empty if rules can't be built

        # Construct the base agent instructions including the ruleset
        agent_instructions = self.BASE_AGENT_INSTRUCTIONS.format(
            target_language=self.target_language,
            ruleset_str=ruleset_str
        )

        # Create the Agent instance (can be reused for the batch)
        agent = Agent(
            name=f"Translator_{self.target_language}",
            instructions=agent_instructions,
            model=self.model
            # No tools needed for this simple translation task
        )

        logger.info(f"Starting translation batch for {self.target_language} with {len(batch)} records.")

        # Process records one by one using Runner.run_sync
        for record in batch:
            record_id = record.get('Record ID')
            source_text = record.get('src_enUS')
            context = record.get('Path', 'N/A')

            if not record_id or not source_text:
                logger.warning(f"Skipping record with missing ID ('{record_id}') or source text ('{source_text}').")
                continue

            logger.info(f"Translating Record ID: {record_id} to {self.target_language}")

            # Construct the user input for the Runner
            user_input = f"Source Text (enUS):\n{source_text}\n\nPath (Context):\n{context}\n\nPlease provide ONLY the {self.target_language} translation."

            try:
                # Run the agent synchronously for this record
                # TODO: Consider Runner.run for async execution if needed later
                result = Runner.run_sync(agent, input=user_input)

                # --- Updated Result Handling ---
                response_text = None
                if result:
                    # Try accessing final_output based on RunResult's likely structure/string representation
                    if hasattr(result, 'final_output') and result.final_output:
                        response_text = str(result.final_output).strip()
                    # Fallback to output_text if final_output didn't work
                    elif hasattr(result, 'output_text') and result.output_text:
                        response_text = str(result.output_text).strip()
                    # As a last resort, try converting the whole result object, assuming __str__ gives the desired text
                    elif str(result).strip(): 
                         response_text = str(result).strip()
                         # Optional: Add a check here if str(result) might contain extra unwanted info
                         # based on how RunResult's __str__ is implemented.
                # --- End Updated Result Handling ---

                if response_text: # Check if we successfully extracted non-empty text
                    translations.append({
                        'Record ID': record_id,
                        translation_key: response_text
                    })
                    logger.debug(f"Received translation for {record_id}: {response_text[:100]}...")
                else:
                    # Updated warning log for clarity
                    logger.warning(f"Could not extract valid translation text for Record ID {record_id}. Raw result: {result}")
                    self.error_handler.log_error(
                        f"Failed to extract valid translation text from agent result for Record ID {record_id}",
                        ErrorCategory.API, ErrorSeverity.HIGH,
                        details={'record_id': record_id, 'raw_result': str(result)} # Log the raw result string
                    )

            except openai.OpenAIError as e:
                message = f"OpenAI API error processing Record ID {record_id}."
                logger.error(message, exc_info=e)
                # Use HIGH severity for OpenAI API errors
                self.error_handler.log_error(message, ErrorCategory.API, ErrorSeverity.HIGH, exception=e)
            except Exception as e:
                 message = f"Unexpected error processing Record ID {record_id}"
                 logger.error(message, exc_info=e)
                 # Use HIGH severity for other unexpected errors during translation
                 self.error_handler.log_error(message, ErrorCategory.SYSTEM, ErrorSeverity.HIGH, exception=e)
            
            # Optional: Add a small delay between requests to avoid rate limits
            # time.sleep(0.5)

        logger.info(f"Finished translating batch for {self.target_language}. Got {len(translations)} translations.")
        return translations 