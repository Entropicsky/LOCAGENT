"""
Manages loading, parsing, and merging of Markdown-based ruleset files.

Based on Spec 9.2.1.
"""

import os
import re
import logging
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict
import copy # Import copy module

# Assuming error_handler is importable
try:
    from smite2_translation.error_handling import ErrorHandler, ErrorCategory, ErrorSeverity
except ImportError:
    # Define dummy classes if import fails (e.g., when running tests without full package install)
    class ErrorHandler:
        def log_error(self, *args, **kwargs): pass
    class ErrorCategory:
        UNKNOWN = None
        INPUT_DATA = None
        RULESET = None
        FILE_IO = None
        SYSTEM = None
        CONFIGURATION = None
    class ErrorSeverity:
        MEDIUM = None
        HIGH = None
        CRITICAL = None
        WARNING = None

logger = logging.getLogger(__name__)

class RulesetManager:
    """Loads, parses, and provides access to translation rulesets from Markdown files."""

    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        """Initialize the RulesetManager.

        Args:
            error_handler: An instance of ErrorHandler for logging issues.
        """
        self.error_handler = error_handler or ErrorHandler()
        # Use a standard dict for the merged ruleset, initialized empty
        self.merged_ruleset: Dict[str, Any] = {}
        self.supported_languages: Set[str] = set()
        logger.info("RulesetManager initialized.")

    def _extract_sections(self, content: str) -> Dict[str, str]:
        """Extracts sections based on H2 headers or simple headers ending in a colon."""
        sections = {}
        # Pattern to find potential header lines
        header_pattern = r"^(?:##\s+(.+?)\s*|([^:\n]+):)\s*$"
        header_matches = list(re.finditer(header_pattern, content, re.MULTILINE))

        if not header_matches:
            logger.warning("No headers found in ruleset content using pattern.")
            # Fallback: Check if the entire content should be treated as a single section?
            # For now, return empty if no headers found.
            return {}

        for i, current_match in enumerate(header_matches):
            title = (current_match.group(1) or current_match.group(2)).strip()
            content_start_pos = current_match.end()

            # Determine the end position for the current section's content
            if i + 1 < len(header_matches):
                # End position is the start of the next header match
                content_end_pos = header_matches[i+1].start()
            else:
                # If it's the last header, content goes to the end of the string
                content_end_pos = len(content)

            # Extract the content between the end of the current header line and the start of the next
            section_content = content[content_start_pos:content_end_pos].strip()

            if title:
                sections[title] = section_content
                logger.debug(f"Extracted section: '{title}'")

        if not sections:
             logger.warning("No sections extracted despite finding headers.") # Should not happen if headers found

        return sections

    def _extract_glossary(self, glossary_content: str) -> Dict[str, str]:
        """Extracts term-definition pairs from the Glossary section content.
        Handles both Markdown table format and simple 'Term: Definition' lines.
        """
        glossary = {}
        # Regex for Markdown table format: | Term | Definition | Notes |
        table_row_pattern = r"^\|\s*(.+?)\s*\|\s*(.+?)\s*\|.*$"
        # Regex for simple "Term: Definition" lines
        simple_def_pattern = r"^\*?\*?(.+?)\*?\*?:\s*(.*?)(?=(?:^\*?\*?.+?\*?\*?:)|\Z)"

        # 1. Process table rows
        found_table = False
        for match in re.finditer(table_row_pattern, glossary_content, re.MULTILINE):
            term = match.group(1).strip()
            definition = match.group(2).strip()
            # Skip table header row
            if term.lower() == 'english' or '-' in term:
                continue
            if term:
                found_table = True
                definition = re.sub(r'\s*\n\s*', ' ', definition)
                glossary[term] = definition
                logger.debug(f"Extracted glossary term (table): '{term}'")

        # 2. Process simple definition lines (run regardless of table presence)
        for match in re.finditer(simple_def_pattern, glossary_content, re.MULTILINE | re.DOTALL):
            # Avoid accidentally matching table content if table was found
            if found_table and '|' in match.group(0):
                 continue

            term = match.group(1).strip()
            definition_raw = match.group(2).strip()
            definition = re.sub(r'\s*\n\s*', ' ', definition_raw)
            # Add/overwrite term
            if term:
                glossary[term] = definition
                logger.debug(f"Extracted glossary term (simple): '{term}'")

        return glossary

    def _parse_ruleset(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Parses a single Markdown ruleset file."""
        logger.info(f"Parsing ruleset file: {file_path}")
        # Use defaultdict temporarily, convert to dict before returning
        parsed_data: Dict[str, Any] = defaultdict(list)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            sections = self._extract_sections(content)
            if not sections:
                 logger.warning(f"No sections extracted from {file_path}. Skipping file.")
                 return None # Indicate failure or empty file

            has_data = False
            for title, section_content in sections.items():
                # Only process if section_content is not empty after stripping
                stripped_content = section_content.strip()
                if not stripped_content:
                    continue # Skip empty sections

                normalized_title = title.lower().strip()
                # Use the normalized key 'glossary' consistently
                if normalized_title in ['glossary', 'game-specific glossary']:
                    glossary_dict = self._extract_glossary(stripped_content)
                    if glossary_dict: # Only assign if not empty
                         # Store glossary using the consistent key 'glossary'
                         parsed_data['glossary'] = glossary_dict
                         has_data = True
                elif normalized_title == 'target languages':
                     langs = [lang.strip() for lang in stripped_content.split('\n') if re.match(r'^[a-z]{2}[A-Z]{2}$', lang.strip())]
                     if langs:
                         parsed_data['target_languages'].extend(langs)
                         has_data = True
                else:
                    # Store ALL other sections by title
                    key = normalized_title.replace(' ', '_')
                    # Special handling for general_rules: split into lines
                    if key == 'general_rules':
                        rules = [rule.strip() for rule in stripped_content.split('\n') if rule.strip()]
                        if rules:
                             parsed_data[key].extend(rules) # Extend list of rules
                             has_data = True
                    # For other sections, store the whole block as a single string in a list
                    elif stripped_content:
                        parsed_data[key].append(stripped_content)
                        has_data = True

            # Convert defaultdict back to dict
            final_parsed_data = dict(parsed_data)

            # Return dict if data was found, otherwise None
            return final_parsed_data if has_data else None

        except FileNotFoundError:
            message = f"Ruleset file not found: {file_path}"
            logger.error(message)
            self.error_handler.log_error(message, ErrorCategory.FILE_IO, ErrorSeverity.CRITICAL)
            return None
        except Exception as e:
            message = f"Error parsing ruleset file {file_path}"
            logger.error(message, exc_info=e)
            self.error_handler.log_error(message, ErrorCategory.RULESET, ErrorSeverity.CRITICAL, exception=e)
            return None

    def _merge_rulesets(self, base_ruleset: Dict[str, Any], new_ruleset: Dict[str, Any]) -> Dict[str, Any]:
        """Merges new ruleset data into the base ruleset.
        New rules overwrite base rules for glossary terms. List items are appended.
        """
        # Start with a shallow copy for non-nested items
        merged = base_ruleset.copy()

        # Handle glossary merge explicitly first
        if 'glossary' in new_ruleset and isinstance(new_ruleset['glossary'], dict):
            # Get a COPY of the base glossary (or empty dict), update it, assign back
            base_glossary_copy = merged.get('glossary', {}).copy()
            if not isinstance(base_glossary_copy, dict): # Should not happen if base was dict or {}
                 logger.warning("Base glossary copy was not a dict, starting fresh.")
                 base_glossary_copy = {}

            new_glossary = new_ruleset['glossary']
            logger.debug(f"Glossary Merge: Base Copy has {len(base_glossary_copy)} items, New has {len(new_glossary)} items.")
            logger.debug(f"Glossary - Base Copy: {base_glossary_copy}")
            logger.debug(f"Glossary - New : {new_glossary}")

            # Update the copy
            base_glossary_copy.update(new_glossary)
            merged['glossary'] = base_glossary_copy # Assign the updated copy back
            logger.debug(f"Glossary - Final: {merged['glossary']}")

        # Handle merging for other keys (lists, simple values)
        for key, value in new_ruleset.items():
            if key == 'glossary': # Already handled
                continue

            if isinstance(value, list):
                # Ensure the base list exists before extending
                # Make a copy if it exists to avoid modifying original if base_ruleset is reused
                base_list = merged.get(key, []).copy()
                if not isinstance(base_list, list):
                     logger.warning(f"Base key '{key}' was not list, creating new list.")
                     base_list = []
                # Append items only if they are not already present
                for item in value:
                    if item not in base_list:
                        base_list.append(item)
                merged[key] = base_list
            else:
                 # Simple overwrite for non-dict/list types
                 merged[key] = value # Overwrite base value with new value
        return merged


    def load_rulesets(self, ruleset_dir: str):
        """Loads and merges all ruleset files from a directory. Prioritizes global_ruleset.md."""
        logger.info(f"Loading rulesets from directory: {ruleset_dir}")
        # Use a standard dict for the merged ruleset, initialized empty
        self.merged_ruleset: Dict[str, Any] = {}
        self.supported_languages = set()

        if not os.path.isdir(ruleset_dir):
            message = f"Ruleset directory not found or is not a directory: {ruleset_dir}"
            logger.error(message)
            self.error_handler.log_error(message, ErrorCategory.FILE_IO, ErrorSeverity.CRITICAL)
            return

        try:
            all_files = [f for f in os.listdir(ruleset_dir) if f.endswith('.md')]
            if not all_files:
                logger.warning(f"No Markdown ruleset files found in {ruleset_dir}")
                return

            # Separate global and other files
            global_file = 'global_ruleset.md'
            other_files = sorted([f for f in all_files if f != global_file])

            # Process global file first if it exists
            files_to_process = []
            global_file_path = os.path.join(ruleset_dir, global_file)
            if global_file in all_files:
                logger.info(f"Processing global ruleset first: {global_file_path}")
                parsed_data = self._parse_ruleset(global_file_path)
                if parsed_data:
                    # Initialize merged_ruleset with global data
                    self.merged_ruleset = self._merge_rulesets({}, parsed_data)
                    logger.debug(f"Initialized with rules from {global_file}")

            # Process other files, merging into the potentially existing global data
            logger.info(f"Processing other ruleset files: {other_files}")
            for filename in other_files:
                file_path = os.path.join(ruleset_dir, filename)
                parsed_data = self._parse_ruleset(file_path)
                if parsed_data: # Only merge if parsing was successful and returned data
                    self.merged_ruleset = self._merge_rulesets(self.merged_ruleset, parsed_data)
                    logger.debug(f"Merged rules from {filename}")

            # Extract supported languages from the final merged ruleset
            target_langs_list = self.merged_ruleset.get('target_languages', [])
            if isinstance(target_langs_list, list):
                 self.supported_languages = set(target_langs_list)
            else:
                 logger.warning("'target_languages' key did not contain a list after merge.")
                 self.supported_languages = set()

            if self.supported_languages:
                logger.info(f"Detected supported languages: {sorted(list(self.supported_languages))}")
            else:
                 logger.warning("No target languages specified in rulesets.")

        except Exception as e:
            message = f"An unexpected error occurred while loading rulesets from {ruleset_dir}"
            logger.error(message, exc_info=e)
            self.error_handler.log_error(message, ErrorCategory.SYSTEM, ErrorSeverity.CRITICAL, exception=e)
            # Reset state in case of partial load failure
            self.merged_ruleset = {}
            self.supported_languages = set()


    def get_ruleset(self) -> Dict[str, Any]:
        """Returns the merged ruleset data."""
        if not self.merged_ruleset:
             logger.warning("Requesting ruleset, but none has been loaded successfully.")
        # Return a copy to prevent modification via reference
        return self.merged_ruleset.copy()

    def get_supported_languages(self) -> List[str]:
        """Returns a list of supported target language codes (e.g., ['frFR', 'deDE'])."""
        if not self.supported_languages:
             logger.warning("Requesting supported languages, but none were found during ruleset loading.")
        return sorted(list(self.supported_languages))

# Example Usage (simplified standalone setup):
if __name__ == '__main__':
    from enum import Enum, auto
    import json

    # Minimal dummy classes for standalone run
    class ErrorCategory(Enum):
        UNKNOWN = auto()
        INPUT_DATA = auto()
        RULESET = auto()
        FILE_IO = auto()
        SYSTEM = auto()
        CONFIGURATION = auto()
    class ErrorSeverity(Enum):
        MEDIUM = auto()
        HIGH = auto()
        CRITICAL = auto()
        WARNING = auto()
    class ErrorHandler:
        def __init__(self):
            self.errors = []
        def log_error(self, msg, cat=None, sev=None, **kwargs):
            self.errors.append({'message': msg, 'category': str(cat), 'severity': str(sev)})
        def generate_error_report(self):
            return self.errors
    logging.basicConfig(level=logging.INFO) # Use INFO level for example output

    handler = ErrorHandler()
    rules_manager = RulesetManager(error_handler=handler)

    DUMMY_RULESET_DIR = './data/rulesets_dummy'
    os.makedirs(DUMMY_RULESET_DIR, exist_ok=True)

    # global.md
    with open(os.path.join(DUMMY_RULESET_DIR, 'global.md'), 'w', encoding='utf-8') as f:
        f.write("""
## Target Languages
frFR
deDE
esES

## General Rules
- Use formal address.
- Do not translate proper nouns like 'Smite'.

## Glossary
Smite: Smite (Do not translate)
Minion: Sbire (frFR), Diener (deDE), Esbirro (esES)
Fire Giant: Géant de feu (frFR), Feuerriese (deDE), Gigante de fuego (esES)
""")

    # french_specific.md
    with open(os.path.join(DUMMY_RULESET_DIR, 'french_specific.md'), 'w', encoding='utf-8') as f:
        f.write("""
## General Rules
- Use French punctuation rules (e.g., space before :).

## Glossary
Tower: Tour
Phoenix: Phénix
Minion: Sbire
""")

    # empty.md (to test handling empty/invalid files)
    with open(os.path.join(DUMMY_RULESET_DIR, 'empty.md'), 'w', encoding='utf-8') as f:
        f.write("No rules here") # No valid H2 sections

    rules_manager.load_rulesets(DUMMY_RULESET_DIR)
    merged = rules_manager.get_ruleset()
    langs = rules_manager.get_supported_languages()

    print("\n--- Merged Ruleset ---")
    print(json.dumps(merged, indent=2, ensure_ascii=False))

    print("\n--- Supported Languages ---")
    print(langs)

    print("\n--- Error Report ---")
    print(json.dumps(handler.generate_error_report(), indent=2, ensure_ascii=False)) 