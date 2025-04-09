"""Unit tests for the RulesetManager class."""

import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock

from smite2_translation.core.ruleset_manager import RulesetManager
from smite2_translation.error_handling import ErrorHandler, ErrorCategory, ErrorSeverity

# Fixture for ErrorHandler mock
@pytest.fixture
def mock_error_handler():
    """Provides a mock ErrorHandler instance."""
    return MagicMock(spec=ErrorHandler)

# Fixture for RulesetManager instance
@pytest.fixture
def ruleset_manager(mock_error_handler):
    """Provides a RulesetManager instance with a mocked error handler."""
    return RulesetManager(error_handler=mock_error_handler)

# Fixture for creating temporary ruleset files
@pytest.fixture
def temp_ruleset_dir(tmp_path):
    """Creates a temporary directory for ruleset files."""
    rules_dir = tmp_path / "rulesets"
    rules_dir.mkdir()
    return rules_dir

# Helper function to create a ruleset file
@pytest.fixture
def create_ruleset_file(temp_ruleset_dir):
    def _create_file(filename: str, content: str):
        file_path = temp_ruleset_dir / filename
        file_path.write_text(content, encoding='utf-8')
        return file_path
    return _create_file


# --- Test Cases ---

def test_ruleset_manager_initialization(mock_error_handler):
    """Test RulesetManager initializes correctly."""
    manager = RulesetManager(error_handler=mock_error_handler)
    assert manager.error_handler is mock_error_handler
    assert manager.merged_ruleset == {}
    assert manager.supported_languages == set()

def test_load_rulesets_non_existent_dir(ruleset_manager, mock_error_handler):
    """Test loading from a directory that does not exist."""
    non_existent_dir = "./non_existent_rulesets"
    ruleset_manager.load_rulesets(non_existent_dir)

    assert ruleset_manager.merged_ruleset == {}
    assert ruleset_manager.supported_languages == set()
    mock_error_handler.log_error.assert_called_once()
    args, _ = mock_error_handler.log_error.call_args
    assert "Ruleset directory not found" in args[0]
    assert non_existent_dir in args[0]
    assert args[1] == ErrorCategory.FILE_IO
    assert args[2] == ErrorSeverity.CRITICAL

def test_load_rulesets_empty_dir(ruleset_manager, temp_ruleset_dir, mock_error_handler):
    """Test loading from an empty directory."""
    ruleset_manager.load_rulesets(str(temp_ruleset_dir))

    assert ruleset_manager.merged_ruleset == {}
    assert ruleset_manager.supported_languages == set()
    # Expect a warning log, but no error handler call (unless configured otherwise)
    mock_error_handler.log_error.assert_not_called()
    # We could potentially check logger output if needed, but let's skip for now

# Test parsing a single valid file
def test_load_single_valid_ruleset(ruleset_manager, create_ruleset_file, mock_error_handler):
    """Test loading and parsing a single valid ruleset file."""
    content = """
## Target Languages
frFR
deDE

## General Rules
- Rule 1
- Rule 2

## Glossary
TermA: Definition A
TermB: Definition B
Multi Line Term: First line.
  Second line.

## Style Guide
- Style point 1.
- Style point 2.
    """
    ruleset_file = create_ruleset_file("valid_rules.md", content)
    ruleset_manager.load_rulesets(str(ruleset_file.parent))

    mock_error_handler.log_error.assert_not_called() # No errors expected

    # Check supported languages
    expected_langs = ['deDE', 'frFR']
    assert ruleset_manager.get_supported_languages() == expected_langs

    # Check merged ruleset content
    merged_rules = ruleset_manager.get_ruleset()
    assert sorted(merged_rules.get('target_languages', [])) == expected_langs
    assert merged_rules.get('general_rules') == ['- Rule 1', '- Rule 2']

    expected_glossary = {
        'TermA': 'Definition A',
        'TermB': 'Definition B',
        'Multi Line Term': 'First line. Second line.' # Check multiline handling
    }
    assert merged_rules.get('glossary') == expected_glossary

    assert merged_rules.get('style_guide') == ['- Style point 1.\n- Style point 2.'] # Content stored as is

# Test merging multiple files
def test_load_and_merge_rulesets(ruleset_manager, create_ruleset_file, mock_error_handler):
    """Test loading and merging rules from multiple files."""
    # File 1: global_ruleset.md
    content1 = """
## Target Languages
frFR
deDE

## General Rules
- Global Rule 1

## Glossary
TermA: Global Def A
TermC: Global Def C

## Style Guide
- Global Style 1
    """
    create_ruleset_file("global_ruleset.md", content1)

    # File 2: french.md (loaded after global.md due to sorting)
    content2 = """
## Target Languages
esES

## General Rules
- French Rule 1

## Glossary
TermA: French Def A (Override)
TermB: French Def B (New)

## Style Guide
- French Style 1

## French Section
Content specific to French.
    """
    create_ruleset_file("french.md", content2)

    # Load from the directory containing both files
    ruleset_manager.load_rulesets(str(create_ruleset_file("_", "").parent)) # Get parent dir

    mock_error_handler.log_error.assert_not_called()

    # Check supported languages (merged and sorted)
    expected_langs = ['deDE', 'esES', 'frFR']
    assert ruleset_manager.get_supported_languages() == expected_langs

    # Check merged ruleset content
    merged_rules = ruleset_manager.get_ruleset()

    # Target languages list should contain unique, sorted entries
    assert sorted(merged_rules.get('target_languages', [])) == expected_langs

    # General rules should be appended
    # Use set comparison to ignore order for general rules list
    assert set(merged_rules.get('general_rules', [])) == set([
        '- Global Rule 1',
        '- French Rule 1'
    ])

    # Glossary should merge, with french.md overriding TermA
    expected_glossary = {
        'TermA': 'French Def A (Override)', # Overridden
        'TermC': 'Global Def C',
        'TermB': 'French Def B (New)',      # Added
    }
    assert merged_rules.get('glossary') == expected_glossary

    # Style guide content should be appended (as list items)
    assert merged_rules.get('style_guide') == [
        '- Global Style 1',
        '- French Style 1'
    ]

    # Check custom section from french.md
    assert merged_rules.get('french_section') == ['Content specific to French.']

# Test file with sections but no content
def test_load_ruleset_no_content(ruleset_manager, create_ruleset_file, mock_error_handler):
    """Test loading a file with valid sections but no actual rules/terms/langs."""
    content = """
## Target Languages

## General Rules

## Glossary

## Style Guide
    """
    ruleset_file = create_ruleset_file("no_content.md", content)
    ruleset_manager.load_rulesets(str(ruleset_file.parent))

    mock_error_handler.log_error.assert_not_called()
    assert ruleset_manager.get_ruleset() == {} # Should be empty as no actual data parsed
    assert ruleset_manager.get_supported_languages() == []

# Test file with only text (no sections)
def test_load_ruleset_no_sections(ruleset_manager, create_ruleset_file, mock_error_handler):
    """Test loading a file with text but no valid H2 sections."""
    content = "This file has no H2 headers.\nJust some text."
    ruleset_file = create_ruleset_file("no_sections.md", content)
    ruleset_manager.load_rulesets(str(ruleset_file.parent))

    mock_error_handler.log_error.assert_not_called() # Parsing just finds nothing
    assert ruleset_manager.get_ruleset() == {}
    assert ruleset_manager.get_supported_languages() == []

# Test handling of completely empty file
def test_load_empty_file(ruleset_manager, create_ruleset_file, mock_error_handler):
    """Test loading a completely empty .md file."""
    ruleset_file = create_ruleset_file("empty.md", "")
    ruleset_manager.load_rulesets(str(ruleset_file.parent))

    mock_error_handler.log_error.assert_not_called() # Should just parse as empty, no error
    assert ruleset_manager.get_ruleset() == {}
    assert ruleset_manager.get_supported_languages() == []

# Test loading actual project rulesets
def test_load_actual_rulesets(ruleset_manager, temp_ruleset_dir, mock_error_handler):
    """Test loading and merging ALL actual rules files from the project.

    Copies all real *.md rules files to the temp dir to avoid direct dependency
    and checks that no errors occur during loading and merging.
    """
    source_rules_dir = Path('rules')

    # Check if source directory exists
    if not source_rules_dir.is_dir():
         pytest.skip(f"Source rules directory '{source_rules_dir}' not found. Skipping test.")

    # Find all .md files in the source directory
    source_files = list(source_rules_dir.glob('*.md'))

    # Check if source files exist before copying
    if not source_files:
        pytest.skip(f"No *.md files found in '{source_rules_dir}' directory. Skipping test.")

    # Copy all found .md files to the temporary test directory
    try:
        import shutil
        copied_files = []
        for src_file in source_files:
            dest_file = temp_ruleset_dir / src_file.name
            shutil.copy(src_file, dest_file)
            copied_files.append(dest_file.name)
        print(f"\nCopied {len(copied_files)} files for test: {copied_files}") # Print copied files
    except Exception as e:
        pytest.fail(f"Failed to copy ruleset files for testing: {e}")

    # Load rulesets from the temporary directory
    ruleset_manager.load_rulesets(str(temp_ruleset_dir))

    # THE MAIN CHECK: Ensure no critical errors were logged during parsing/merging
    mock_error_handler.log_error.assert_not_called()

    # Optional: Basic sanity checks on the result
    merged_rules = ruleset_manager.get_ruleset()
    assert merged_rules # Ensure the merged result is not empty
    # Check that glossary exists and is likely populated (most files have it)
    assert 'glossary' in merged_rules
    assert isinstance(merged_rules['glossary'], dict)
    assert len(merged_rules['glossary']) > 0
    # Supported languages check might be less reliable if not all files define them
    # assert ruleset_manager.get_supported_languages() # Just check if it's not None/empty list might suffice

# Add more tests here for parsing, merging, glossary extraction, language detection, etc. 