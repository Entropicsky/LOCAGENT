"""Unit tests for the TranslationAgent class."""

import pytest
import os
from unittest.mock import MagicMock, patch, call

# Import the class to test
from smite2_translation.agents.translation_agent import TranslationAgent

# Import dependencies to mock
from smite2_translation.error_handling import ErrorHandler, ErrorCategory, ErrorSeverity
from smite2_translation.core.ruleset_manager import RulesetManager
import openai # For mocking OpenAIError
from agents import Agent # Add missing import

# Mock the agents module if it's not actually installable in the test environment
# Or ensure the test environment has access to it.
# For now, assume we can patch 'agents.Runner'

# --- Fixtures ---

@pytest.fixture
def mock_error_handler():
    """Provides a mock ErrorHandler instance."""
    return MagicMock(spec=ErrorHandler)

@pytest.fixture
def mock_ruleset_manager():
    """Provides a mock RulesetManager instance with basic data."""
    manager = MagicMock(spec=RulesetManager)
    # Define default return values for methods used by the agent
    manager.get_ruleset.return_value = {
        'general_rules': ['- Rule 1', '- Rule 2'],
        'glossary': {'Hello': 'Bonjour', 'World': 'Monde'},
        'style_guide': ['Style point 1.']
    }
    manager.get_supported_languages.return_value = ['frFR'] # Example
    return manager

@pytest.fixture
def translation_agent(mock_ruleset_manager, mock_error_handler):
    """Provides a TranslationAgent instance with mocked dependencies."""
    # Mock os.getenv temporarily for successful init
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
        agent = TranslationAgent(
            target_language='frFR',
            ruleset_manager=mock_ruleset_manager,
            error_handler=mock_error_handler,
            model="test-model"
        )
    return agent

# Define MockRunResult here or import if defined globally
# For simplicity in unit tests, we can mock the return value directly
class SimpleMockResult:
    def __init__(self, text):
        self.output_text = text

@patch('agents.Runner.run_sync')
def test_translate_batch_success(mock_run_sync, translation_agent, mock_error_handler):
    """Test successful translation of a batch."""
    # Configure the mock Runner.run_sync to return simple objects
    mock_run_sync.side_effect = [
        SimpleMockResult("Bonjour le monde"),
        SimpleMockResult("Au revoir")
    ]

    batch = [
        {'Record ID': '1', 'src_enUS': 'Hello World', 'Context': 'Greeting'},
        {'Record ID': '2', 'src_enUS': 'Goodbye', 'Context': 'Farewell'}
    ]

    translations = translation_agent.translate_batch(batch)

    assert len(translations) == 2
    assert translations[0] == {'Record ID': '1', 'tgt_frFR': 'Bonjour le monde'}
    assert translations[1] == {'Record ID': '2', 'tgt_frFR': 'Au revoir'}

    # Verify run_sync was called twice with the correct Agent and inputs
    assert mock_run_sync.call_count == 2
    
    # Check first call
    first_call_args, first_call_kwargs = mock_run_sync.call_args_list[0]
    assert isinstance(first_call_args[0], Agent) # Check Agent object passed
    assert "Hello World" in first_call_kwargs['input'] # Check input keyword argument
    # Assert context based on 'Path' field or 'N/A' if missing (here it's missing)
    assert "Path (Context):\nN/A" in first_call_kwargs['input']
    
    # Check second call
    second_call_args, second_call_kwargs = mock_run_sync.call_args_list[1]
    assert isinstance(second_call_args[0], Agent)
    assert "Goodbye" in second_call_kwargs['input']
    # Assert context based on 'Path' field or 'N/A' if missing (here it's missing)
    assert "Path (Context):\nN/A" in second_call_kwargs['input']

    # Verify the agent instructions passed to run_sync contain rules
    agent_arg = first_call_args[0]
    assert "Bonjour" in agent_arg.instructions # Check glossary in instructions
    assert "Rule 1" in agent_arg.instructions # Check rules in instructions

    mock_error_handler.log_error.assert_not_called()

@patch('agents.Runner.run_sync')
def test_translate_batch_missing_source(mock_run_sync, translation_agent, mock_error_handler):
    """Test that records with missing source text are skipped."""
    batch = [
        {'Record ID': '1', 'src_enUS': '', 'Context': 'Greeting'}, # Empty source
        {'Record ID': '2', 'src_enUS': 'Valid', 'Context': 'Test'}
    ]
    # Only the second record should trigger a run_sync call
    mock_run_sync.return_value = SimpleMockResult("Valide")

    translations = translation_agent.translate_batch(batch)

    assert len(translations) == 1 # Only the valid record should be processed
    assert translations[0] == {'Record ID': '2', 'tgt_frFR': 'Valide'}
    mock_run_sync.assert_called_once() # Should only be called for the valid record
    mock_error_handler.log_error.assert_not_called() # Skipping is a warning, not error

@patch('agents.Runner.run_sync')
def test_translate_batch_empty_result(mock_run_sync, translation_agent, mock_error_handler):
    """Test handling when the runner returns None or empty final_output."""
    batch = [{'Record ID': '1', 'src_enUS': 'Test', 'Context': 'Test'}]
    # Simulate different empty/invalid results
    mock_run_sync.side_effect = [
        SimpleMockResult(None),         # output_text is None
        SimpleMockResult(""),          # output_text is empty string
        SimpleMockResult("   "),        # output_text is whitespace
        None,                         # run_sync returns None
        object()                      # run_sync returns object without output_text
    ]

    # Test with output_text = None
    translations_none = translation_agent.translate_batch(batch)
    assert len(translations_none) == 0
    # Use ANY to avoid matching exact object string representation inside details
    from unittest.mock import ANY
    mock_error_handler.log_error.assert_called_with(
        "No valid translation output received (check agent result structure or content) for Record ID 1",
        ErrorCategory.API, # Positional category
        ErrorSeverity.HIGH, # Positional severity
        details={'record_id': '1', 'result_received': ANY} # Match details dict with ANY for result
    )
    mock_error_handler.reset_mock()

    # Test with output_text = ""
    translations_empty = translation_agent.translate_batch(batch)
    assert len(translations_empty) == 0
    mock_error_handler.log_error.assert_called_with(
        "No valid translation output received (check agent result structure or content) for Record ID 1",
        ErrorCategory.API,
        ErrorSeverity.HIGH,
        details={'record_id': '1', 'result_received': ANY} # Use ANY here too
    )
    mock_error_handler.reset_mock()

    # Test with output_text = "   "
    translations_ws = translation_agent.translate_batch(batch)
    assert len(translations_ws) == 0 # Whitespace only should also be treated as empty
    mock_error_handler.log_error.assert_called_with(
        "No valid translation output received (check agent result structure or content) for Record ID 1",
        ErrorCategory.API,
        ErrorSeverity.HIGH,
        details={'record_id': '1', 'result_received': ANY} # Use ANY here too
    )
    mock_error_handler.reset_mock()

    # Test with result = None
    translations_result_none = translation_agent.translate_batch(batch)
    assert len(translations_result_none) == 0
    mock_error_handler.log_error.assert_called_with(
        "No valid translation output received (check agent result structure or content) for Record ID 1",
        ErrorCategory.API,
        ErrorSeverity.HIGH,
        details={'record_id': '1', 'result_received': 'None'} # None case is specific string
    )
    mock_error_handler.reset_mock()

    # Test with result = object()
    test_obj = object()
    translations_result_obj = translation_agent.translate_batch(batch)
    assert len(translations_result_obj) == 0
    mock_error_handler.log_error.assert_called_with(
        "No valid translation output received (check agent result structure or content) for Record ID 1",
        ErrorCategory.API,
        ErrorSeverity.HIGH,
        # Use ANY for the result_received in the details for the object() case as well
        details={'record_id': '1', 'result_received': ANY}
    )

@patch('agents.Runner.run_sync')
def test_translate_batch_api_error(mock_run_sync, translation_agent, mock_error_handler):
    """Test handling of OpenAI API errors during batch translation."""
    # Simulate an API error
    batch = [{'Record ID': '1', 'src_enUS': 'Test', 'Context': 'Test'}]
    error_message = "API rate limit exceeded"

    # Create a mock response object that mimics httpx.Response enough for the error init
    mock_response = MagicMock()
    mock_response.request = MagicMock() # Add the expected .request attribute

    mock_run_sync.side_effect = openai.RateLimitError(
        message=error_message, 
        response=mock_response, # Pass the mock response
        body=None
    )

    translations = translation_agent.translate_batch(batch)

    assert len(translations) == 0 # No successful translation
    mock_run_sync.assert_called_once()
    # Check that the API error was logged
    mock_error_handler.log_error.assert_called_once()
    args, kwargs = mock_error_handler.log_error.call_args
    assert "OpenAI API error processing Record ID" in args[0]
    assert args[1] == ErrorCategory.API
    assert args[2] == ErrorSeverity.HIGH
    assert isinstance(kwargs.get('exception'), openai.RateLimitError)

def test_agent_initialization_success(translation_agent, mock_ruleset_manager, mock_error_handler):
    """Test successful initialization of TranslationAgent."""
    assert translation_agent.target_language == 'frFR'
    assert translation_agent.ruleset_manager is mock_ruleset_manager
    assert translation_agent.error_handler is mock_error_handler
    assert translation_agent.model == "test-model"
    # Check that error handler wasn't called for missing key
    mock_error_handler.log_error.assert_not_called()

def test_agent_initialization_no_api_key(mock_ruleset_manager, mock_error_handler):
    """Test initialization fails or warns if OPENAI_API_KEY is not set."""
    # Ensure the key is not in the environment for this test
    with patch.dict(os.environ, {}, clear=True):
        agent = TranslationAgent(
            target_language='frFR',
            ruleset_manager=mock_ruleset_manager,
            error_handler=mock_error_handler
        )
        # Check that the specific error was logged
        mock_error_handler.log_error.assert_called_once()
        args, kwargs = mock_error_handler.log_error.call_args
        assert "OPENAI_API_KEY environment variable not set" in args[0]
        assert args[1] == ErrorCategory.CONFIGURATION
        assert args[2] == ErrorSeverity.HIGH

def test_construct_prompt_rules_basic(translation_agent, mock_ruleset_manager):
    """Test basic construction of the ruleset string for the prompt."""
    rules_str = translation_agent._construct_prompt_rules()

    assert "**General Rules:**" in rules_str
    assert "- - Rule 1" in rules_str # Note the double dash due to list formatting
    assert "- - Rule 2" in rules_str

    assert "**Glossary (Use these exact translations):**" in rules_str
    assert "- Hello: Bonjour" in rules_str
    assert "- World: Monde" in rules_str

    assert "**Style Guide:**" in rules_str # Check other section title
    assert "Style point 1." in rules_str # Check other section content

def test_construct_prompt_rules_empty(translation_agent, mock_ruleset_manager):
    """Test prompt construction when the ruleset manager returns empty data."""
    mock_ruleset_manager.get_ruleset.return_value = {}
    rules_str = translation_agent._construct_prompt_rules()
    expected = "(No specific rules or glossary provided for this language.)"
    assert rules_str == expected

def test_construct_prompt_rules_no_glossary(translation_agent, mock_ruleset_manager):
    """Test prompt construction when only rules are present."""
    mock_ruleset_manager.get_ruleset.return_value = {
        'general_rules': ['- Only Rule 1']
    }
    rules_str = translation_agent._construct_prompt_rules()
    assert "**General Rules:**" in rules_str
    assert "- - Only Rule 1" in rules_str
    assert "**Glossary" not in rules_str
    assert "(No specific rules" not in rules_str # Should not show the empty message

def test_construct_prompt_rules_no_rules(translation_agent, mock_ruleset_manager):
    """Test prompt construction when only glossary is present."""
    mock_ruleset_manager.get_ruleset.return_value = {
        'glossary': {'Term': 'Definition'}
    }
    rules_str = translation_agent._construct_prompt_rules()
    assert "**General Rules:**" not in rules_str
    assert "**Glossary (Use these exact translations):**" in rules_str
    assert "- Term: Definition" in rules_str
    assert "(No specific rules" not in rules_str # Should not show the empty message

# Add more tests for _construct_prompt_rules and translate_batch 