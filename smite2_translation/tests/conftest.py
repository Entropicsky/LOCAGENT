# tests/conftest.py

import pytest
# Add fixtures here as needed, e.g., for mocking API calls or database connections.

# Example mock fixture (can be expanded later)
@pytest.fixture
def mock_openai_client(mocker):
    """Mocks the OpenAI client."""
    mock = mocker.patch('openai.OpenAI', autospec=True)
    # Configure mock responses as needed for tests
    # mock_instance = mock.return_value
    # mock_instance.chat.completions.create.return_value = ...
    return mock

@pytest.fixture
def mock_generate_text(mocker):
    """Mocks the ai.generateText function assumed from Spec 8.5."""
    # Assuming ai.generateText exists and needs mocking
    try:
        # Adjust the path based on actual project structure
        mock = mocker.patch('ai.generateText', autospec=True)
    except ModuleNotFoundError:
        # If 'ai' module isn't directly importable at the root, adjust path
        # Example: mock = mocker.patch('smite2_translation.utils.api_utils.generateText', autospec=True)
        # For now, create a basic mock if the path is uncertain
        mock = mocker.MagicMock(name='generateText')
        # Try patching a plausible location if needed for specific tests later
        # mocker.patch('smite2_translation.utils.api_utils.generateText', return_value=mock)

    # Configure default mock response if needed
    # mock.return_value = mocker.MagicMock(text="Mocked translation")
    return mock 