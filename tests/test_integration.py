import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import pandas as pd # Added for DataFrame checks

# Correct project root calculation relative to this file's location
# Assuming tests/ is one level down from the project root (LocAgent)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now import modules from the package
from smite2_translation import main as main_script
from smite2_translation.core.data_processor import DataProcessor # Import the class
from smite2_translation.agents.translation_agent import Agent # Import Agent for isinstance check


# Helper class to mock the structure returned by Runner.run_sync
class MockRunResult:
    """Helper class to mock the structure returned by Runner.run_sync."""
    def __init__(self, content):
        # Simplify the structure - assume the result has a direct attribute for output
        self.output_text = content
        # Remove the old complex structure
        # class MockText:
        #     value = content
        # class MockContent:
        #     text = MockText()
        # class MockMessage:
        #     content = [MockContent()] # Content is a list
        # self.data = [MockMessage()] # data is a list containing the message

# Integration test
@patch('smite2_translation.agents.translation_agent.Runner.run_sync')
@patch('sys.exit') # Patch sys.exit to prevent test runner termination
@patch.object(DataProcessor, 'save_output_csv') # Patch the save method
def test_main_script_flow(mock_save_output, mock_exit, mock_run_sync, tmp_path):
    """Test the main script flow from input loading to output saving."""

    # --- Arrange ---\n    # Mock the API responses
    mock_run_sync.side_effect = [
        MockRunResult("Bonjour"), # Mock translation for "Hello"
        MockRunResult("Monde")    # Mock translation for "World"
    ]

    # Define paths using tmp_path for isolation
    base_data_dir = tmp_path / "data"
    input_dir = base_data_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_file = input_dir / "dummy_input.csv"
    # Update dummy input to match real format: "Record ID","src_enUS","Path" with leading comma in ID
    input_file.write_text('"Record ID","src_enUS","Path"\n,REC001,"Hello","HW.Test.Path1"\n,REC002,"World","HW.Test.Path2"')

    output_dir = tmp_path / "output" # Output will be attempted here

    rules_dir = tmp_path / "rules"
    rules_dir.mkdir(exist_ok=True)
    (rules_dir / "global_ruleset.md").write_text("# Global Rules\n")
    # Add a dummy language-specific ruleset that might be looked for
    (rules_dir / "french_translation_ruleset.md").write_text("lang_code: frFR\n# French Rules\n")


    # Prepare mocked command line arguments using paths within tmp_path
    test_args = [
        "main.py", # Script name (ignored by parse_args)
        "--input", str(input_file),
        "--output", str(output_dir),
        "--languages", "frFR", # Test with one language
        "--rules-dir", str(rules_dir),
        # Add --verbose for potentially more debug info if needed later
        # "--verbose"
    ]

    # --- Act ---\n    # Patch sys.argv and run the main script function
    with patch.object(sys, 'argv', test_args):
        try:
            main_script.main()
        except SystemExit as e:
             # Allow SystemExit if mock_exit was called (e.g., due to errors)
             # Check if it exited with the expected code if necessary
             pass # Or add specific checks on e.code
        except Exception as e:
            # Catch any unexpected exception during main execution
            pytest.fail(f"main() raised an unexpected exception: {type(e).__name__}: {e}")

    # --- Assert ---\n    # Check that Runner.run_sync was called twice (once per record)
    assert mock_run_sync.call_count == 2, f"Expected 2 calls to run_sync, got {mock_run_sync.call_count}"

    # Check first call details (optional but good)
    # Note: The input to the agent might change if main.py adds context from 'Path'
    first_call_args, first_call_kwargs = mock_run_sync.call_args_list[0]
    assert isinstance(first_call_args[0], Agent), "First arg to run_sync should be an Agent instance"
    assert "Hello" in first_call_kwargs['input'], "Input text 'Hello' not found in first run_sync call input"
    # We might later assert that context from 'Path' is also included if main.py is updated


    # Check that save_output_csv was called once
    try:
        mock_save_output.assert_called_once()
    except AssertionError as e:
        pytest.fail(f"DataProcessor.save_output_csv was not called. Did main() exit early? Error log: {e}")


    # Get the arguments passed to the mocked save_output_csv
    call_args, call_kwargs = mock_save_output.call_args

    # --- Updated Argument Assertions ---
    assert 'translations_list' in call_kwargs, "Expected 'translations_list' keyword argument in save_output_csv call"
    saved_translations_list = call_kwargs['translations_list']
    assert isinstance(saved_translations_list, list), "Expected 'translations_list' argument to be a list of dicts"
    assert all(isinstance(item, dict) for item in saved_translations_list), "Items in translations_list should be dicts"
    assert len(saved_translations_list) == 2, f"Expected 2 translation dicts, got {len(saved_translations_list)}"
    # Check content of the translation list (raw output from agent)
    expected_trans_list = [
        {'Record ID': 'REC001', 'tgt_frFR': 'Bonjour'}, # Agent should return cleaned ID and translation key
        {'Record ID': 'REC002', 'tgt_frFR': 'Monde'}
    ]
    # Simple comparison (might need more robust checks for complex dicts/order)
    assert saved_translations_list == expected_trans_list, f"Unexpected translations list. Got: {saved_translations_list}, Expected: {expected_trans_list}"

    assert 'input_df' in call_kwargs, "Expected 'input_df' keyword argument in save_output_csv call"
    saved_input_df = call_kwargs['input_df']
    assert isinstance(saved_input_df, pd.DataFrame), "Expected 'input_df' argument to be a DataFrame"
    # --- End Updated Argument Assertions ---

    assert 'output_file' in call_kwargs, "Expected 'output_file' keyword argument"
    saved_output_path = call_kwargs['output_file']

    # --- Assertions about the input DataFrame content (before merge) ---
    expected_input_columns = ['Record ID', 'src_enUS', 'Path'] # Based on dummy_input.csv
    assert all(col in saved_input_df.columns for col in expected_input_columns), \
        f"Input DataFrame missing expected columns. Got: {saved_input_df.columns.tolist()}, Expected: {expected_input_columns}"
    assert len(saved_input_df) == 2, f"Expected 2 rows in input DataFrame, got {len(saved_input_df)}"
    # Check Record IDs are cleaned in the input DF before passing
    assert saved_input_df.loc[0, 'Record ID'] == 'REC001'
    assert saved_input_df.loc[1, 'Record ID'] == 'REC002'
    assert saved_input_df.loc[0, 'src_enUS'] == 'Hello'
    assert saved_input_df.loc[1, 'src_enUS'] == 'World'

    # --- Assertions on the output file path ---
    # Construct expected output path based on OUTPUT DIR provided in args and input filename
    # Assuming the convention: <output_dir>/<input_filename_base>_translations.csv
    # Correct indices for input file (--input value) and output dir (--output value)
    input_filename_base = Path(test_args[2]).stem # test_args[2] is the input file path
    expected_output_path = Path(test_args[4]) / f"{input_filename_base}_translations.csv" # test_args[4] is the output dir path
    # Compare string representations as save_output_path might be string
    assert saved_output_path == str(expected_output_path), f"Expected save path {expected_output_path}, got {saved_output_path}"

    # Check that sys.exit was not called unexpectedly (e.g., due to handled errors)
    # This depends on whether errors should cause exit or just be logged.
    # If errors are expected to be handled gracefully without exiting, assert not called.
    # If critical errors *should* cause exit, we might need to adjust the mock_exit setup.
    # For now, let's assume graceful handling or mock_exit catches intended exits.
    # mock_exit.assert_not_called() # Enable this if no exit is expected
