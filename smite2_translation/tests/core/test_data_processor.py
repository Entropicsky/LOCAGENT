import pytest
import pandas as pd
from pathlib import Path
import logging # Added import
from unittest.mock import MagicMock, call

from smite2_translation.core.data_processor import DataProcessor
from smite2_translation.error_handling import ErrorHandler, ErrorCategory, ErrorSeverity # Corrected import path

# Fixture for ErrorHandler
@pytest.fixture
def error_handler():
    # return ErrorHandler() # Use real handler if testing interactions
    return MagicMock(spec=ErrorHandler) # Use mock for isolated unit tests

@pytest.fixture
def processor(error_handler):
    """Provides a DataProcessor instance with a mocked error handler."""
    return DataProcessor(error_handler=error_handler)

# Fixture for temporary CSV file creation
@pytest.fixture
def temp_csv_file(tmp_path):
    """Factory fixture to create temporary CSV files."""
    def _create_csv(filename, data, encoding='utf-8', index=False):
        file_path = tmp_path / filename
        if isinstance(data, dict):
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=index, encoding=encoding)
        elif isinstance(data, list): # Handle list of dicts
            if data: # If list is not empty
                 df = pd.DataFrame(data)
                 df.to_csv(file_path, index=index, encoding=encoding)
            else: # Handle empty list case (write headers only or empty file)
                 # Write empty file or just headers depending on test needs
                 # This part might need adjustment based on specific test case
                 # Example: write headers if columns are known
                 # pd.DataFrame(columns=['Record ID', 'src_enUS', 'Context']).to_csv(file_path, index=False, encoding=encoding)
                 file_path.touch() # Creates an empty file
        else:
            raise TypeError("Unsupported data type for CSV creation")

        return str(file_path) # Return path as string
    return _create_csv


# --- Test Cases ---

def test_data_processor_initialization(error_handler):
    """Test DataProcessor initializes correctly with an error handler."""
    processor = DataProcessor(error_handler)
    assert processor.error_handler is error_handler

def test_load_valid_csv(processor, temp_csv_file):
    """Test loading a valid CSV file."""
    data = {
        'Record ID': ['1', '2'],
        'src_enUS': ['Hello', 'World'],
        'Context': ['Greeting', 'General']
    }
    file_path = temp_csv_file("valid.csv", data)
    loaded_data = processor.load_input_csv(file_path)

    expected_data = [
        {'Record ID': '1', 'src_enUS': 'Hello', 'Context': 'Greeting'},
        {'Record ID': '2', 'src_enUS': 'World', 'Context': 'General'}
    ]
    # Convert loaded data (potentially with numpy types) to standard python types for comparison
    loaded_data_pythonic = pd.DataFrame(loaded_data).astype(object).where(pd.notnull(pd.DataFrame(loaded_data)), None).to_dict('records')

    assert loaded_data_pythonic == expected_data
    processor.error_handler.log_error.assert_not_called()

def test_load_missing_file(processor):
    """Test loading a non-existent CSV file."""
    file_path = "non_existent_file.csv"
    loaded_data = processor.load_input_csv(file_path)
    assert loaded_data == []
    # Check if log_error was called with expected arguments
    processor.error_handler.log_error.assert_called_once()
    args, kwargs = processor.error_handler.log_error.call_args
    # Check positional arguments for message, category, severity
    assert file_path in args[0] # Check message content
    assert args[1] == ErrorCategory.INPUT_DATA # Check category (position 1)
    assert args[2] == ErrorSeverity.CRITICAL  # Check severity (position 2)

# Updated test_load_empty_csv to use caplog
def test_load_empty_csv(processor, temp_csv_file, caplog):
    """Test loading an empty CSV file (only headers or completely empty)."""
    caplog.set_level(logging.WARNING) # Ensure warnings are captured

    # File with only headers
    headers_only_path = temp_csv_file("empty_headers.csv", [])
    # Manually write headers because DataFrame([]) doesn't write anything
    with open(headers_only_path, 'w', encoding='utf-8') as f:
        f.write("Record ID,src_enUS,Context\n")

    loaded_data_headers = processor.load_input_csv(str(headers_only_path))
    assert loaded_data_headers == []
    # Expect a WARNING log for files with only headers (via logger, not error_handler)
    assert any(rec.levelname == 'WARNING' and "contains only headers or is empty" in rec.message for rec in caplog.records)
    processor.error_handler.log_error.assert_not_called() # Ensure error handler wasn't called for this case

    caplog.clear() # Clear logs for the next check
    processor.error_handler.reset_mock() # Reset mock for the next part

    # Completely empty file
    empty_file_path = temp_csv_file("completely_empty.csv", []) # Creates empty file via touch()
    loaded_data_empty = processor.load_input_csv(str(empty_file_path))
    assert loaded_data_empty == []
    # Check that the specific ERROR was logged via the error_handler for truly empty files
    processor.error_handler.log_error.assert_called_once()
    args, kwargs = processor.error_handler.log_error.call_args
    # Check positional arguments
    assert "Input file is empty" in args[0]
    assert args[1] == ErrorCategory.INPUT_DATA
    assert args[2] == ErrorSeverity.HIGH


def test_load_missing_columns(processor, temp_csv_file):
    """Test loading CSV missing required columns."""
    data = {'Record ID': ['1'], 'SomeOtherColumn': ['Data']}
    file_path = temp_csv_file("missing_cols.csv", data)
    loaded_data = processor.load_input_csv(file_path)
    assert loaded_data == []
    processor.error_handler.log_error.assert_called_once()
    args, kwargs = processor.error_handler.log_error.call_args
    assert "Missing required columns" in args[0]
    assert "src_enUS" in args[0] # Check if the missing column name is mentioned
    # Check positional arguments
    assert args[1] == ErrorCategory.INPUT_DATA
    assert args[2] == ErrorSeverity.CRITICAL

# Updated test_load_csv_encoding
def test_load_csv_encoding(processor, tmp_path, caplog):
    """Test loading CSV with non-utf8 encoding and invalid bytes."""
    caplog.set_level(logging.WARNING) # Capture warnings for fallback

    # 1. Test a file that *is* valid latin-1 (but assume we only want UTF-8)
    # This test is now less relevant as we removed the fallback, but keep structure
    data_latin1 = {
        'Record ID': ['1'],
        'src_enUS': ['Héllö Wörld']
    }
    file_path_latin1 = tmp_path / "latin1_encoded.csv"
    df_latin1 = pd.DataFrame(data_latin1)
    df_latin1.to_csv(file_path_latin1, index=False, encoding='latin-1')

    error_handler_latin1 = MagicMock(spec=ErrorHandler)
    processor_latin1 = DataProcessor(error_handler=error_handler_latin1)
    loaded_latin1 = processor_latin1.load_input_csv(str(file_path_latin1))
    # Expect loading to fail because it's not UTF-8
    assert loaded_latin1 == []
    error_handler_latin1.log_error.assert_called_once()
    args_l1, kwargs_l1 = error_handler_latin1.log_error.call_args
    assert "UTF-8 decoding failed" in args_l1[0]
    assert isinstance(kwargs_l1.get('exception'), UnicodeDecodeError)
    assert args_l1[1] == ErrorCategory.INPUT_DATA
    assert args_l1[2] == ErrorSeverity.CRITICAL

    caplog.clear() # Clear logs for the next check

    # 2. Test handling of truly invalid UTF-8 bytes
    file_path_invalid = tmp_path / "invalid_utf8.csv"
    with open(file_path_invalid, 'wb') as f:
        # Write headers in UTF-8, then an invalid UTF-8 byte sequence
        f.write(b"Record ID,src_enUS\n")
        f.write(b"1,Invalid\x80Data\n") # x80 is an invalid start byte in UTF-8

    error_handler_invalid = MagicMock(spec=ErrorHandler)
    processor_invalid = DataProcessor(error_handler=error_handler_invalid)
    loaded_invalid = processor_invalid.load_input_csv(str(file_path_invalid))

    # Expect loading to fail due to decoding error
    assert loaded_invalid == []
    # Check that the specific decoding error was logged via the error handler
    processor_invalid.error_handler.log_error.assert_called_once()
    args, kwargs = processor_invalid.error_handler.log_error.call_args
    assert "UTF-8 decoding failed" in args[0]
    assert isinstance(kwargs.get('exception'), UnicodeDecodeError)
    assert args[1] == ErrorCategory.INPUT_DATA
    assert args[2] == ErrorSeverity.CRITICAL


def test_save_valid_output(processor, temp_csv_file, tmp_path):
    """Test saving processed data correctly merges with input."""
     # Setup: Create a valid input CSV first
    input_data_dict = {
        'Record ID': ['1', '2'],
        'src_enUS': ['Hello', 'World'],
        'Context': ['Greeting', 'General']
    }
    input_file_path = temp_csv_file("input_for_save.csv", input_data_dict)

    # Load the input data using the processor to get the list of dicts
    input_data_list = processor.load_input_csv(input_file_path)
    # Convert the loaded list of dicts to a DataFrame
    input_df = pd.DataFrame(input_data_list)
    # Ensure Record ID is string in the DataFrame for merge
    if 'Record ID' in input_df.columns:
        input_df['Record ID'] = input_df['Record ID'].astype(str)
    else:
        # Handle case where input might be invalid (though load should prevent)
        input_df = pd.DataFrame() # Or raise error if input must be valid

    # Simulate translations
    translations = [
        # Ensure Record IDs are strings here too for matching
        {'Record ID': '1', 'tgt_frFR': 'Bonjour', 'tgt_deDE': None}, # Use None for missing data
        {'Record ID': '2', 'tgt_frFR': 'Monde', 'tgt_deDE': 'Welt'}
    ]
    output_file = tmp_path / "output.csv"

    # Pass the input DataFrame to the save function
    success = processor.save_output_csv(translations, str(output_file), input_df=input_df)

    assert success is True
    assert output_file.exists()

    # Read the output file and verify content
    # Read with keep_default_na=False and na_values='' to handle empty strings vs actual NaN - Removed na_values
    # Ensure Record ID is read as string for comparison
    output_df = pd.read_csv(output_file, dtype={'Record ID': str}, keep_default_na=False)

    assert 'Record ID' in output_df.columns
    assert 'src_enUS' in output_df.columns
    assert 'Context' in output_df.columns
    assert 'tgt_frFR' in output_df.columns
    assert 'tgt_deDE' in output_df.columns
    assert len(output_df) == 2

    # Check specific values
    # Convert to dicts for easier comparison, handling potential NaN
    # Replace NaN/NaT with None before comparison - Removed this step as fillna('') handles it
    output_records = output_df.to_dict('records') # Convert directly after reading

    expected_records = [
        # Expect empty string for tgt_deDE due to fillna('') in save method
        {'Record ID': '1', 'src_enUS': 'Hello', 'Context': 'Greeting', 'tgt_frFR': 'Bonjour', 'tgt_deDE': ''},
        {'Record ID': '2', 'src_enUS': 'World', 'Context': 'General', 'tgt_frFR': 'Monde', 'tgt_deDE': 'Welt'}
    ]
    assert output_records == expected_records


def test_save_output_no_input(processor, tmp_path):
    """Test saving without providing input data (should only save translations)."""
    translations = [{'Record ID': '1', 'tgt_frFR': 'Bonjour'}]
    output_file = tmp_path / "output_no_input.csv"

    # Pass None for input_df (using the correct keyword)
    success = processor.save_output_csv(translations, str(output_file), input_df=None)

    assert success is True
    assert output_file.exists()
    output_df = pd.read_csv(output_file, dtype={'Record ID': str})
    assert list(output_df.columns) == ['Record ID', 'tgt_frFR']
    assert len(output_df) == 1
    assert output_df['Record ID'][0] == '1'
    assert output_df['tgt_frFR'][0] == 'Bonjour'
    processor.error_handler.log_error.assert_not_called()

# Updated test_save_output_creates_directory
def test_save_output_creates_directory(processor, tmp_path, temp_csv_file):
    """Test that saving creates the output directory if it doesn't exist."""
    # Setup: Create a valid input CSV first
    input_data_dict = {
        'Record ID': ['1'],
        'src_enUS': ['Hello'],
    }
    input_file_path = temp_csv_file("input_for_dir_creation.csv", input_data_dict)
    # Load input as list of dicts and convert to DataFrame
    input_data_list = processor.load_input_csv(input_file_path)
    input_df = pd.DataFrame(input_data_list)
    if 'Record ID' in input_df.columns:
        input_df['Record ID'] = input_df['Record ID'].astype(str)
    else:
        input_df = pd.DataFrame()

    translations = [{'Record ID': '1', 'tgt_frFR': 'Bonjour'}]
    output_dir = tmp_path / "new_output_dir"
    output_file = output_dir / "output.csv"

    assert not output_dir.exists() # Directory should not exist yet
    # Pass the DataFrame using the correct keyword
    success = processor.save_output_csv(translations, str(output_file), input_df=input_df)

    assert success is True
    assert output_dir.exists() # Directory should now exist
    assert output_file.exists() # File should exist within the new directory

    # Verify content briefly
    output_df = pd.read_csv(output_file)
    assert 'tgt_frFR' in output_df.columns
    assert 'src_enUS' in output_df.columns # Check if input columns are merged
    assert output_df['tgt_frFR'][0] == 'Bonjour'


def test_save_output_directory_error(processor, tmp_path):
    """Test saving fails gracefully if directory creation fails (e.g., permissions)."""
    # Use an actual ErrorHandler for this test to check its state
    real_error_handler = ErrorHandler()
    processor_with_real_handler = DataProcessor(error_handler=real_error_handler)

    translations = [{'Record ID': '1', 'tgt_frFR': 'Bonjour'}]
    # Make the target directory's base a file, causing save to fail
    invalid_output_dir_base = tmp_path / "existing_file.txt"
    invalid_output_dir_base.touch() # Create it as a file
    output_file = invalid_output_dir_base / "output.csv"

    # Call the function and expect it to fail and log the error (using correct keyword)
    success = processor_with_real_handler.save_output_csv(translations, str(output_file), input_df=None)

    assert success is False

    # Check that the error handler logged the OSError
    assert len(real_error_handler.errors) == 1
    logged_error = real_error_handler.errors[0]
    assert "Failed to save output CSV file" in logged_error['message']
    assert "OS error" in logged_error['message']
    assert logged_error['category'] == str(ErrorCategory.FILE_IO)
    assert logged_error['severity'] == str(ErrorSeverity.CRITICAL)
    # Check the stored exception type name instead of the raw object
    assert logged_error.get('exception_type') == 'OSError'