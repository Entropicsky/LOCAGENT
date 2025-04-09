from typing import Dict, List, Any, Optional
import pandas as pd
import os
import logging

# Assuming error_handler is importable
try:
    from smite2_translation.error_handling import ErrorHandler, ErrorCategory, ErrorSeverity
except ImportError:
    # Define dummy classes if import fails (e.g., when running tests without full package install)
    class ErrorHandler:
        def log_error(self, *args, **kwargs): pass
    class ErrorCategory: UNKNOWN = None; INPUT_DATA = None; SYSTEM = None; FILE_IO = None
    class ErrorSeverity: MEDIUM = None; HIGH = None; CRITICAL = None

logger = logging.getLogger(__name__)

class DataProcessor:
    """Processes input and output CSV data.

    Handles loading, validation, and saving of translation data.
    Based on Spec 4 and Spec 9.2.3.
    """

    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        """Initialize the data processor.

        Args:
            error_handler: An instance of ErrorHandler for logging issues.
        """
        self.error_handler = error_handler or ErrorHandler() # Use a default if None
        logger.info("DataProcessor initialized.")

    def load_input_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """Loads and validates input CSV file.

        Args:
            file_path: Path to the input CSV file.

        Returns:
            A list of dictionaries, where each dictionary represents a valid row.
            Returns an empty list if the file cannot be read or is entirely invalid.
        """
        logger.info(f"Attempting to load input CSV: {file_path}")
        if not os.path.exists(file_path):
            message = f"Input file not found: {file_path}"
            logger.error(message)
            self.error_handler.log_error(message, ErrorCategory.INPUT_DATA, ErrorSeverity.CRITICAL)
            return []

        try:
            # Attempt to read with standard UTF-8 first
            try:
                # Ensure Record ID is read as string consistently
                df = pd.read_csv(file_path, dtype={'Record ID': str}, keep_default_na=False)
            except UnicodeDecodeError as ude:
                # If UTF-8 fails, log critical error and stop. Do not fallback.
                message = f"UTF-8 decoding failed for {file_path}. File may have incorrect encoding."
                logger.error(message, exc_info=ude)
                self.error_handler.log_error(
                    message, ErrorCategory.INPUT_DATA, ErrorSeverity.CRITICAL, exception=ude
                )
                return []
            except pd.errors.EmptyDataError:
                # Handle files that are truly empty (no headers, no data)
                message = f"Input file is empty: {file_path}"
                logger.error(message) # Log as error as it likely prevents processing
                self.error_handler.log_error(message, ErrorCategory.INPUT_DATA, ErrorSeverity.HIGH)
                return []

            # Check if DataFrame is empty *after* successful read (e.g., only headers)
            if df.empty:
                logger.warning(f"Input file contains only headers or is empty: {file_path}")
                # Optionally log via handler, depends on whether this is an error or just needs skipping
                # self.error_handler.log_error(f"Input file contains only headers: {file_path}", ErrorCategory.INPUT_DATA, ErrorSeverity.WARNING)
                return []

            # --- Validation (Spec 4.2) ---
            required_columns = ['Record ID', 'src_enUS']
            valid_rows = []
            processed_ids = set()

            # Check for required columns
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                message = f"Missing required columns in {file_path}: {', '.join(missing_cols)}"
                logger.error(message)
                self.error_handler.log_error(message, ErrorCategory.INPUT_DATA, ErrorSeverity.CRITICAL)
                return [] # Cannot proceed without required columns

            # --- Data Cleaning: Strip leading comma from Record ID ---
            if 'Record ID' in df.columns:
                # Ensure it's string type first for reliable string operations
                df['Record ID'] = df['Record ID'].astype(str)
                # Apply the strip operation using .str accessor
                df['Record ID'] = df['Record ID'].str.lstrip(',')
                logger.debug("Stripped leading commas from 'Record ID' column.")
            # --- End Data Cleaning ---

            # Ensure Record ID is string type for consistent processing (redundant after cleaning, but safe)
            df['Record ID'] = df['Record ID'].astype(str)

            # Iterate and validate rows
            for index, row in df.iterrows():
                record_id = row['Record ID'] # Already string and cleaned
                src_text = str(row['src_enUS']) # Ensure source text is also string
                is_valid = True
                error_messages = []

                # Check for empty required fields
                if not record_id:
                    is_valid = False
                    error_messages.append("Record ID is empty")
                if not src_text:
                    is_valid = False
                    error_messages.append("src_enUS is empty")

                # Check for duplicate Record IDs
                if record_id and record_id in processed_ids:
                    is_valid = False
                    error_messages.append(f"Duplicate Record ID found: {record_id}")
                elif record_id:
                    processed_ids.add(record_id)

                if is_valid:
                    # Ensure all values in the row dict are standard python types (esp. strings)
                    row_dict = {k: str(v) if pd.notna(v) else '' for k, v in row.to_dict().items()}
                    valid_rows.append(row_dict)
                else:
                    message = f"Skipping invalid row {index + 2} in {file_path}: {'; '.join(error_messages)}"
                    logger.warning(message)
                    self.error_handler.log_error(
                        message, ErrorCategory.INPUT_DATA, ErrorSeverity.MEDIUM,
                        details={"row_index": index + 2, "row_data": row.to_dict()}
                    )

            logger.info(f"Loaded {len(valid_rows)} valid rows from {file_path}")
            return valid_rows

        except Exception as e:
            # Catch-all for other unexpected errors during load/validation
            message = f"Failed to load or process CSV file {file_path}"
            logger.error(message, exc_info=True)
            self.error_handler.log_error(
                message, ErrorCategory.INPUT_DATA, ErrorSeverity.CRITICAL, exception=e
            )
            return []

    def save_output_csv(
        self,
        translations_list: List[Dict[str, Any]],
        output_file: str,
        input_df: Optional[pd.DataFrame] = None
    ) -> bool:
        """Saves translations to output CSV file, optionally merging with input data.

        Args:
            translations_list: A list of dictionaries, each containing at least
                               'Record ID' and translated columns like 'tgt_frFR'.
            output_file: Path to the output CSV file.
            input_df: The original DataFrame of input row dictionaries (used to preserve columns and order).
                      If None or empty, only translations are saved.

        Returns:
            True if saving was successful, False otherwise.
        """
        logger.info(f"Preparing to save output CSV: {output_file}")

        if not translations_list:
            message = "Cannot save output CSV: No translation data provided."
            logger.warning(message)
            # Don't log as error, just return False as nothing to save
            # self.error_handler.log_error(message, ErrorCategory.SYSTEM, ErrorSeverity.MEDIUM)
            return False

        try:
            # Convert to DataFrame for easier merging
            trans_df = pd.DataFrame(translations_list)

            # Check if 'Record ID' exists in the translations DataFrame
            if 'Record ID' not in trans_df.columns:
                message = "The 'Record ID' column is missing from the translation data."
                logger.error(message)
                self.error_handler.log_error(message, ErrorCategory.DATA_VALIDATION, ErrorSeverity.CRITICAL)
                return False

            # Ensure 'Record ID' is string type before aggregation
            trans_df['Record ID'] = trans_df['Record ID'].astype(str)

            # Aggregate translations: group by 'Record ID' and take the first non-null value for each column
            # This handles cases where multiple translations might exist for the same ID (e.g., from different batches)
            # We assume the first valid translation encountered is the desired one.
            logger.debug(f"Aggregating translations by 'Record ID'. Original count: {len(trans_df)}")
            trans_df = trans_df.groupby('Record ID', as_index=False).first()
            logger.debug(f"Aggregated translation count: {len(trans_df)}")

            output_df = None
            final_column_order = []

            if input_df is not None and not input_df.empty:
                logger.debug("Merging translations with provided input data.")
                # Check if 'Record ID' exists in the input DataFrame
                if 'Record ID' not in input_df.columns:
                    message = "The 'Record ID' column is missing from the input DataFrame."
                    logger.error(message)
                    self.error_handler.log_error(message, ErrorCategory.DATA_VALIDATION, ErrorSeverity.CRITICAL)
                    return False # Cannot merge without the key

                # Ensure 'Record ID' is string type in input_df as well for merging
                input_df['Record ID'] = input_df['Record ID'].astype(str)

                # Define the desired final column order based on input + new translation cols
                original_input_cols = list(input_df.columns)
                # Find translation columns not already in input (e.g., 'Translation_frFR')
                # Exclude 'Record ID' itself from the list of new columns
                new_translation_cols = [
                    col for col in trans_df.columns
                    if col not in original_input_cols and col != 'Record ID'
                ]
                final_column_order = original_input_cols + new_translation_cols
                logger.debug(f"Final column order determined: {final_column_order}")

                # Perform a left merge: keep all rows from input_df, add matching translations
                # Suffixes handle potential column name clashes beyond 'Record ID' if any (unlikely here)
                output_df = pd.merge(input_df, trans_df, on='Record ID', how='left', suffixes=('', '_trans'))

                # Ensure all required columns are present, handling potential merge issues
                # Reindex ensures columns exist even if no match was found during merge
                output_df = output_df.reindex(columns=final_column_order)

            else:
                # No input data, just save the aggregated translations
                logger.debug("Saving aggregated translations only (no input data provided).")
                output_df = trans_df
                # Ensure Record ID is string type (redundant check, but safe)
                output_df['Record ID'] = output_df['Record ID'].astype(str)
                final_column_order = list(trans_df.columns) # Use translation column order
                logger.debug(f"Final column order (translations only): {final_column_order}")

            # Ensure final DataFrame exists and has the correct columns
            if output_df is None:
                 message = "Failed to create output DataFrame."
                 logger.error(message)
                 self.error_handler.log_error(message, ErrorCategory.SYSTEM, ErrorSeverity.CRITICAL)
                 return False

            # Reorder columns to the desired final order and fill any NaN/NaT with empty strings
            # This handles cases where a merge didn't find a match for some rows/columns
            output_df = output_df[final_column_order].fillna('')

            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                logger.info(f"Creating output directory: {output_dir}")
                os.makedirs(output_dir)

            # Save to CSV
            output_df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"Successfully saved output CSV to: {output_file}")
            return True

        except OSError as e: # Catch file system errors specifically
            message = f"Failed to save output CSV file {output_file} due to OS error."
            logger.error(message, exc_info=True)
            self.error_handler.log_error(
                message, ErrorCategory.FILE_IO, ErrorSeverity.CRITICAL, exception=e
            )
            return False
        except Exception as e: # Catch other unexpected errors
            message = f"An unexpected error occurred while saving output CSV file {output_file}"
            logger.error(message, exc_info=True)
            self.error_handler.log_error(
                message, ErrorCategory.SYSTEM, ErrorSeverity.CRITICAL, exception=e
            )
            return False

# Example Usage (assuming ErrorHandler instance exists)
# if __name__ == '__main__':
#     import logging
#     from smite2_translation.utils.logging_utils import setup_logging
#     from smite2_translation.error_handling import ErrorHandler
#     setup_logging()
#     handler = ErrorHandler()
#     processor = DataProcessor(error_handler=handler)
#
#     # Create dummy input CSV
#     dummy_input_path = './data/input/dummy_input.csv'
#     os.makedirs(os.path.dirname(dummy_input_path), exist_ok=True)
#     dummy_data = pd.DataFrame([
#         {'Record ID': '101', 'src_enUS': 'Hello', 'Context': 'Greeting'},
#         {'Record ID': '102', 'src_enUS': 'World {var}', 'Context': 'Noun'},
#         {'Record ID': '103', 'src_enUS': '', 'Context': 'Empty'}, # Invalid
#         {'Record ID': '101', 'src_enUS': 'Duplicate', 'Context': 'Error'}, # Invalid
#         {'Record ID': '104', 'src_enUS': 'Valid', 'Context': 'Test'},
#     ])
#     dummy_data.to_csv(dummy_input_path, index=False)
#
#     # Load data
#     loaded_data = processor.load_input_csv(dummy_input_path)
#     print("\nLoaded Data:", loaded_data)
#
#     # Dummy translations
#     dummy_translations = [
#         {'Record ID': '101', 'tgt_frFR': 'Bonjour', 'tgt_deDE': 'Hallo'},
#         {'Record ID': '102', 'tgt_frFR': 'Monde {var}', 'tgt_deDE': 'Welt {var}'},
#         # '103' is missing because it was invalid
#         {'Record ID': '104', 'tgt_frFR': 'Valide', 'tgt_deDE': 'Gültig'},
#         # Extra data not in original input - should be ignored by update logic
#         {'Record ID': '999', 'tgt_frFR': 'Extra', 'tgt_deDE': 'Zusätzlich'}
#     ]
#
#     # Save data
#     dummy_output_path = './data/output/dummy_output.csv'
#     success = processor.save_output_csv(dummy_translations, dummy_output_path, loaded_data)
#     print(f"\nSave successful: {success}")
#
#     if success:
#         print("\nOutput file content:")
#         print(pd.read_csv(dummy_output_path).to_string())
#
#     print("\n--- Error Report ---")
#     print(handler.generate_error_report()) 