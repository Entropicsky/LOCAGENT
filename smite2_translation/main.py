"""
Main orchestration script for the Smite 2 Translation AI Agent.

Handles command-line arguments, initializes components, manages the translation workflow,
and saves the results.
"""

import argparse
import os
import sys
import logging
from typing import List, Dict, Any
import pandas as pd
from pathlib import Path

# Import core components (assuming they are importable)
try:
    from smite2_translation.core.data_processor import DataProcessor
    from smite2_translation.core.ruleset_manager import RulesetManager
    # from smite2_translation.core.translation_memory import TranslationMemory # Add when implemented
    from smite2_translation.agents.translation_agent import TranslationAgent
    from smite2_translation.error_handling.error_handler import ErrorHandler, ErrorSeverity, ErrorCategory
    from smite2_translation.utils.logging_utils import setup_logging
    # from smite2_translation import config # Add when implemented
except ImportError as e:
    print(f"ERROR: Failed to import necessary modules. Ensure packages are installed and paths are correct. {e}", file=sys.stderr)
    sys.exit(1)

# Setup basic logging until proper setup is implemented
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the Smite 2 Translation AI Agent.")
    
    parser.add_argument(
        "-i", "--input", 
        required=True, 
        help="Path to the input CSV file containing source text."
    )
    parser.add_argument(
        "-o", "--output", 
        required=True, 
        help="Path to the output directory for translated CSV files."
    )
    parser.add_argument(
        "-l", "--languages", 
        nargs='+', 
        default=None, # Default to all languages found in rulesets if None
        help="List of target language codes (e.g., frFR deDE esES). Defaults to all languages in rulesets."
    )
    parser.add_argument(
        "-r", "--rules-dir", 
        default="./rules", # Default relative path
        help="Directory containing the ruleset files (.md)."
    )
    # parser.add_argument(
    #     "--tm-db", 
    #     default="./translation_memory.db", 
    #     help="Path to the Translation Memory SQLite database file."
    # )
    parser.add_argument(
        "--model", 
        default="gpt-4-turbo-preview", # Or config.DEFAULT_MODEL when config exists
        help="OpenAI model to use for translation agents."
    )
    # parser.add_argument(
    #     "--batch-size", 
    #     type=int, 
    #     default=100, # Example default
    #     help="Number of records to process in each batch (if batching implemented)."
    # )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)."
    )

    args = parser.parse_args()
    
    # Basic validation
    if not os.path.isfile(args.input):
        parser.error(f"Input file not found: {args.input}")
    # Output is directory, check/create later
    if not os.path.isdir(args.rules_dir):
         parser.error(f"Rules directory not found: {args.rules_dir}")

    return args

def main():
    """Main execution function."""
    args = parse_arguments()

    # --- 0. Setup Logging --- 
    log_level = logging.DEBUG if args.verbose else logging.INFO
    # TODO: Get log directory from config or args?
    setup_logging(log_level=log_level)
    logger = logging.getLogger(__name__) # Re-get logger after setup

    logger.info(f"Starting translation process with arguments: {args}")

    # --- 1. Initialize Components ---
    logger.info("Initializing components...")
    error_handler = ErrorHandler() # TODO: Add log file config
    ruleset_manager = RulesetManager(error_handler=error_handler)
    data_processor = DataProcessor(error_handler=error_handler)
    # tm_engine = TranslationMemory(db_path=args.tm_db, error_handler=error_handler) # Add when implemented

    # --- 2. Load Rulesets and Data ---
    logger.info(f"Loading rulesets from: {args.rules_dir}")
    ruleset_manager.load_rulesets(args.rules_dir)
    if error_handler.has_critical_errors():
         logger.critical("Critical errors occurred during ruleset loading. Aborting.")
         # TODO: Generate and save/print error report before exiting
         sys.exit(1)
    
    supported_languages = ruleset_manager.get_supported_languages()
    logger.info(f"Rulesets loaded. Detected languages (may differ from target): {supported_languages}")

    logger.info(f"Loading input data from: {args.input}")
    input_data = data_processor.load_input_csv(args.input)
    if not input_data:
         logger.error("Failed to load or found no valid data in input file. Aborting.")
         # TODO: Generate error report
         sys.exit(1)
    logger.info(f"Loaded {len(input_data)} records from input file.")

    # --- 3. Determine Target Languages ---
    target_languages = args.languages
    if target_languages is None:
         # If no languages specified, attempt to use all languages found in rulesets
         # This assumes rulesets actually define target languages section reliably.
         # If not, this needs adjustment or make --languages required.
         target_languages = supported_languages 
         if not target_languages:
              logger.critical("No target languages specified via --languages and none found in rulesets. Cannot proceed.")
              sys.exit(1)
         logger.info(f"No languages specified, using all languages found in rulesets: {target_languages}")
    else:
         # Optional: Validate specified languages against supported/available rules?
         logger.info(f"Target languages specified: {target_languages}")

    # --- 4. Initialize Agents and Translate (Sequential for now) ---
    logger.info(f"Starting translation process for {len(target_languages)} languages.")

    # Aggregate results and save output
    # Initialize with input data to ensure all original rows/cols are preserved
    if input_data:
        final_output_df = pd.DataFrame(input_data)
        # Ensure Record ID is string for consistent merging
        if 'Record ID' in final_output_df.columns:
            final_output_df['Record ID'] = final_output_df['Record ID'].astype(str)
        else:
            logger.error("Input data is missing 'Record ID' column. Cannot proceed with merging.")
            # Handle error appropriately, maybe sys.exit or raise exception
            # For now, let's allow processing but merging might fail later
            pass # Or raise ValueError("Input data missing 'Record ID'")
    else:
        final_output_df = pd.DataFrame() # Start empty if no valid input

    # Collect translations from all languages into a single list
    all_translations_list = []
    for target_language in target_languages:
        logger.info(f"Processing target language: {target_language}")
        # TODO: Add check if ruleset exists for this lang_code?
        
        translation_agent = TranslationAgent(
            target_language=target_language,
            ruleset_manager=ruleset_manager, # Provide full manager
            error_handler=error_handler,
            model=args.model
            # tm_engine=tm_engine # Add when implemented
        )

        # TODO: Implement batching based on args.batch_size if needed
        # For now, process all input_data as one batch
        logger.info(f"Calling translate_batch for {len(input_data)} records...")
        batch_translations = translation_agent.translate_batch(input_data)
        
        if batch_translations:
             logger.debug(f"Received {len(batch_translations)} translations for {target_language}")
             all_translations_list.extend(batch_translations) # Add raw translations to the list
        else:
             logger.warning(f"No translations returned for {target_language}")

    # --- Saving Logic ---
    if not all_translations_list:
        logger.warning("No translations were generated. Output file will not be created.")
    else:
        # Pass the raw list of translation dicts and the original *valid* input data DataFrame
        logger.info(f"Attempting to save {len(all_translations_list)} translations to {args.output}")

        # Convert the original loaded input data (list of dicts) to DataFrame for saving
        # This ensures we use the cleaned/validated input data
        input_df_for_save = pd.DataFrame(input_data)
        if 'Record ID' in input_df_for_save.columns:
            input_df_for_save['Record ID'] = input_df_for_save['Record ID'].astype(str)
        else:
            logger.warning("Input data used for saving lacks 'Record ID'. Merge might be incomplete.")
            # Consider if this should be a critical error depending on requirements
            input_df_for_save = pd.DataFrame() # Default to empty if ID is missing

        # --- Construct Output Filename --- 
        # Use the input filename stem and the output directory
        input_filename_base = Path(args.input).stem
        output_filename = f"{input_filename_base}_translations.csv"
        full_output_path = Path(args.output) / output_filename 
        # Ensure the output directory exists (save_output_csv also does this, but doesn't hurt to be explicit)
        # output_dir_path = Path(args.output)
        # output_dir_path.mkdir(parents=True, exist_ok=True)
        # --- End Construct Output Filename ---

        logger.info(f"Attempting to save translations to file: {full_output_path}")

        success = data_processor.save_output_csv(
            translations_list=all_translations_list,  # Pass the collected list of dicts
            output_file=str(full_output_path), # Pass the constructed full path
            input_df=input_df_for_save # Pass the DataFrame with the correct keyword
        )

        if success:
            logger.info(f"Successfully saved output to {full_output_path}")
        else:
            logger.error(f"Failed to save output to {full_output_path}. Error should be logged by DataProcessor.")
            # Error is logged within save_output_csv, main loop continues to error reporting

    # --- 6. Error Reporting ---
    logger.info("Generating final error report...")
    report = error_handler.generate_error_report()
    print("\n--- Error Summary ---")
    import json
    print(json.dumps(report, indent=2))
    # TODO: Save report to file?

    if report["critical_errors_present"]:
         logger.critical("Process completed with critical errors.")
         sys.exit(1)
    elif report["total_errors"] > 0:
         logger.warning("Process completed with non-critical errors.")
         sys.exit(0) # Exit normally but indicate warnings
    else:
         logger.info("Translation process completed successfully.")
         sys.exit(0)

if __name__ == "__main__":
    main() 