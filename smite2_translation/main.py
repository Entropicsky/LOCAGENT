"""
Main orchestration script for the Smite 2 Translation AI Agent.

Handles command-line arguments, initializes components, manages the translation workflow,
and saves the results.
"""

import os
import sys
import csv
import json
import logging
import argparse
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
import pickle  # For saving partial progress

# Import from config.py for centralized configuration
from smite2_translation.config import RULESET_DIR, DEFAULT_MODEL, SUPPORTED_LANGUAGES

# Only need load_config now
from smite2_translation.utils.config_loader import load_config
from smite2_translation.utils.logging_utils import setup_logging
from smite2_translation.error_handling.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
from smite2_translation.agents.translation_agent import TranslationAgent
from smite2_translation.core.data_processor import DataProcessor
from smite2_translation.core.ruleset_manager import RulesetManager
from smite2_translation.agents.quality_assessor import QualityAssessmentAgent

# Setup basic logging until proper setup is implemented
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Smite 2 Translation Agent")
    
    # Required args
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input CSV file with content to translate"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output directory for translated content"
    )
    
    # Optional args
    parser.add_argument(
        "-l", "--languages",
        nargs="+",
        help="Target language codes to translate to (e.g., frFR deDE). If not specified, all available languages will be used."
    )
    parser.add_argument(
        "--all-languages",
        action="store_true",
        help="Process all available languages (found in ruleset files)"
    )
    parser.add_argument(
        "-r", "--rules-dir",
        default=RULESET_DIR,  # Use value from config.py
        help="Directory containing ruleset files for languages."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,  # Directly use the value from config.py
        help="OpenAI model to use for translation."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)."
    )
    parser.add_argument(
        "--auto-retry",
        action="store_true",
        default=True,
        help="Enable auto-retry for critical QA issues. The system will attempt to fix critical issues automatically. Defaults to enabled, use --no-auto-retry to disable."
    )
    parser.add_argument(
        "--no-auto-retry",
        action="store_false",
        dest="auto_retry",
        help="Disable auto-retry for critical QA issues."
    )
    
    args = parser.parse_args()
    return args

def main():
    """Main execution function."""
    args = parse_arguments()
    config = load_config()
    
    # Set log level based on verbose flag
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level=log_level)
    
    logger = logging.getLogger(__name__) # Re-get logger after setup

    # Test file writing capability and create a debug file
    try:
        os.makedirs(args.output, exist_ok=True)
        test_file_path = os.path.join(args.output, "test_write.txt")
        
        # Create a debug file to monitor what's happening
        debug_file_path = os.path.join(args.output, "translation_debug.txt")
        with open(debug_file_path, 'w') as f:
            f.write(f"Translation process started at {pd.Timestamp.now()}\n")
            f.write(f"Args: {args}\n")
            f.write(f"Output directory: {os.path.abspath(args.output)}\n")
            f.write(f"Input file: {os.path.abspath(args.input)}\n")
        
        # Try writing to a test file
        with open(test_file_path, 'w') as f:
            f.write("Test write capability")
        logger.info(f"Successfully wrote test file to {test_file_path}")
        
        # Update debug file
        with open(debug_file_path, 'a') as f:
            f.write(f"Test write successful at {pd.Timestamp.now()}\n")
            
        os.remove(test_file_path)  # Clean up
        logger.info("Test file removed")
    except Exception as e:
        logger.error(f"***CRITICAL ERROR*** Failed to write test file: {str(e)}")
        logger.error(f"Writing to directory: {args.output}")
        import traceback
        logger.error(f"Exception traceback: {traceback.format_exc()}")
        return

    logger.info(f"Starting translation process with arguments: {args}")

    # --- 1. Initialize Components ---
    logger.info("Initializing components...")
    error_handler = ErrorHandler() # TODO: Add log file config
    data_processor = DataProcessor(error_handler=error_handler)
    # tm_engine = TranslationMemory(db_path=args.tm_db, error_handler=error_handler) # Add when implemented

    # --- 2. Load Rulesets and Data ---
    logger.info("Initializing RulesetManager and QualityAssessmentAgent...")
    ruleset_manager = RulesetManager(error_handler=error_handler)
    # Get ruleset directory from config or default
    ruleset_dir = config.get('ruleset_directory', './rules') 
    logger.info(f"Loading rulesets from: {os.path.abspath(ruleset_dir)}")
    ruleset_manager.load_rulesets(ruleset_dir)
    # Use get_ruleset which returns Dict[str, Dict[str, Any]]
    language_rulesets = ruleset_manager.get_ruleset() 

    qa_agent: QualityAssessmentAgent | None = None # Define type hint
    if not language_rulesets:
        logger.warning("No language-specific rulesets loaded. Quality assessment will be skipped.")
        # Continue without QA agent
    else:
        loaded_langs = list(language_rulesets.keys())
        logger.info(f"Successfully loaded rulesets for languages: {loaded_langs}")
        # Check if requested languages have rulesets (only when specific languages are requested)
        if args.languages and not args.all_languages:
            missing_ruleset_langs = [lang for lang in args.languages if lang not in loaded_langs]
            if missing_ruleset_langs:
                 logger.warning(f"Rulesets missing for requested languages: {missing_ruleset_langs}. QA will be skipped for these.")
        qa_agent = QualityAssessmentAgent(language_rulesets) # Initialize agent

    logger.info(f"Loading input data from: {args.input}")
    # Load data as list first
    input_list = data_processor.load_input_csv(args.input)
    # Convert to DataFrame, handle empty list case
    input_data = pd.DataFrame(input_list) if input_list else pd.DataFrame()

    # Check if DataFrame is empty after conversion
    if input_data.empty:
         logger.error("Failed to load or found no valid data in input file after processing. Aborting.")
         # Error likely already logged by load_input_csv, but double-check logic
         # Exit if no data to process
         sys.exit(1)
    logger.info(f"Loaded {len(input_data)} records from input file.")

    # --- 3. Determine Target Languages ---
    target_languages = args.languages
    process_all_languages = args.all_languages
    
    # If --all-languages flag is set, use all languages found in rulesets
    if process_all_languages:
        target_languages = ruleset_manager.get_supported_languages()
        if not target_languages:
            logger.critical("No languages found in rulesets. Cannot process all languages.")
            sys.exit(1)
        logger.info(f"Processing all available languages: {target_languages}")
    # Otherwise, use languages specified or default to all available languages
    elif target_languages is None:
        # If no languages specified, attempt to use all languages found in rulesets
        target_languages = ruleset_manager.get_supported_languages()
        if not target_languages:
            logger.critical("No target languages specified via --languages and none found in rulesets. Cannot proceed.")
            sys.exit(1)
        logger.info(f"No languages specified, using all languages found in rulesets: {target_languages}")
    else:
        # Optional: Validate specified languages against supported/available rules?
        logger.info(f"Target languages specified: {target_languages}")

    # --- 4. Batch Translate --- 
    logger.info(f"Starting batch translation process for {len(target_languages)} languages.")

    # Prepare the full batch data once (list of dicts needed by translate_batch)
    batch_input_data = []
    for index, row in input_data.iterrows():
        # Ensure required fields are present and valid before adding to batch
        record_id = str(row.get('Record ID', ''))
        src_text = str(row.get('src_enUS', '')) if pd.notna(row.get('src_enUS')) else ''
        context = str(row.get('Context', '')) if pd.notna(row.get('Context')) else '' # Assuming context might be relevant
        path = str(row.get('Path', '')) if pd.notna(row.get('Path')) else '' # Include Path if used by agent

        if record_id and src_text: # Only include valid records
            batch_input_data.append({
                'Record ID': record_id,
                'src_enUS': src_text,
                'Context': context, # Pass context if agent uses it
                'Path': path        # Pass Path if agent uses it
            })
        else:
            logger.warning(f"Excluding record from batch due to missing ID or source text: ID='{record_id}', Source='{src_text[:20]}...'")

    if not batch_input_data:
        logger.error("No valid records prepared for batch translation. Aborting.")
        sys.exit(1)

    # Dictionary to hold all results, keyed by Record ID
    # Value will be a dict holding source, context, and translations per lang
    # Initialize with data from the input DataFrame for easy lookup later
    all_record_data = {}
    for index, row in input_data.iterrows():
         record_id = str(row.get('Record ID',''))
         if record_id: # Only process rows with an ID
             all_record_data[record_id] = {
                 'Record ID': record_id,
                 'Source Text': str(row.get('src_enUS','')) if pd.notna(row.get('src_enUS')) else '',
                 'Context': str(row.get('Context','')) if pd.notna(row.get('Context')) else '',
                 'QA_Issues': {} # Initialize dict for QA issues per language
                 # Add other original columns if needed for QA report later?
             }

    # --- Process each language and perform translations ---
    results_df = None
    translation_cols = [] # Track columns containing translations
    
    # Create a temporary directory for storing intermediate results
    temp_dir = args.output # Use the output directory as the temporary directory
    os.makedirs(temp_dir, exist_ok=True)
    logger.info(f"Created directory for output: {temp_dir}")
    
    # Setup output files
    input_filename_base = Path(args.input).stem
    machine_output_path = os.path.join(args.output, f"{input_filename_base}_translations.csv")
    
    # Use model from args (which defaults to config value if not specified)
    logger.info(f"Using model: {args.model}")
    
    # Create agents for all target languages upfront
    language_agents = {}
    for lang_code in target_languages:
        language_agents[lang_code] = TranslationAgent(
            target_language=lang_code,
            ruleset_manager=ruleset_manager,
            error_handler=error_handler,
            model=args.model
        )
    
    # Track already processed records to avoid duplication
    processed_records = set()
    
    # Check if output file already exists and load processed records
    if os.path.exists(machine_output_path):
        try:
            logger.info(f"Found existing output file. Loading to identify already processed records.")
            existing_df = pd.read_csv(machine_output_path)
            if 'Record ID' in existing_df.columns:
                processed_records = set(existing_df['Record ID'].astype(str).tolist())
                logger.info(f"Loaded {len(processed_records)} already processed records from existing file.")
        except Exception as e:
            logger.warning(f"Failed to load existing output file: {str(e)}. Starting fresh.")
    
    # Create output CSV header if needed
    if not processed_records:
        # Prepare the header with all language translation columns
        header_cols = ['Record ID', 'Source Text', 'Context', 'Path']
        for lang_code in target_languages:
            header_cols.append(f"Translation_{lang_code}")
            
        # Create empty DataFrame with just the header and save
        header_df = pd.DataFrame(columns=header_cols)
        header_df.to_csv(machine_output_path, index=False, encoding='utf-8')
        logger.info(f"Created output file with header: {machine_output_path}")
    
    # --- Process Each Record ---
    logger.info(f"Processing translations record-by-record for {len(target_languages)} languages...")
    records_processed = 0
    records_skipped = 0
    
    # Process each record
    for record in batch_input_data:
        record_id = record.get('Record ID')
        
        # Skip if already processed
        if record_id in processed_records:
            records_skipped += 1
            logger.info(f"Skipping already processed record: {record_id}")
            continue
        
        # Prepare data for this record
        source_text = record.get('src_enUS', '')
        context = record.get('Context', '')
        path = record.get('Path', '')
        
        record_data = {
            'Record ID': record_id,
            'Source Text': source_text,
            'Context': context,
            'Path': path
        }
        
        # Log that we're processing this record
        logger.info(f"Processing record {record_id} ({records_processed + 1}/{len(batch_input_data) - records_skipped})...")
        
        # Process each language for this record
        for lang_code in target_languages:
            translation_key = f"Translation_{lang_code}"
            logger.info(f"  Translating {record_id} to {lang_code}...")
            
            # Get the agent for this language
            translation_agent = language_agents.get(lang_code)
            if not translation_agent:
                logger.warning(f"No agent for {lang_code}. Skipping language for record {record_id}.")
                record_data[translation_key] = "[AGENT_ERROR]"
                continue
            
            # Translate
            try:
                trans_result = translation_agent.translate_record(record)
                translated_text = trans_result.get(f"tgt_{lang_code}")
                debug_info = trans_result.get('debug_info', '')
                
                # Store translation
                record_data[translation_key] = translated_text
                
                # Write test file to verify we can write to the directory
                test_file_path = os.path.join(args.output, f"translation_check_{record_id}_{lang_code}.txt")
                try:
                    with open(test_file_path, 'w') as f:
                        f.write(f"Test write at translation of {record_id} to {lang_code}\n")
                    os.remove(test_file_path)  # Clean up
                except Exception as test_error:
                    logger.error(f"Failed to write test file: {str(test_error)}")
                    
            except Exception as e:
                logger.error(f"Error translating {record_id} to {lang_code}: {str(e)}")
                record_data[translation_key] = "[TRANSLATION_ERROR]"
        
        # Record is fully processed for all languages, append to output file
        try:
            # Create a one-row DataFrame
            record_df = pd.DataFrame([record_data])
            
            # Append to the output CSV file
            record_df.to_csv(machine_output_path, mode='a', header=False, index=False, encoding='utf-8')
            
            # Update our progress log
            logger.info(f"Successfully wrote record {record_id} with all translations to {machine_output_path}")
            
            # Update count
            records_processed += 1
            processed_records.add(record_id)
            
            # Update debug file
            debug_file_path = os.path.join(args.output, "translation_debug.txt")
            with open(debug_file_path, 'a') as f:
                f.write(f"Completed record {record_id} at {pd.Timestamp.now()}\n")
                f.write(f"Progress: {records_processed}/{len(batch_input_data) - records_skipped} records\n")
                f.flush()
                
        except Exception as write_error:
            logger.error(f"Failed to write record {record_id} to output file: {str(write_error)}")
            import traceback
            logger.error(f"Exception traceback: {traceback.format_exc()}")
    
    logger.info(f"Completed processing {records_processed} records. Skipped {records_skipped} already processed records.")

    # --- Step 5: Perform QA and Assemble Final Results --- 
    logger.info("Performing Quality Assessment and assembling final results...")
    all_results = [] # Initialize list for final DataFrame rows

    # Add argument for auto-retry of critical issues
    auto_retry_critical = args.auto_retry if hasattr(args, 'auto_retry') else False
    if auto_retry_critical:
        logger.info("Auto-retry for critical issues is enabled. Will attempt to fix critical issues.")
    else:
        logger.info("Auto-retry is disabled. QA will identify issues but not attempt to fix them.")

    # Iterate through the collected record data which now includes translations
    for record_id, record_data in all_record_data.items():
        source_text = record_data.get('Source Text', '')
        # Initialize QA issues dict for this record if not already present (should be)
        record_data.setdefault('QA_Issues', {}) 

        # Perform QA for each target language where a translation exists
        for lang_code in target_languages:
            translation_key = f"Translation_{lang_code}"
            translated_text = record_data.get(translation_key)

            # Check if QA agent is available AND language ruleset exists AND translation seems valid
            if qa_agent and lang_code in language_rulesets and translated_text and translated_text != "[BATCH_ERROR]":
                logger.debug(f"  Assessing quality for {record_id} ({lang_code})...")
                try:
                    qa_issues = qa_agent.assess_quality(
                        source_text=source_text,
                        target_text=translated_text,
                        target_language=lang_code,
                        record_id=record_id
                    )
                    
                    # Store the list of QA issues (even if empty)
                    record_data['QA_Issues'][lang_code] = qa_issues 
                    
                    # Auto-retry critical issues if enabled
                    if auto_retry_critical and qa_issues:
                        critical_issues = [issue for issue in qa_issues 
                                         if issue.get('severity', '').upper() == 'CRITICAL']
                        
                        if critical_issues:
                            logger.info(f"Found {len(critical_issues)} critical issues for {record_id} ({lang_code}). Attempting to fix...")
                            
                            # Prepare fix request with info about the issues
                            fix_request = {
                                'Record ID': record_id,
                                'src_enUS': source_text,
                                'Context': record_data.get('Context', ''),
                                'qa_issues': critical_issues,
                                'current_translation': translated_text
                            }
                            
                            # Try to fix the translation
                            try:
                                fixed_translation = translation_agent.fix_translation(fix_request)
                                
                                if fixed_translation:
                                    # Verify the fix with QA again
                                    fixed_qa_issues = qa_agent.assess_quality(
                                        source_text=source_text,
                                        target_text=fixed_translation,
                                        target_language=lang_code,
                                        record_id=record_id
                                    )
                                    
                                    # Check if critical issues are fixed
                                    fixed_critical_issues = [issue for issue in fixed_qa_issues 
                                                           if issue.get('severity', '').upper() == 'CRITICAL']
                                    
                                    if len(fixed_critical_issues) < len(critical_issues):
                                        # Some or all critical issues fixed, update the translation
                                        logger.info(f"Fixed {len(critical_issues) - len(fixed_critical_issues)} out of {len(critical_issues)} critical issues for {record_id}")
                                        record_data[translation_key] = fixed_translation
                                        record_data['QA_Issues'][lang_code] = fixed_qa_issues
                                        record_data[f"Debug_{lang_code}"] += " ||| [AUTO-FIXED] Translation was automatically fixed to address critical issues."
                                    else:
                                        logger.warning(f"Auto-fix attempt did not resolve critical issues for {record_id}")
                            except Exception as fix_exc:
                                logger.error(f"Error attempting to auto-fix translation for {record_id}: {str(fix_exc)}")
                    
                    if qa_issues:
                        logger.debug(f"  QA Issues found ({record_id}, {lang_code}): {len(qa_issues)}")

                except Exception as qa_exc:
                     logger.error(f"  Unexpected error during QA for {record_id} ({lang_code})", exc_info=qa_exc)
                     error_handler.log_error(f"QA assessment failed for {record_id}/{lang_code}", ErrorCategory.SYSTEM, ErrorSeverity.HIGH, details=str(qa_exc))
                     record_data['QA_Issues'][lang_code] = [{"type": "QA_SYSTEM_ERROR", "error": str(qa_exc)}] # Record QA system error
            
            elif qa_agent and lang_code not in language_rulesets:
                 logger.debug(f"  Skipping QA for {record_id} ({lang_code}) - No ruleset loaded.")
                 record_data['QA_Issues'][lang_code] = [{"type": "QA_SKIPPED", "error": "No ruleset loaded for language"}]
            elif not translated_text or translated_text == "[BATCH_ERROR]":
                 logger.debug(f"  Skipping QA for {record_id} ({lang_code}) - Translation failed, empty, or batch error.")
                 record_data['QA_Issues'][lang_code] = [{"type": "QA_SKIPPED", "error": "Translation missing or failed"}]
            elif not qa_agent: # Handles case where qa_agent is None
                 logger.debug(f"  Skipping QA for {record_id} ({lang_code}) - QA Agent not available.")
                 record_data['QA_Issues'][lang_code] = [{"type": "QA_SKIPPED", "error": "QA Agent not available"}]
            # --- End Quality Assessment for this language ---

        # Convert the QA issues dict to a JSON string before adding to final results
        # Important: Get the QA issues dict safely, default to empty dict if missing
        qa_issues_dict = record_data.get('QA_Issues', {})
        record_data['QA_Issues'] = json.dumps(qa_issues_dict) 
        
        # Append the fully processed record dictionary to the final list
        all_results.append(record_data)

    logger.info("Finished QA checks and assembled final results.")
    
    # --- Create DataFrames and Save Output --- 
    # Create DataFrame from the final list of results
    if not all_results:
        logger.warning("No results were generated during processing. Output files will be empty or may fail.")
        results_df = pd.DataFrame() # Create empty df to avoid errors later
    else:
        results_df = pd.DataFrame(all_results)

    # --- Save the machine-readable output --- 
    logger.info("Preparing machine-readable output file...")
    # Identify original columns present in input_df (use the initially loaded df)
    # Need to ensure input_df is available here. Let's reload it if necessary or pass it through.
    # Assuming input_df was loaded successfully earlier and is accessible.
    original_cols = list(input_data.columns) 
    # Identify new translation columns created in results_df
    translation_cols = [col for col in results_df.columns if col.startswith('Translation_')]
    
    # Select necessary columns from results_df for merging (Record ID and translations)
    # Ensure 'Record ID' exists in results_df if it's not empty
    merge_cols = []
    if 'Record ID' in results_df.columns:
        merge_cols = ['Record ID'] + translation_cols
        results_to_merge = results_df[merge_cols]
    else:
        logger.warning("No 'Record ID' found in results DataFrame. Cannot merge translations back.")
        # Create empty df with Record ID for merge safety, assuming input_df has 'Record ID'
        results_to_merge = pd.DataFrame({'Record ID': []}) 

    # Merge with the original input data using 'Record ID'
    # Use left merge to keep all original rows and add translations
    # Need to ensure input_df has 'Record ID' of compatible type (string)
    input_df_for_merge = input_data.copy()
    if 'Record ID' in input_df_for_merge.columns:
        input_df_for_merge['Record ID'] = input_df_for_merge['Record ID'].astype(str)
    else:
        logger.error("Original input_df missing 'Record ID' for merge. Cannot create machine output.")
        # Set final_output_df to indicate failure or empty state
        final_output_df = pd.DataFrame() 
        # Skip saving this file later

    if 'Record ID' in input_df_for_merge.columns: # Proceed only if merge is possible
        final_output_df = pd.merge(input_df_for_merge, results_to_merge, on='Record ID', how='left')
        
        # Ensure the column order: original columns first, then new translations
        final_col_order = original_cols + [col for col in translation_cols if col not in original_cols]
        # Reorder columns, only including those that exist in the merged DataFrame
        final_output_df = final_output_df[[col for col in final_col_order if col in final_output_df.columns]]

    # --- Save the machine-readable output (conditional) --- 
    if not final_output_df.empty:
        # Construct machine output filename
        output_dir = args.output # Output directory specified by user
        os.makedirs(output_dir, exist_ok=True) # Ensure directory exists
        input_filename_base = Path(args.input).stem
        machine_output_filename = f"{input_filename_base}_translations.csv"
        machine_output_path = os.path.join(output_dir, machine_output_filename)

        logger.info(f"Saving machine-readable output to: {machine_output_path}")
         
        # Create the simplified translations list expected by the test
        simplified_translations = []
        for record in all_results:
            trans_dict = {
                'Record ID': record['Record ID']
            }
            # Add all language translations in the expected format
            for lang_code in target_languages:
                # The test expects 'tgt_frFR' key format, but we store as 'Translation_frFR'
                # Map from our internal format to the expected format
                trans_key = f"Translation_{lang_code}"
                if trans_key in record:
                    trans_dict[f"tgt_{lang_code}"] = record[trans_key]
            # Only add if we have at least one translation
            if len(trans_dict) > 1:  # More than just Record ID
                simplified_translations.append(trans_dict)
        
        # Call data_processor.save_output_csv with the correct parameters
        try:
            # Use exactly the keyword arguments expected by the test
            success = data_processor.save_output_csv(
                translations_list=simplified_translations,  # Required keyword by test
                output_file=machine_output_path,
                input_df=final_output_df  # Required keyword by test
            )
            if success:
                logger.info(f"Machine-readable output saved successfully to {machine_output_path}.")
            else:
                logger.error(f"Failed to save machine-readable output to {machine_output_path}.")
                error_handler.log_error("save_output_csv returned False", ErrorCategory.FILE_IO, ErrorSeverity.HIGH)
        except Exception as e:
            logger.error(f"Failed to save machine-readable output to {machine_output_path}", exc_info=True)
            error_handler.log_error(f"Failed to save machine output: {e}", ErrorCategory.FILE_IO, ErrorSeverity.HIGH)
    else:
        logger.warning("Machine-readable output DataFrame is empty. Skipping save.")


    # --- Save the human-readable QA report --- 
    logger.info("Preparing human-readable QA report file...")
    # Define columns for the QA report (ensure columns exist in results_df)
    qa_report_cols_base = ['Record ID', 'Source Text', 'Context', 'QA_Issues']
    qa_report_cols = []
    if not results_df.empty:
        # Include base columns if they exist, plus all translation columns
        qa_report_cols = [col for col in qa_report_cols_base if col in results_df.columns] + translation_cols
        # Ensure unique columns in the desired order
        qa_report_cols = list(dict.fromkeys(qa_report_cols)) 
        # Select only existing columns from results_df to avoid errors
        qa_report_df = results_df[[col for col in qa_report_cols if col in results_df.columns]]
    else:
        # Create empty df with expected columns if no results
        qa_report_df = pd.DataFrame(columns=list(dict.fromkeys(qa_report_cols_base + translation_cols))) 


    # Construct QA report filename
    output_dir = args.output # Use same output directory
    os.makedirs(output_dir, exist_ok=True)
    input_filename_base = Path(args.input).stem
    qa_report_filename = f"{input_filename_base}_qa_report.csv"
    qa_report_path = os.path.join(output_dir, qa_report_filename)

    logger.info(f"Saving human-readable QA report to: {qa_report_path}")
    try:
        qa_report_df.to_csv(qa_report_path, index=False, encoding='utf-8')
        logger.info("QA report saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save QA report to {qa_report_path}", exc_info=True)
        error_handler.log_error(f"Failed to save QA report: {e}", ErrorCategory.FILE_IO, ErrorSeverity.HIGH)

    # --- Generate Detailed Debug Report ---
    logger.info("Generating detailed debug report...")
    
    # Create a detailed debug data structure
    debug_data = []
    
    for record_id, record_data in all_record_data.items():
        source_text = record_data.get('Source Text', '')
        context = record_data.get('Context', '')
        
        for lang_code in target_languages:
            translation_key = f"Translation_{lang_code}"
            translated_text = record_data.get(translation_key, '')
            
            # Get language-specific ruleset
            lang_ruleset = {}
            if language_rulesets and lang_code in language_rulesets:
                lang_ruleset = language_rulesets[lang_code]
            
            # Extract rules applied to this language
            rules_info = {}
            if lang_ruleset:
                # Extract general rules
                if 'general_rules' in lang_ruleset:
                    rules_info['general_rules'] = lang_ruleset['general_rules']
                
                # Extract formatting rules
                if 'formatting_rules' in lang_ruleset:
                    rules_info['formatting_rules'] = lang_ruleset['formatting_rules']
                
                # Extract term translations
                if 'terms' in lang_ruleset:
                    # Take only the first 20 term entries to avoid making the debug file too large
                    terms_sample = dict(list(lang_ruleset['terms'].items())[:20])
                    rules_info['terms_sample'] = terms_sample
                    rules_info['total_terms'] = len(lang_ruleset['terms'])
                
                # Extract style guidelines
                if 'style_guidelines' in lang_ruleset:
                    rules_info['style_guidelines'] = lang_ruleset['style_guidelines']
            
            # Get QA issues for this record/language
            qa_issues = []
            try:
                qa_issues_dict = json.loads(record_data['QA_Issues'])
                if lang_code in qa_issues_dict:
                    qa_issues = qa_issues_dict[lang_code]
            except (json.JSONDecodeError, KeyError):
                # Handle case where QA_Issues isn't valid JSON or doesn't have language key
                pass
            
            # Get the agent's debug info about applied rules
            debug_key = f"Debug_{lang_code}"
            agent_debug_info = record_data.get(debug_key, "No debug information captured")
            
            # Create debug entry
            debug_entry = {
                'Record ID': record_id,
                'Source Text': source_text,
                'Context': context,
                'Language': lang_code,
                'Translation': translated_text,
                'Agent Debug Info': agent_debug_info,
                'Rules Available': json.dumps(rules_info, indent=2, ensure_ascii=False),
                'QA Issues': json.dumps(qa_issues, indent=2, ensure_ascii=False)
            }
            
            debug_data.append(debug_entry)
    
    # Create debug report DataFrame
    debug_df = pd.DataFrame(debug_data)
    
    # Save debug report
    debug_report_filename = f"{input_filename_base}_debug_report.csv"
    debug_report_path = os.path.join(output_dir, debug_report_filename)
    
    logger.info(f"Saving detailed debug report to: {debug_report_path}")
    try:
        debug_df.to_csv(debug_report_path, index=False, encoding='utf-8')
        logger.info("Debug report saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save debug report to {debug_report_path}", exc_info=True)
        error_handler.log_error(f"Failed to save debug report: {e}", ErrorCategory.FILE_IO, ErrorSeverity.HIGH)

    # --- Error Reporting --- 
    report = error_handler.generate_error_report()
    print("\n--- Error Summary ---")
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