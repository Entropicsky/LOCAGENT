# Smite 2 Translation Agent

This project implements an AI-powered translation agent system designed to translate game text for Smite 2. It ensures consistency and quality through language-specific rulesets, glossaries, and automated quality assessment.

## Features

### Core Translation Features
* **Multi-language Support**: Translate to multiple target languages (French, German, Spanish, Portuguese, Japanese, Chinese, Polish, Ukrainian, Russian, Turkish)
* **Record-by-record Processing**: Translates each record to all target languages before moving to the next one
* **Language-specific Rulesets**: Applies detailed language-specific rules using Markdown files for consistent formatting, terminology, and cultural adaptation
* **Advanced LLM Integration**: Utilizes OpenAI gpt-4o model for high-quality translations
* **Continuous Output**: Writes completed translations to CSV immediately after processing to avoid data loss

### Quality & Robustness Features
* **Automated Quality Assessment**: Checks translations for rule compliance and critical issues
* **Auto-retry Mechanism**: Automatically attempts to fix translations with critical quality issues
* **Resumable Processing**: Detects and skips already processed records, allowing interruption and continuation
* **Detailed Logging**: Comprehensive logging for troubleshooting and monitoring
* **Progress Tracking**: Real-time progress updates during translation

### Technical Features
* **Centralized Configuration**: All configuration values defined in a single location
* **Modular Design**: Well-structured codebase with separation of concerns
* **Comprehensive Error Handling**: Categorized error handling with severity levels
* **Test Suite**: Unit and integration tests for key components

## Setup

### Prerequisites

* Python 3.10+
* `pip` package manager
* OpenAI API key

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Entropicsky/LocAgent.git
   cd LocAgent
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Environment Variables:**
   Create a `.env` file in the project root and add your OpenAI API key:
   ```dotenv
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

### Basic Command Structure

Run the main translation script from the project root directory:

```bash
python -m smite2_translation.main -i INPUT_FILE -o OUTPUT_DIR [OPTIONS]
```

### Command-line Arguments

#### Required Arguments:
* `-i, --input`: Path to the input CSV file containing source text
* `-o, --output`: Directory where output files will be saved

#### Language Selection Arguments:
* `-l, --languages`: Specific language codes to translate to (e.g., `frFR deDE jaJP`)
* `--all-languages`: Process all available languages found in ruleset files

#### Configuration Arguments:
* `-r, --rules-dir`: Directory containing ruleset files (default: `./rules`)
* `--model`: OpenAI model name (default: `gpt-4o`)
* `--verbose`: Enable DEBUG level logging for detailed output
* `--auto-retry`: Enable auto-fixing for critical QA issues (enabled by default)
* `--no-auto-retry`: Disable auto-fixing for critical QA issues

### Available Language Codes

* `frFR`: French
* `deDE`: German
* `esLA`: Latin American Spanish
* `ptBR`: Brazilian Portuguese
* `jaJP`: Japanese
* `zhCN`: Simplified Chinese
* `plPL`: Polish
* `ukUA`: Ukrainian
* `ruRU`: Russian
* `trTR`: Turkish

### Example Commands

#### Translate to a Single Language
```bash
python -m smite2_translation.main -i input/input.csv -o output --languages deDE --verbose
```

#### Translate to Multiple Specific Languages
```bash
python -m smite2_translation.main -i input/input.csv -o output -l frFR deDE jaJP --verbose
```

#### Translate to All Available Languages
```bash
python -m smite2_translation.main -i input/input.csv -o output --all-languages --verbose
```

#### Use a Different Model
```bash
python -m smite2_translation.main -i input/input.csv -o output -l deDE --model gpt-4o-mini --verbose
```

#### Disable Auto-retry for Quality Issues
```bash
python -m smite2_translation.main -i input/input.csv -o output -l deDE --no-auto-retry --verbose
```

## Input File Format

The input CSV file should contain the following columns:
* `Record ID`: Unique identifier for each record
* `src_enUS`: Source English text to translate
* `Context` (optional): Context information to help guide translation
* `Path` (optional): Path information for additional context

Example:
```csv
Record ID,src_enUS,Context,Path
001,Welcome to SMITE 2!,Main Menu,UI/Welcome
002,Attack,Button Label,UI/Combat
```

## Output Format

The system produces several output files:

### Main Translation Output
A CSV file named `[input_filename]_translations.csv` containing:
* Original record data (Record ID, Source Text, Context, Path)
* Translations for each target language in columns named `Translation_[langCode]`

### Quality Assessment Report
A CSV file named `[input_filename]_qa_report.csv` with quality metrics for each translation.

### Debug Report
A CSV file named `[input_filename]_debug_report.csv` with detailed debugging information.

### Log Files
Detailed logs in the `logs/` directory, including `translation_app.log`.

## Translation Workflow

1. **Data Loading**: The system loads the input CSV file and rulesets
2. **Record Processing**: For each record:
   * Translates the record to all target languages
   * Performs quality assessment on translations
   * Auto-retries critical quality issues if enabled
   * Writes completed translations to the output CSV
3. **Completion**: Generates final reports and summaries

## Project Structure

```
.
├── .cursor/              # Agent-specific notes, rules, tools
│   ├── notes/            # Project documentation and notes
│   ├── rules/            # Agent rules
│   └── tools/            # Agent tools
├── input/                # Input CSV files
├── logs/                 # Log files generated by the application
├── output/               # Output translated CSV files
├── rules/                # Language-specific rulesets (.md files)
│   ├── brazilian_portuguese_translation_ruleset.md
│   ├── chinese_translation_ruleset.md
│   ├── french_translation_ruleset.md
│   ├── german_translation_ruleset.md
│   ├── japanese_translation_ruleset.md
│   ├── latin_american_spanish_translation_ruleset.md
│   ├── polish_translation_ruleset.md
│   ├── russian_translation_ruleset.md
│   ├── turkish_translation_ruleset.md
│   ├── ukrainian_translation_ruleset.md
│   └── global_ruleset.md # Global rules applied to all languages
├── smite2_translation/   # Main source code package
│   ├── agents/           # AI agent implementations
│   │   ├── translation_agent.py  # Core translation agent
│   │   └── quality_assessor.py   # Quality assessment agent
│   ├── core/             # Core logic
│   │   ├── data_processor.py     # CSV processing
│   │   └── ruleset_manager.py    # Ruleset handling
│   ├── error_handling/   # Error handling framework
│   │   └── error_handler.py      # Error categorization
│   ├── utils/            # Utility functions
│   │   ├── config_loader.py      # Configuration management
│   │   ├── logging_utils.py      # Logging setup
│   │   └── qa_tools.py           # Quality assessment tools
│   ├── config.py         # Central configuration values
│   └── main.py           # Main script entry point
├── tests/                # Unit and integration tests
│   ├── agents/           # Tests for agent components
│   ├── core/             # Tests for core components
│   └── utils/            # Tests for utility functions
├── .env                  # Environment variables (API keys) - Not committed
├── .gitignore            # Git ignore file
├── requirements.txt      # Python package dependencies
└── README.md             # This documentation file
```

## Rulesets

Rulesets are Markdown files in the `rules/` directory that define language-specific translation rules, glossaries, and style guides. Each ruleset includes:

* **Grammatical Rules**: Gender, case, verb forms, etc.
* **Formatting Rules**: Capitalization, punctuation, number formatting
* **Terminology Rules**: Game-specific terms and consistent translations
* **Cultural Adaptation Rules**: Formality, cultural references, humor
* **Exception Handling Rules**: Untranslatable terms, character limitations
* **Glossary**: Specific term translations that must be used consistently

## Error Handling

The system categorizes errors by:

* **Category**: API, Configuration, Input Data, Ruleset, Translation, etc.
* **Severity**: Critical, High, Medium, Low, Info

Critical errors are reported and can trigger auto-retry mechanisms to fix issues automatically.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Troubleshooting

### Common Issues

* **OpenAI API Key**: Ensure your API key is correctly set in the `.env` file
* **Ruleset Loading**: Check that ruleset files exist in the specified `--rules-dir` directory
* **Input CSV Format**: Verify your input CSV has the required columns and proper formatting
* **Output Directory**: Ensure the output directory exists and is writable

### Logs

Check the logs in the `logs/` directory for detailed error messages and debugging information. 