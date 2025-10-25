# Bid Document Processing System

A sophisticated RAG (Retrieval-Augmented Generation) pipeline for extracting structured information from bid documents using Docling, ChromaDB, LangChain, and Google Gemini.

## Features

- **Document Processing**: Supports PDF and HTML files using Docling with OCR and table extraction
- **RAG Pipeline**: Uses ChromaDB for vector storage and similarity search
- **AI Extraction**: Leverages Google Gemini 2.5 Pro for intelligent information extraction
- **Structured Output**: Extracts 20 predefined fields from bid documents
- **Multi-Phase Extraction**: Progressive refinement approach for comprehensive data extraction

## Extracted Fields

The system extracts the following 20 fields from bid documents:

- `bid_number` - RFP, solicitation, or contract numbers
- `title` - Bid title and description
- `due_date` - Submission deadline
- `bid_submission_type` - Electronic/paper submission method
- `term_of_bid` - Contract duration and validity period
- `pre_bid_meeting` - Pre-bid conference information
- `installation` - Installation requirements
- `bid_bond_requirement` - Bond requirements
- `delivery_date` - Delivery timeline
- `payment_terms` - Payment schedule and terms
- `additional_documentation` - Required supporting documents
- `mfg_for_registration` - Manufacturer registration requirements
- `contract_cooperative` - Contract type and cooperative agreements
- `model_no` - Product model numbers
- `part_no` - Part numbers and component codes
- `product` - Product description
- `contact_info` - Contact person and details
- `company_name` - Company information
- `bid_summary` - Executive summary
- `product_specification` - Technical specifications

## Prerequisites

- Python 3.13+
- Google Gemini API key
- UV package manager

## Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd emply-assignment
   ```

2. **Install UV** (if not already installed)

   ```bash
   # On Windows (PowerShell)
   irm https://astral.sh/uv/install.ps1 | iex

   # On macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install dependencies using UV**

   ```bash
   uv sync
   ```

4. **Set up environment variables**

   ```bash
   # Copy the template
   cp .env.template .env

   # Edit .env and add your Google Gemini API key
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

## Usage

### Basic Usage

1. **Prepare your bid documents**

   - Place PDF and HTML files in the `Bid1/` and `Bid2/` directories
   - The system will automatically process all supported files

2. **Run the extraction pipeline**
   ```bash
   uv run python main.py
   ```

### Configuration

You can modify the following settings in `main.py`:

```python
# Bid directories to process
BID_DIRECTORIES = ["Bid2"]  # Add "Bid1" to process both

# Output directory for results
OUTPUT_DIR = "./output"

# ChromaDB storage path
CHROMA_DB_PATH = "./chroma_db"

# Supported file formats
SUPPORTED_FORMATS = ['.pdf', '.html', '.htm']

# Embedding model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# LLM model
GEMINI_MODEL = "gemini-2.5-pro"
```

## Output

The system generates several output files:

- `{directory}_rag_extraction.json` - Individual directory results
- `combined_rag_extraction.json` - Combined results from all directories
- `chroma_db_{directory}/` - Vector database for each directory

### Output Format

```json
{
  "directory": "Bid2",
  "extraction_method": "RAG with LangChain + Gemini",
  "model": "gemini-2.5-pro",
  "embedding_model": "BAAI/bge-small-en-v1.5",
  "extracted_data": {
    "bid_number": "JA-207652",
    "title": "Student and Staff Computing Devices",
    "due_date": "2024-01-15"
    // ... other fields
  }
}
```

## How It Works

1. **Document Processing**: Uses Docling to convert PDFs/HTML to structured format with OCR and table extraction
2. **Chunking**: Creates semantic chunks using HybridChunker
3. **Embedding**: Generates embeddings using BAAI/bge-small-en-v1.5 model
4. **Vector Storage**: Stores chunks in ChromaDB for similarity search
5. **Multi-Phase Extraction**: Uses 5 progressive queries to extract all 20 fields:
   - Phase 1: Basic information and product overview
   - Phase 2: Contract terms and process requirements
   - Phase 3: Refinement search for missing fields
6. **AI Processing**: Google Gemini 2.5 Pro processes each query with relevant context
7. **Structured Output**: Results are validated and formatted as JSON

## Dependencies

- **Document Processing**: Docling for PDF/HTML parsing
- **Vector Store**: ChromaDB for embeddings storage
- **Embeddings**: Sentence Transformers (BAAI/bge-small-en-v1.5)
- **LLM**: Google Gemini 2.5 Pro via LangChain
- **Data Validation**: Pydantic for structured data models
- **Environment**: Python-dotenv for configuration

## Troubleshooting

### Common Issues

1. **Google API Key Error**

   - Ensure your `GOOGLE_API_KEY` is set in `.env`
   - Verify the API key has Gemini access enabled

2. **Memory Issues with Large Documents**

   - The system processes documents in chunks
   - Consider reducing the number of documents processed simultaneously

3. **No Files Found**

   - Check that files are in the correct directories (`Bid1/`, `Bid2/`)
   - Ensure files have supported extensions (`.pdf`, `.html`, `.htm`)

4. **ChromaDB Issues**
   - Delete the `chroma_db*` directories to reset the vector store
   - Ensure write permissions in the project directory

### Performance Tips

- The system includes 3-second delays between API calls to avoid rate limiting
- Processing time depends on document size and complexity
- Results are cached in ChromaDB for faster subsequent runs

## License

This project is part of an assignment for document processing and RAG implementation.
