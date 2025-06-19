# Clinical Trials Summary Dashboard

A Streamlit-based dashboard that provides an interactive interface for querying and analyzing clinical trials data using Retrieval-Augmented Generation (RAG) with OpenAI's GPT models and Chroma vector store.

## Features

- AI-powered chat interface for querying clinical trials
- Interactive statistics and visualizations
- Local vector storage for efficient retrieval
- OpenAI LLM integration for better quality answers

## Prerequisites

- Python 3.8+
- OpenAI API key
- Git

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/Clinical-Trials-Summary-Dashboard.git
cd Clinical-Trials-Summary-Dashboard
```

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up your OpenAI API key**:
   - Create a `.env` file in the project root with:
```
OPENAI_API_KEY=sk-<your-openai-api-key-here>
```

## Data Pipeline

Before running the dashboard, you need to populate the vector store with clinical trials data using the data pipeline:

### Quick Start
```bash
# Fetch 10,000 trials from 2024 onwards
python data_pipeline.py --start-date 2024-01-01 --max-results 10000

# Fetch all trials from 2024 onwards (may take a while)
python data_pipeline.py --start-date 2024-01-01
```

### Pipeline Options
- `--start-date`: Start date for fetching trials (YYYY-MM-DD format)
- `--max-results`: Maximum number of trials to fetch (optional)
- `--force-refresh`: Clear existing data and re-ingest
- `--batch-size`: Batch size for processing (default: 500)

## Usage

1. **Run the data pipeline** (first time only):
```bash
python data_pipeline.py --start-date 2024-01-01 --max-results 10000
```

2. **Run the Streamlit app**:
```bash
streamlit run app/app.py
```

3. **Access the dashboard**:
   - Open your browser and go to `http://localhost:8501`
   - Use the chat interface to ask questions about clinical trials
   - Explore the search and statistics features

## Features in Detail

### Chat Interface
- Natural language queries about clinical trials
- Context-aware responses using RAG (OpenAI GPT + Chroma)
- Conversation history tracking

### Statistics
- Study type distribution
- Status tracking
- Interactive visualizations

## Project Structure

```
Clinical-Trials-Summary-Dashboard/
├── app/
│   └── app.py                  # Streamlit application
├── src/
│   ├── data/
│   │   └── clinical_trials.py  # Data fetching and preprocessing
│   ├── rag/
│   │   ├── rag_manager.py      # RAG pipeline manager (OpenAI + Chroma)
│   │   ├── document_processor.py
│   │   ├── query_analyzer.py
│   │   ├── text_processor.py
│   │   └── __init__.py
│   ├── prompts/
│   │   └── templates.py
│   └── test_functions.py
├── data/
│   ├── chroma_db/              # Chroma vector store directory
│   └── embeddings/             # (if used)
├── data_pipeline.py            # Bulk data ingestion pipeline
├── requirements.txt
└── README.md
```

## Acknowledgments

- [OpenAI](https://openai.com/) for LLM APIs
- [Streamlit](https://streamlit.io/) for the web interface
- [ChromaDB](https://www.trychroma.com/) for vector storage