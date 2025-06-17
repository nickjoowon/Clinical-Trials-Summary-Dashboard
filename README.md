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

3. **Set up your OpenAI API key:**:
   - Create a .env file in the project root with:
```
OPENAI_API_KEY=sk-<your-openai-api-key-here>
```

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
│   └── embeddings/          
├── requirements.txt
└── README.md
```

## Usage

1. **Run the Streamlit app**:
```bash
streamlit run app/app.py
```

2. **Access the dashboard**:
   - Open your browser and go to `http://localhost:8501`
   - Use the chat interface to ask questions about clinical trials
   - Explore the statistics features

## Features in Detail

### Chat Interface
- Natural language queries about clinical trials
- Context-aware responses using RAG


### Statistics
- Study type distribution
- Status tracking
- Interactive visualizations

## Acknowledgments

- [OpenAI](https://openai.com/) for providing the local LLM
- [Streamlit](https://streamlit.io/) for the web interface
- [ChromaDB](https://www.trychroma.com/) for vector storage