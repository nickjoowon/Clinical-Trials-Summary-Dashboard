# Clinical Trials Summary Dashboard

A Streamlit-based dashboard that provides an interactive interface for querying and analyzing clinical trials data using RAG (Retrieval-Augmented Generation) with Ollama's Mistral model.

## Features

- 🤖 AI-powered chat interface for querying clinical trials
- 🔍 Advanced search capabilities
- 📊 Interactive statistics and visualizations
- 💾 Local vector storage for efficient retrieval
- 🆓 Free LLM integration using Ollama's Mistral model

## Prerequisites

- Python 3.8+
- Ollama (for local LLM)
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

3. **Install and Setup Ollama**:
   - Download Ollama from [ollama.ai](https://ollama.ai/download)
   - Install and run Ollama
   - Pull the Mistral model:
```bash
ollama pull mistral
```

## Project Structure

```
Clinical-Trials-Summary-Dashboard/
├── app/
│   └── app.py              # Streamlit application
├── src/
│   ├── data/
│   │   └── clinical_trials.py
│   ├── rag/
│   │   ├── rag_manager.py
│   │   ├── document_processor.py
│   │   ├── text_processor.py
│   │   └── vector_store.py
│   └── prompts/
│       └── templates.py
├── data/
│   └── embeddings/         # Vector store directory
├── requirements.txt
└── README.md
```

## Usage

1. **Start Ollama**:
```bash
ollama serve
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
- Context-aware responses using RAG
- Conversation history tracking

### Search
- Semantic search across clinical trials
- Filtering and sorting options
- Detailed trial information display

### Statistics
- Study type distribution
- Status tracking
- Interactive visualizations

## Acknowledgments

- [Ollama](https://ollama.ai/) for providing the local LLM
- [Streamlit](https://streamlit.io/) for the web interface
- [ChromaDB](https://www.trychroma.com/) for vector storage