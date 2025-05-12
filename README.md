# Clinical Trials Summary Dashboard

A Streamlit-based dashboard for exploring and analyzing clinical trials data from ClinicalTrials.gov. The application uses semantic search and embeddings to provide intelligent insights about clinical trials.

## Features

- **Chat Interface**: Ask questions about clinical trials and get AI-powered responses
- **Semantic Search**: Search through clinical trials using natural language queries
- **Statistics Dashboard**: View distributions and trends in clinical trial data
- **Embedding-based Analysis**: Uses FAISS and sentence transformers for efficient similarity search

## Project Structure

```
Clinical-Trials-Summary-Dashboard/
├── app/
│   └── app.py              # Streamlit dashboard application
├── src/
│   ├── data/
│   │   └── clinical_trials.py    # Data fetching and preprocessing
│   ├── processing/
│   │   └── text_processor.py     # Text processing and chunking
│   └── embeddings/
│       └── embedding_manager.py  # Embedding creation and management
├── data/
│   └── embeddings/         # Directory for storing embeddings
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Clinical-Trials-Summary-Dashboard.git
cd Clinical-Trials-Summary-Dashboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit dashboard:
```bash
streamlit run app/app.py
```

2. Access the dashboard in your web browser at `http://localhost:8501`

3. Use the navigation sidebar to access different features:
   - **Chat**: Ask questions about clinical trials
   - **Search**: Search for specific trials
   - **Statistics**: View trial distributions and trends

## Features in Detail

### Chat Interface
- Ask natural language questions about clinical trials
- Get AI-powered responses based on relevant trial information
- View conversation history during your session

### Semantic Search
- Search using natural language queries
- Results ranked by semantic similarity
- View detailed trial information including:
  - Study type
  - Status
  - Sponsor
  - Dates
  - Full text content

### Statistics Dashboard
- View distributions of:
  - Study types
  - Trial statuses
- Explore raw trial data in an interactive table

## Technical Details

- Uses ClinicalTrials.gov API v2 for data fetching
- Implements FAISS for efficient similarity search
- Uses sentence-transformers for creating embeddings
- Streamlit for the web interface

## Dependencies

- streamlit>=1.29.0
- sentence-transformers>=2.2.2
- faiss-cpu>=1.7.4
- pandas>=2.1.0
- numpy>=1.24.0
- requests>=2.31.0
- nltk>=3.8.1

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.