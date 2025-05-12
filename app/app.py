import streamlit as st
import pandas as pd
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.clinical_trials import fetch_clinical_trials, preprocess_trial_data
from src.processing.text_processor import prepare_trials_for_embedding
from src.embeddings.embedding_manager import EmbeddingManager

# Set page config
st.set_page_config(
    page_title="Clinical Trials Dashboard",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if 'embedding_manager' not in st.session_state:
    st.session_state.embedding_manager = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def load_or_create_embeddings():
    """Load existing embeddings or create new ones."""
    embedding_dir = "data/embeddings"
    
    if os.path.exists(os.path.join(embedding_dir, "index.faiss")):
        # Load existing embeddings
        st.session_state.embedding_manager = EmbeddingManager()
        st.session_state.embedding_manager.load_index(embedding_dir)
        return True
    else:
        # Create new embeddings
        with st.spinner("Fetching clinical trials data..."):
            trials_data = fetch_clinical_trials(max_results=100)
            processed_trials = preprocess_trial_data(trials_data)
            
        with st.spinner("Preparing data for embedding..."):
            embedding_data = prepare_trials_for_embedding(processed_trials)
            
        with st.spinner("Creating embeddings..."):
            st.session_state.embedding_manager = EmbeddingManager()
            texts = [item['text'] for item in embedding_data]
            embeddings = st.session_state.embedding_manager.create_embeddings(texts)
            
            st.session_state.embedding_manager.build_index(embeddings)
            st.session_state.embedding_manager.add_metadata(embedding_data)
            st.session_state.embedding_manager.save_index(embedding_dir)
        
        return True

def generate_response(query: str) -> str:
    """Generate a response based on the query and relevant trial information."""
    # Search for relevant trials
    results = st.session_state.embedding_manager.search(query, k=3)
    
    # Format the response
    response = f"Based on the clinical trials data, here's what I found:\n\n"
    
    for i, result in enumerate(results, 1):
        response += f"**Trial {i}: {result['metadata']['title']}**\n"
        response += f"- Study Type: {result['metadata']['study_type']}\n"
        response += f"- Status: {result['metadata']['status']}\n"
        response += f"- Sponsor: {result['metadata']['sponsor']}\n"
        response += f"- Key Information: {result['text'][:200]}...\n\n"
    
    return response

def main():
    st.title("Clinical Trials Dashboard")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Chat", "Search", "Statistics"])
    
    if page == "Chat":
        st.header("Chat with Clinical Trials Assistant")
        
        # Load embeddings if not already loaded
        if st.session_state.embedding_manager is None:
            if not load_or_create_embeddings():
                st.error("Failed to load or create embeddings")
                return
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about clinical trials..."):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                response = generate_response(prompt)
                st.write(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    elif page == "Search":
        st.header("Search Clinical Trials")
        
        # Load embeddings if not already loaded
        if st.session_state.embedding_manager is None:
            if not load_or_create_embeddings():
                st.error("Failed to load or create embeddings")
                return
        
        # Search interface
        query = st.text_input("Enter your search query:")
        k = st.slider("Number of results to show", 1, 20, 5)
        
        if query:
            results = st.session_state.embedding_manager.search(query, k=k)
            
            for i, result in enumerate(results, 1):
                with st.expander(f"{i}. {result['metadata']['title']} (Score: {result['score']:.2f})"):
                    st.write("**Text:**")
                    st.write(result['text'])
                    st.write("**Metadata:**")
                    st.write(f"- Study Type: {result['metadata']['study_type']}")
                    st.write(f"- Status: {result['metadata']['status']}")
                    st.write(f"- Sponsor: {result['metadata']['sponsor']}")
                    st.write(f"- Start Date: {result['metadata']['start_date']}")
                    st.write(f"- Completion Date: {result['metadata']['completion_date']}")
    
    else:  # Statistics page
        st.header("Clinical Trials Statistics")
        
        # Load data
        trials_data = fetch_clinical_trials(max_results=100)
        processed_trials = preprocess_trial_data(trials_data)
        df = pd.DataFrame(processed_trials)
        
        # Display basic statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Study Types")
            study_type_counts = df['study_type'].value_counts()
            st.bar_chart(study_type_counts)
        
        with col2:
            st.subheader("Status Distribution")
            status_counts = df['status'].value_counts()
            st.bar_chart(status_counts)
        
        # Display raw data
        st.subheader("Raw Data")
        st.dataframe(df)

if __name__ == "__main__":
    main()
