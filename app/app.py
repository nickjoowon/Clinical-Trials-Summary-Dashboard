import streamlit as st
import pandas as pd
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.clinical_trials import fetch_clinical_trials, preprocess_trial_data
from src.rag.rag_manager import RAGManager

# Set page config
st.set_page_config(
    page_title="Clinical Trials Dashboard",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if 'rag_manager' not in st.session_state:
    st.session_state.rag_manager = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_rag_system():
    """Initialize the RAG system with clinical trials data."""
    try:
        # Create RAG manager
        st.session_state.rag_manager = RAGManager()
        
        # Fetch and process trials
        with st.spinner("Fetching clinical trials data..."):
            trials_data = fetch_clinical_trials(max_results=100)
            processed_trials = preprocess_trial_data(trials_data)
        
        # Add trials to RAG system
        with st.spinner("Processing trials for RAG system..."):
            st.session_state.rag_manager.add_trials(processed_trials)
        
        return True
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return False

def generate_response(query: str) -> str:
    """Generate a response using the RAG system."""
    try:
        response = st.session_state.rag_manager.get_response(query)
        return response
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}"

def main():
    st.title("Clinical Trials Dashboard")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Chat", "Search", "Statistics"])
    
    if page == "Chat":
        st.header("Chat with Clinical Trials Assistant")
        
        # Initialize RAG system if not already initialized
        if st.session_state.rag_manager is None:
            if not initialize_rag_system():
                st.error("Failed to initialize RAG system. Please check if Ollama is running.")
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
                with st.spinner("Thinking..."):
                    response = generate_response(prompt)
                    st.write(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Display usage statistics
        if st.session_state.rag_manager:
            with st.expander("Usage Statistics"):
                stats = st.session_state.rag_manager.get_usage_stats()
                st.write(f"Total Queries: {stats['total_queries']}")
                st.write(f"Total Tokens: {stats['total_tokens']}")
                st.write(f"Last Reset: {stats['last_reset']}")
    
    elif page == "Search":
        st.header("Search Clinical Trials")
        
        # Initialize RAG system if not already initialized
        if st.session_state.rag_manager is None:
            if not initialize_rag_system():
                st.error("Failed to initialize RAG system. Please check if Ollama is running.")
                return
        
        # Search interface
        query = st.text_input("Enter your search query:")
        k = st.slider("Number of results to show", 1, 20, 5)
        
        if query:
            with st.spinner("Searching..."):
                response = st.session_state.rag_manager.get_response(query)
                st.write(response)
    
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
