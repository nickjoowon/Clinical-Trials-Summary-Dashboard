import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path

# Get the absolute path of the project root directory
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

# Now import the modules
from src.rag.rag_manager import RAGManager
from src.data.clinical_trials import fetch_clinical_trials, preprocess_trial_data

# Set page config
st.set_page_config(
    page_title="Clinical Trials Summary Dashboard",
    page_icon="🏥",
    layout="wide"
)

def initialize_rag_system():
    """Initialize the RAG system with clinical trials data."""
    try:
        # Create RAG manager
        rag_manager = RAGManager()
        
        # Check if we already have data in the vector store
        db_stats = rag_manager.get_database_stats()
        
        if db_stats.get('total_documents', 0) > 0:
            st.success("Using existing clinical trials data from local storage.")
            return rag_manager
        
        # If no existing data, fetch and process new trials
        else:
            with st.spinner("Fetching clinical trials data..."):
                trials_data = fetch_clinical_trials(max_results=500)
                processed_trials = preprocess_trial_data(trials_data)
        
            # Add trials to RAG system
            with st.spinner("Processing trials for RAG system..."):
                rag_manager.add_trials(processed_trials)
                st.success("Successfully loaded and processed clinical trials data.")
        
        return rag_manager
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return None

# Initialize session state
if 'rag_manager' not in st.session_state:
    st.session_state.rag_manager = initialize_rag_system()
    if st.session_state.rag_manager is None:
        st.error("Failed to initialize RAG system. Please check if Ollama is running and try again.")

def generate_response(query: str) -> str:
    """Generate a response using the RAG system."""
    try:
        response = st.session_state.rag_manager.get_response(query)
        return response
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}"

def main():
    st.title("Clinical Trials Summary Dashboard")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Chat", "Search", "Statistics"])
    
    if page == "Chat":
        st.header("Chat with Clinical Trials Data")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about clinical trials"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                try:
                    response = generate_response(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Add download button for responses
                    st.download_button(
                        label="Download Response",
                        data=response,
                        file_name="clinical_trial_response.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    error_msg = f"Error processing your request: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        # Display usage statistics
        if st.session_state.rag_manager:
            with st.expander("Usage Statistics"):
                stats = st.session_state.rag_manager.get_usage_stats()
                st.write(f"Total Queries: {stats['total_queries']}")
                st.write(f"Total Tokens: {stats['total_tokens']}")
                st.write(f"Last Reset: {stats['last_reset']}")
    
    elif page == "Search":
        st.header("Search Clinical Trials")
        
        # Search input
        search_query = st.text_input("Enter your search query")
        if search_query:
            results = st.session_state.rag_manager.vector_store.similarity_search(search_query)
            for i, result in enumerate(results, 1):
                st.markdown(f"**Result {i}:**")
                st.markdown(result.page_content)
                st.markdown("---")
    
    elif page == "Statistics":
        st.header("Database Statistics")
        
        # Display database stats
        stats = st.session_state.rag_manager.get_database_stats()
        st.json(stats)
        
        # Display usage stats
        usage_stats = st.session_state.rag_manager.get_usage_stats()
        st.subheader("Usage Statistics")
        st.json(usage_stats)

if __name__ == "__main__":
    main()
