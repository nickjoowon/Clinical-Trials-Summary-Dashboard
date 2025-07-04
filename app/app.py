import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables
load_dotenv()

# Get the absolute path of the project root directory
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

# Now import the modules
from src.rag.rag_manager import RAGManager
from src.data.clinical_trials import fetch_clinical_trials, preprocess_trial_data
from langchain.schema import Document

# Set page config
st.set_page_config(
    page_title="Clinical Trials Summary Dashboard",
    page_icon="🏥",
    layout="wide"
)

# Set Seaborn style
sns.set_theme(style="whitegrid")
plt.style.use("seaborn-v0_8")

def get_trials_data(rag_manager):
    """Get all trials data from the vector store."""
    if rag_manager is None:
        st.error("RAG system is not properly initialized. Please check your OpenAI API key and try again.")
        return []
    
    try:
        # Get all documents from the vector store using the get() method
        # This retrieves ALL documents without any limit
        all_data = rag_manager.vector_store.get()
        
        # Convert the raw data back to Document objects
        documents = []
        for i in range(len(all_data["ids"])):
            doc = Document(
                page_content=all_data["documents"][i],
                metadata=all_data["metadatas"][i] if all_data["metadatas"] else {}
            )
            documents.append(doc)
        
        return documents
    except Exception as e:
        st.error(f"Error retrieving documents: {str(e)}")
        return []

def create_trials_per_year_chart(docs):
    """Create a bar chart showing the number of clinical trials per year."""
    # Extract years from start dates in the metadata
    years = []
    for doc in docs:
        start_date = doc.metadata.get('start_date')
        if start_date and start_date != 'N/A':
            try:
                # Try different date formats
                for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y']:
                    try:
                        year = datetime.strptime(start_date, fmt).year
                        years.append(year)
                        break
                    except ValueError:
                        continue
            except Exception as e:
                continue
    
    if not years:
        # Create empty chart with message
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No valid start dates found in the data', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title("Number of Clinical Trials by Year", pad=20)
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Trials")
        plt.tight_layout()
        return fig
    
    # Count trials per year
    year_counts = Counter(years)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Year': list(year_counts.keys()),
        'Number of Trials': list(year_counts.values())
    }).sort_values('Year')
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bar plot
    sns.barplot(data=df, x='Year', y='Number of Trials', palette='flare', ax=ax)
    
    # Customize the plot
    ax.set_title("Number of Clinical Trials by Year", pad=20)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Trials")
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for i, v in enumerate(df['Number of Trials']):
        ax.text(i, v, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def create_phase_distribution_chart(docs):
    """Create a bar chart showing the distribution of trials by phase."""
    # Extract phases
    phases = []
    for doc in docs:
        phase = doc.metadata.get('phase', 'N/A')
        if phase and phase != 'N/A':
            # Split multiple phases if present
            for p in phase.split(','):
                phases.append(p.strip())
    # Count phases
    phase_counts = Counter(phases)
    # Define phase order
    phase_order = {
        'EARLY_PHASE1': 0,
        'PHASE1': 1,
        'PHASE2': 2,
        'PHASE3': 3,
        'PHASE4': 4,
        'N/A': 5
    }
    # Sort phases
    sorted_phases = sorted(
        phase_counts.items(),
        key=lambda x: phase_order.get(x[0], 999)  # Put unknown phases at the end
    )
    # Create DataFrame
    df = pd.DataFrame({
        'Phase': [phase for phase, _ in sorted_phases],
        'Count': [count for _, count in sorted_phases]
    })
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    # Create bar plot
    sns.barplot(data=df, x='Phase', y='Count', palette='crest', ax=ax)
    # Customize the plot
    ax.set_title("Distribution of Trials by Phase", pad=20)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Number of Trials")
    # Add value labels on top of bars
    for i, v in enumerate(df['Count']):
        ax.text(i, v, str(v), ha='center', va='bottom')
    plt.tight_layout()
    return fig

def create_study_type_chart(docs):
    """Create a horizontal bar chart showing the distribution of study types."""
    # Extract study types
    study_types = [doc.metadata.get('study_type', 'N/A') for doc in docs]
    
    # Count study types
    type_counts = Counter(study_types)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Study Type': list(type_counts.keys()),
        'Count': list(type_counts.values())
    }).sort_values('Count', ascending=True)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create horizontal bar plot
    sns.barplot(data=df, y='Study Type', x='Count', palette='Set2', ax=ax)
    
    # Customize the plot
    ax.set_title("Distribution of Study Types", pad=20)
    ax.set_xlabel("Number of Trials")
    ax.set_ylabel("")
    
    # Add value labels
    for i, v in enumerate(df['Count']):
        ax.text(v, i, str(v), ha='left', va='center')
    
    plt.tight_layout()
    return fig

def create_status_distribution_chart(docs):
    """Create a vertical bar chart showing the distribution of trial statuses."""
    # Extract statuses
    statuses = [doc.metadata.get('status', 'N/A') for doc in docs]
    
    # Count statuses
    status_counts = Counter(statuses)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Status': list(status_counts.keys()),
        'Count': list(status_counts.values())
    }).sort_values('Count', ascending=False)  # Sort by count in descending order
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bar plot
    sns.barplot(data=df, x='Status', y='Count', palette='pastel', ax=ax)
    
    # Customize the plot
    ax.set_title("Distribution of Trial Statuses", pad=20)
    ax.set_xlabel("Status")
    ax.set_ylabel("Number of Trials")
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for i, v in enumerate(df['Count']):
        ax.text(i, v, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def create_top_conditions_chart(docs):
    """Create a vertical bar chart showing the top topics being studied."""
    # Extract conditions from the document content more effectively
    conditions = []
    for doc in docs:
        # First try to get conditions from metadata
        if 'conditions' in doc.metadata:
            conditions_str = doc.metadata['conditions']
            if conditions_str and conditions_str != 'N/A':
                # Split by comma and clean up
                for condition in conditions_str.split(','):
                    condition = condition.strip()
                    if condition and condition.lower() != 'n/a':
                        conditions.append(condition)
        
        # If no conditions in metadata, try to extract from content
        if not conditions:
            content = doc.page_content
            
            # Look for the conditions section in the structured text
            if 'Conditions:' in content:
                try:
                    # Find the conditions section and get the next line
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if 'Conditions:' in line:
                            # Get the next non-empty line after "Conditions:"
                            for j in range(i + 1, min(i + 3, len(lines))):
                                next_line = lines[j].strip()
                                if next_line and next_line != 'N/A':
                                    # Split by comma and clean up
                                    for condition in next_line.split(','):
                                        condition = condition.strip()
                                        if condition and condition.lower() != 'n/a' and len(condition) > 2:
                                            conditions.append(condition)
                                    break
                            break
                except Exception as e:
                    continue
    
    # Clean up and count conditions
    conditions = [c.strip() for c in conditions if c and len(c) > 2 and c.lower() != 'n/a']
    
    # If still no conditions, try to extract from the content more broadly
    if not conditions:
        for doc in docs:
            content = doc.page_content
            
            # Look for conditions in the structured format from process_trial
            if 'Conditions:' in content:
                try:
                    # Find the line that contains conditions
                    lines = content.split('\n')
                    for line in lines:
                        if line.strip().startswith('Conditions:'):
                            # Extract everything after "Conditions:"
                            condition_text = line.split('Conditions:')[1].strip()
                            if condition_text and condition_text != 'N/A':
                                # Split by comma and clean up
                                for condition in condition_text.split(','):
                                    condition = condition.strip()
                                    if condition and condition.lower() != 'n/a' and len(condition) > 2:
                                        conditions.append(condition)
                            break
                except Exception as e:
                    continue
    
    # If still no conditions, create a placeholder
    if not conditions:
        # Create empty chart with message
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No condition data found in the trials', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title("Top Conditions Being Studied", pad=20)
        ax.set_xlabel("Condition")
        ax.set_ylabel("Number of Trials")
        plt.tight_layout()
        return fig
    
    # Count conditions and get top ones
    condition_counts = Counter(conditions)
    
    # Get top 10 conditions
    top_conditions = condition_counts.most_common(10)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Condition': [condition for condition, _ in top_conditions],
        'Count': [count for _, count in top_conditions]
    })
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bar plot
    sns.barplot(data=df, x='Condition', y='Count', palette='dark', ax=ax)
    
    # Customize the plot
    ax.set_title("Top Conditions Being Studied", pad=20)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Number of Trials")
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for i, v in enumerate(df['Count']):
        ax.text(i, v, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def initialize_rag_system():
    """Initialize the RAG system with clinical trials data."""
    try:
        # Create RAG manager
        rag_manager = RAGManager()
        
        # Check if we already have data in the vector store
        db_stats = rag_manager.get_database_stats()
        
        if db_stats.get('total_documents', 0) > 0:
            st.success(f"Using existing clinical trials data: {db_stats.get('total_documents', 0)} documents loaded.")
            return rag_manager
        
        # If no existing data, show instructions
        else:
            st.warning("No clinical trials data found in the vector store.")
            st.info("""
            To load clinical trials data, run the data pipeline first:
            
            ```bash
            python data_pipeline.py --start-date 2024-01-01 --max-results 10000
            ```
            
            Or for all trials from 2024:
            ```bash
            python data_pipeline.py --start-date 2024-01-01
            ```
            
            Then restart this application.
            """)
            
            # Option to run pipeline in background (optional)
            if st.button("Run Data Pipeline Now (This may take a while)"):
                with st.spinner("Running data pipeline..."):
                    try:
                        from data_pipeline import ClinicalTrialsDataPipeline
                        pipeline = ClinicalTrialsDataPipeline()
                        success = pipeline.run_pipeline(start_date="2024-01-01")
                        if success:
                            st.success("Data pipeline completed! Refreshing...")
                            st.rerun()
                        else:
                            st.error("Data pipeline failed. Check the logs.")
                    except Exception as e:
                        st.error(f"Error running pipeline: {str(e)}")
            
            return None
        
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return None

# Remove the sidebar data ingestion controls since we're using the pipeline
# Initialize session state
if 'rag_manager' not in st.session_state:
    st.session_state.rag_manager = initialize_rag_system()
    if st.session_state.rag_manager is None:
        st.error("Failed to initialize RAG system. Please run the data pipeline first.")

def generate_response(query: str) -> str:
    """Generate a response using the RAG system."""
    try:
        # Get structured query
        structured_query = st.session_state.rag_manager.query_analyzer.invoke(
            {"question": query}
        )
        
        # Display the structured query for transparency
        st.write("Search Parameters:")
        st.json(structured_query.dict())
        
        # Get and return response
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
    
    elif page == "Search":
        st.header("Search Clinical Trials")
        
        # Search input
        search_query = st.text_input("Enter your search query")
        if search_query:
            # Get structured query
            structured_query = st.session_state.rag_manager.query_analyzer.invoke(
                {"question": search_query}
            )
            
            # Display the structured query for transparency
            st.write("Search Parameters:")
            st.json(structured_query.dict())
            
            # Get and display results
            results = st.session_state.rag_manager.vector_store.similarity_search(
                structured_query.content_search,
                k=5,
                filter=structured_query.dict()
            )
            
            for i, result in enumerate(results, 1):
                st.markdown(f"**Result {i}:**")
                st.markdown(result.page_content)
                st.markdown("---")
    
    elif page == "Statistics":
        st.header("Statistics")
        
        # Check if RAG manager is properly initialized
        if st.session_state.rag_manager is None:
            st.error("RAG system is not properly initialized. Please check your OpenAI API key and try again.")
            return
        
        # Get all trials data
        all_docs = get_trials_data(st.session_state.rag_manager)
        
        # Debug information
        st.info(f"Retrieved {len(all_docs)} documents from the database")
        
        # Show database stats
        db_stats = st.session_state.rag_manager.get_database_stats()
        st.info(f"Database contains {db_stats.get('total_documents', 0)} total documents")
        
        if not all_docs:  # If docs is empty
            st.warning("No clinical trials data available. Please ensure the system is properly initialized.")
            return
        
        # --- Year Filter above charts ---
        st.markdown("---")
        
        # Extract all years from the docs
        years = []
        for doc in all_docs:
            start_date = doc.metadata.get('start_date')
            if start_date and start_date != 'N/A':
                try:
                    # Try different date formats
                    for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y']:
                        try:
                            year = datetime.strptime(start_date, fmt).year
                            years.append(year)
                            break
                        except ValueError:
                            continue
                except Exception as e:
                    continue
        
        years = sorted(set(years))
        year_options = ['All Years'] + [str(y) for y in years]
        
        selected_year = st.selectbox(
            'Filter by Year', 
            options=year_options,
            help="Select a specific year to filter all charts, or 'All Years' to show all data"
        )
        
        # Filter docs if a specific year is selected
        if selected_year != 'All Years':
            filtered_docs = []
            for doc in all_docs:
                start_date = doc.metadata.get('start_date')
                if start_date and start_date != 'N/A':
                    try:
                        for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y']:
                            try:
                                year = datetime.strptime(start_date, fmt).year
                                if str(year) == selected_year:
                                    filtered_docs.append(doc)
                                break
                            except ValueError:
                                continue
                    except Exception as e:
                        continue
            st.success(f"Showing data for {selected_year}: {len(filtered_docs)} trials")
        else:
            filtered_docs = all_docs
            st.info(f"Showing all years: {len(filtered_docs)} trials")
        
        # Show filter summary
        st.markdown(f"### 📈 Statistics for {selected_year}")
        st.markdown(f"**Total trials in view:** {len(filtered_docs)}")
        
        # Show Trials by Year chart only when "All Years" is selected
        if selected_year == 'All Years':
            st.write("### 📊 Trials by Year")
            fig_year = create_trials_per_year_chart(filtered_docs)
            st.pyplot(fig_year)
            plt.close(fig_year)
        
        # Two columns with charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Phase Distribution")
            fig_phase = create_phase_distribution_chart(filtered_docs)
            st.pyplot(fig_phase)
            plt.close(fig_phase)
            
            st.write("### Study Types")
            fig_type = create_study_type_chart(filtered_docs)
            st.pyplot(fig_type)
            plt.close(fig_type)
        
        with col2:
            st.write("### Trial Status")
            fig_status = create_status_distribution_chart(filtered_docs)
            st.pyplot(fig_status)
            plt.close(fig_status)
            
            st.write("### Top Conditions")
            fig_conditions = create_top_conditions_chart(filtered_docs)
            st.pyplot(fig_conditions)
            plt.close(fig_conditions)

if __name__ == "__main__":
    main()
