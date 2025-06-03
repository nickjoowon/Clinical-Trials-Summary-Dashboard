import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime

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

# Set Seaborn style
sns.set_theme(style="whitegrid")
plt.style.use("seaborn-v0_8")

def get_trials_data(rag_manager):
    """Get all trials data from the vector store."""
    all_docs = rag_manager.vector_store.similarity_search("", k=1000)
    return all_docs

def create_trials_per_year_chart(docs):
    """Create a bar chart showing the number of clinical trials per year."""
    # Extract years from start dates
    years = []
    for doc in docs:
        start_date = doc.metadata.get('start_date')
        if start_date:
            try:
                year = datetime.strptime(start_date, '%Y-%m-%d').year
                years.append(year)
            except (ValueError, TypeError):
                continue
    
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
    sns.barplot(data=df, x='Year', y='Number of Trials', palette='viridis', ax=ax)
    
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
    """Create a pie chart showing the distribution of trials by phase."""
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
        'PHASE1': 1,
        'PHASE2': 2,
        'PHASE3': 3,
        'PHASE4': 4,
        'EARLY_PHASE1': 0,
        'N/A': 5
    }
    
    # Sort phases
    sorted_phases = sorted(
        phase_counts.items(),
        key=lambda x: phase_order.get(x[0], 999)  # Put unknown phases at the end
    )
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create pie chart with sorted data
    ax.pie(
        [count for _, count in sorted_phases],
        labels=[phase for phase, _ in sorted_phases],
        autopct='%1.1f%%',
        colors=sns.color_palette('Set3'),
        startangle=90
    )
    
    # Customize the plot
    ax.set_title("Distribution of Trials by Phase", pad=20)
    plt.axis('equal')
    
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
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create horizontal bar plot
    sns.barplot(data=df, y='Study Type', x='Count', palette='muted', ax=ax)
    
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
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar plot
    sns.barplot(data=df, x='Status', y='Count', palette='Set2', ax=ax)
    
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
    # Define topic categories and their keywords
    topic_categories = {
        'Cancer': ['cancer', 'tumor', 'carcinoma', 'leukemia', 'lymphoma', 'melanoma', 'sarcoma'],
        'Cardiovascular': ['heart', 'cardiac', 'cardiovascular', 'hypertension', 'blood pressure'],
        'Neurological': ['brain', 'neurological', 'stroke', 'parkinson', 'alzheimer', 'dementia'],
        'Respiratory': ['lung', 'respiratory', 'asthma', 'copd', 'pneumonia'],
        'Diabetes': ['diabetes', 'diabetic', 'glucose', 'insulin'],
        'Autoimmune': ['autoimmune', 'arthritis', 'lupus', 'multiple sclerosis', 'rheumatoid'],
        'Infectious Disease': ['infection', 'viral', 'bacterial', 'hiv', 'aids', 'covid'],
        'Mental Health': ['depression', 'anxiety', 'mental health', 'psychiatric', 'bipolar'],
        'Pediatric': ['pediatric', 'child', 'infant', 'neonatal'],
        'Geriatric': ['geriatric', 'elderly', 'aging', 'senior']
    }
    
    # Extract conditions from the page content
    conditions = []
    for doc in docs:
        content = doc.page_content.lower()
        # Look for conditions in different possible formats
        if 'conditions:' in content:
            try:
                after = content.split('conditions:')[1]
                # Split by lines, take the first non-empty line
                lines = [line.strip() for line in after.split('\n') if line.strip()]
                if lines:
                    for condition in lines[0].split(','):
                        condition = condition.strip()
                        if condition and condition != 'n/a':
                            conditions.append(condition)
            except:
                pass
        # Also check metadata for conditions
        if 'conditions' in doc.metadata:
            conditions.extend(doc.metadata['conditions'])
    # Remove duplicates and clean up conditions
    conditions = list(set(conditions))
    conditions = [c.strip() for c in conditions if c and c.lower() != 'n/a']
    # Categorize conditions into topics
    topic_counts = Counter()
    for condition in conditions:
        condition_lower = condition.lower()
        categorized = False
        for topic, keywords in topic_categories.items():
            if any(keyword in condition_lower for keyword in keywords):
                topic_counts[topic] += 1
                categorized = True
                break
        if not categorized:
            topic_counts['Other'] += 1
    # Create DataFrame
    df = pd.DataFrame({
        'Topic': list(topic_counts.keys()),
        'Count': list(topic_counts.values())
    }).sort_values('Count', ascending=False)  # Sort by count in descending order
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    # Create bar plot
    sns.barplot(data=df, x='Topic', y='Count', palette='rocket', ax=ax)
    # Customize the plot
    ax.set_title("Top Topics Being Studied", pad=20)
    ax.set_xlabel("Topic")
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
        st.header("Clinical Trials Analysis")
        
        # Get all trials data
        docs = get_trials_data(st.session_state.rag_manager)
        
        # Create a selection menu for graphs
        graph_options = {
            "Trials by Year": create_trials_per_year_chart,
            "Phase Distribution": create_phase_distribution_chart,
            "Study Types": create_study_type_chart,
            "Trial Status": create_status_distribution_chart,
            "Top Conditions": create_top_conditions_chart
        }
        
        selected_graph = st.selectbox(
            "Select a visualization:",
            options=list(graph_options.keys()),
            format_func=lambda x: x
        )
        
        # Display the selected graph
        st.subheader(selected_graph)
        fig = graph_options[selected_graph](docs)
        st.pyplot(fig)
        plt.close(fig)

if __name__ == "__main__":
    main()
