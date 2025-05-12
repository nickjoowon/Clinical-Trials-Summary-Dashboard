import re
from typing import List, Dict, Any
import pandas as pd

def clean_text(text: str) -> str:
    """Clean and normalize text by removing special characters and extra whitespace."""
    if not isinstance(text, str):
        return ""
    
    # Remove special characters and normalize whitespace
    text = re.sub(r'[^\w\s.,;:!?()-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def combine_trial_text(trial: Dict[str, Any]) -> str:
    """Combine relevant text fields into a single document with clear section headers."""
    sections = []
    
    # Add title
    if trial.get('title'):
        sections.append(f"Title: {trial['title']}")
    
    # Add summary
    if trial.get('brief_summary'):
        sections.append(f"Summary: {trial['brief_summary']}")
    
    # Add description
    if trial.get('detailed_description'):
        sections.append(f"Description: {trial['detailed_description']}")
    
    # Add conditions
    if trial.get('conditions'):
        conditions = ', '.join(trial['conditions'])
        sections.append(f"Conditions: {conditions}")
    
    # Add interventions
    if trial.get('interventions'):
        interventions = '; '.join(trial['interventions'])
        sections.append(f"Interventions: {interventions}")
    
    # Add eligibility criteria
    if trial.get('eligibility_criteria'):
        sections.append(f"Eligibility Criteria: {trial['eligibility_criteria']}")
    
    # Add study type and status
    study_info = []
    if trial.get('study_type'):
        study_info.append(f"Study Type: {trial['study_type']}")
    if trial.get('status'):
        study_info.append(f"Status: {trial['status']}")
    if study_info:
        sections.append(' | '.join(study_info))
    
    return '\n\n'.join(sections)

def chunk_text(text: str, max_tokens: int = 512) -> List[str]:
    """Split text into chunks based on sentence boundaries and maximum token count."""
    if not text:
        return []
    
    # Split into sentences using regex
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        # Rough estimate of tokens (words + punctuation)
        sentence_length = len(sentence.split())
        
        if current_length + sentence_length > max_tokens and current_chunk:
            # Join current chunk and add to chunks
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        
        current_chunk.append(sentence)
        current_length += sentence_length
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def prepare_trials_for_embedding(trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prepare trials data for embedding creation."""
    processed_trials = []
    
    for trial in trials:
        # Combine text fields
        combined_text = combine_trial_text(trial)
        
        # Clean the text
        cleaned_text = clean_text(combined_text)
        
        # Split into chunks
        chunks = chunk_text(cleaned_text)
        
        # Create entries for each chunk
        for i, chunk in enumerate(chunks):
            processed_trial = {
                'nct_id': trial['nct_id'],
                'chunk_id': f"{trial['nct_id']}_chunk_{i+1}",
                'text': chunk,
                'metadata': {
                    'title': trial.get('title'),
                    'study_type': trial.get('study_type'),
                    'status': trial.get('status'),
                    'sponsor': trial.get('sponsor'),
                    'start_date': trial.get('start_date'),
                    'completion_date': trial.get('completion_date')
                }
            }
            processed_trials.append(processed_trial)
    
    return processed_trials 