import re
from typing import Dict, Any, List
import html
from datetime import datetime

class TextProcessor:
    def __init__(self):
        # Common patterns to clean
        self.html_pattern = re.compile(r'<[^>]+>')
        self.multiple_spaces = re.compile(r'\s+')
        self.multiple_newlines = re.compile(r'\n\s*\n')
        
        # Common medical abbreviations to expand
        self.medical_abbreviations = {
            'e.g.': 'for example',
            'i.e.': 'that is',
            'vs.': 'versus',
            'w/': 'with',
            'w/o': 'without',
            'q.d.': 'once daily',
            'b.i.d.': 'twice daily',
            't.i.d.': 'three times daily',
            'q.i.d.': 'four times daily',
            'q.h.': 'every hour',
            'q.4h.': 'every 4 hours',
            'q.6h.': 'every 6 hours',
            'q.8h.': 'every 8 hours',
            'q.12h.': 'every 12 hours',
            'q.24h.': 'every 24 hours',
            'p.o.': 'by mouth',
            'p.r.': 'by rectum',
            'i.v.': 'intravenous',
            'i.m.': 'intramuscular',
            's.c.': 'subcutaneous',
            'p.r.n.': 'as needed',
            'stat': 'immediately',
            'N/A': 'not available',
            'N/A.': 'not available',
            'N/A,': 'not available',
            'N/A;': 'not available',
            'N/A:': 'not available',
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text or text == 'N/A':
            return 'N/A'
            
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove HTML tags
        text = self.html_pattern.sub(' ', text)
        
        # Replace multiple spaces with single space
        text = self.multiple_spaces.sub(' ', text)
        
        # Replace multiple newlines with double newline
        text = self.multiple_newlines.sub('\n\n', text)
        
        # Expand medical abbreviations
        for abbr, expansion in self.medical_abbreviations.items():
            text = text.replace(abbr, expansion)
        
        # Clean up any remaining whitespace
        text = text.strip()
        
        return text
    
    def format_date(self, date_str: str) -> str:
        """Format date string to a consistent format."""
        if not date_str or date_str == 'N/A':
            return 'N/A'
            
        try:
            # Try parsing the date
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            return date_obj.strftime('%B %d, %Y')
        except ValueError:
            return date_str
    
    def process_trial(self, trial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single clinical trial's text data."""
        processed_data = trial_data.copy()
        
        # Clean text fields
        text_fields = [
            'title', 'official_title', 'brief_summary', 'detailed_description',
            'eligibility_criteria', 'why_stopped'
        ]
        
        for field in text_fields:
            if field in processed_data:
                processed_data[field] = self.clean_text(processed_data[field])
        
        # Clean and format dates
        date_fields = ['start_date', 'completion_date', 'last_update']
        for field in date_fields:
            if field in processed_data:
                processed_data[field] = self.format_date(processed_data[field])
        
        # Clean lists of text
        list_fields = ['conditions', 'collaborators', 'facility']
        for field in list_fields:
            if field in processed_data and isinstance(processed_data[field], list):
                processed_data[field] = [
                    self.clean_text(item) for item in processed_data[field]
                ]
        
        # Clean intervention data
        if 'intervention_names' in processed_data:
            processed_data['intervention_names'] = [
                self.clean_text(name) for name in processed_data['intervention_names']
            ]
        if 'intervention_descriptions' in processed_data:
            processed_data['intervention_descriptions'] = [
                self.clean_text(desc) for desc in processed_data['intervention_descriptions']
            ]
        
        # Clean outcomes data
        for outcome_type in ['primary_outcomes', 'secondary_outcomes']:
            if outcome_type in processed_data:
                for outcome in processed_data[outcome_type]:
                    if isinstance(outcome, dict):
                        for key in outcome:
                            if isinstance(outcome[key], str):
                                outcome[key] = self.clean_text(outcome[key])
        
        return processed_data
    
    def process_trials_batch(self, trials_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple clinical trials."""
        return [self.process_trial(trial) for trial in trials_data] 