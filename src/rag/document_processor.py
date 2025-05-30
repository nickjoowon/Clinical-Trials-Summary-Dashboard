from typing import List, Dict, Any
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class ClinicalTrialProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def process_trial(self, trial_data: Dict[str, Any]) -> List[Document]:
        """Process a single clinical trial into documents."""
        # Create a comprehensive text representation of the trial
        trial_text = f"""
        Title: {trial_data.get('title', 'N/A')}
        Official Title: {trial_data.get('official_title', 'N/A')}
        NCT ID: {trial_data.get('nct_id', 'N/A')}
        Organization: {trial_data.get('organization_full_name', 'N/A')}
        
        Status Information:
        - Current Status: {trial_data.get('status', 'N/A')}
        - Start Date: {trial_data.get('start_date', 'N/A')}
        - Completion Date: {trial_data.get('completion_date', 'N/A')}
        - Last Update: {trial_data.get('last_update', 'N/A')}
        - Why Stopped: {trial_data.get('why_stopped', 'N/A')}
        
        Study Information:
        - Study Type: {trial_data.get('study_type', 'N/A')}
        - Study Phase: {', '.join(trial_data.get('study_phase', ['N/A']))}
        - Design Allocation: {trial_data.get('design_allocation', 'N/A')}
        - Intervention Model: {trial_data.get('intervention_study_design', 'N/A')}
        - Primary Purpose: {trial_data.get('design_primary_purpose', 'N/A')}
        - Time Perspective: {trial_data.get('design_time_perspective', 'N/A')}
        - Enrollment: {trial_data.get('enrollment', 'N/A')}
        
        Sponsor Information:
        - Lead Sponsor: {trial_data.get('sponsor', 'N/A')}
        - Collaborators: {', '.join(trial_data.get('collaborators', ['N/A']))}
        
        Oversight Information:
        - Has DMC: {trial_data.get('has_dmc', 'N/A')}
        - FDA Regulated Drug: {trial_data.get('is_fda_regulated_drug', 'N/A')}
        - FDA Regulated Device: {trial_data.get('is_fda_regulated_device', 'N/A')}
        - Unapproved Device: {trial_data.get('is_unapproved_device', 'N/A')}
        - PPSD: {trial_data.get('is_ppsd', 'N/A')}
        - US Export: {trial_data.get('is_us_export', 'N/A')}
        
        Description:
        Brief Summary:
        {trial_data.get('brief_summary', 'N/A')}
        
        Detailed Description:
        {trial_data.get('detailed_description', 'N/A')}
        
        Conditions:
        {', '.join(trial_data.get('conditions', ['N/A']))}
        
        Interventions:
        {self._format_interventions(trial_data)}
        
        Outcomes:
        {self._format_outcomes(trial_data)}
        
        Participant Flow:
        {self._format_participant_flow(trial_data)}
        
        Baseline Characteristics:
        {self._format_baseline_characteristics(trial_data)}
        
        Adverse Events:
        {self._format_adverse_events(trial_data)}
        
        Eligibility:
        Criteria:
        {trial_data.get('eligibility_criteria', 'N/A')}
        
        Additional Eligibility Information:
        - Gender: {trial_data.get('eligibility_gender', 'N/A')}
        - Age: {trial_data.get('eligibility_age', 'N/A')}
        - Healthy Volunteers: {trial_data.get('eligibility_healthy_volunteers', 'N/A')}
        - Healthy Volunteers Description: {trial_data.get('eligibility_healthy_volunteers_description', 'N/A')}
        
        Facilities:
        {', '.join(trial_data.get('facility', ['N/A']))}
        """
        
        # Split the text into chunks
        chunks = self.text_splitter.split_text(trial_text)
        
        # Create Document objects with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    'nct_id': trial_data.get('nct_id', 'N/A'),
                    'title': trial_data.get('title', 'N/A'),
                    'chunk_id': i,
                    'status': trial_data.get('status', 'N/A'),
                    'phase': ', '.join(trial_data.get('study_phase', ['N/A'])),
                    'study_type': trial_data.get('study_type', 'N/A'),
                    'organization': trial_data.get('organization_full_name', 'N/A'),
                    'sponsor': trial_data.get('sponsor', 'N/A'),
                    'last_update': trial_data.get('last_update', 'N/A')
                }
            )
            documents.append(doc)
        
        return documents
    
    def _format_interventions(self, trial_data: Dict[str, Any]) -> str:
        """Format intervention information into a readable string."""
        interventions = []
        for i in range(len(trial_data.get('intervention_names', []))):
            intervention = f"""
            Intervention {i+1}:
            - Type: {trial_data.get('intervention_types', ['N/A'])[i]}
            - Name: {trial_data.get('intervention_names', ['N/A'])[i]}
            - Description: {trial_data.get('intervention_descriptions', ['N/A'])[i]}
            """
            interventions.append(intervention)
        return '\n'.join(interventions) if interventions else 'N/A'
    
    def _format_outcomes(self, trial_data: Dict[str, Any]) -> str:
        """Format outcomes information into a readable string."""
        outcomes = []
        
        # Format primary outcomes
        primary_outcomes = trial_data.get('primary_outcomes', [])
        if primary_outcomes:
            outcomes.append("Primary Outcomes:")
            for outcome in primary_outcomes:
                formatted_outcome = f"""
                - Measure: {outcome.get('measure', 'N/A')}
                - Time Frame: {outcome.get('timeFrame', 'N/A')}
                - Description: {outcome.get('description', 'N/A')}
                """
                outcomes.append(formatted_outcome)
        
        # Format secondary outcomes
        secondary_outcomes = trial_data.get('secondary_outcomes', [])
        if secondary_outcomes:
            outcomes.append("\nSecondary Outcomes:")
            for outcome in secondary_outcomes:
                formatted_outcome = f"""
                - Measure: {outcome.get('measure', 'N/A')}
                - Time Frame: {outcome.get('timeFrame', 'N/A')}
                - Description: {outcome.get('description', 'N/A')}
                """
                outcomes.append(formatted_outcome)
        
        # Add outcome measures module information
        outcome_measure_type = trial_data.get('outcome_measure_type')
        outcome_measure_title = trial_data.get('outcome_measure_title')
        outcome_measure_time_frame = trial_data.get('outcome_measure_time_frame')
        
        if any([outcome_measure_type, outcome_measure_title, outcome_measure_time_frame]):
            outcomes.append("\nOutcome Measures Details:")
            outcomes.append(f"- Type: {outcome_measure_type or 'N/A'}")
            outcomes.append(f"- Title: {outcome_measure_title or 'N/A'}")
            outcomes.append(f"- Time Frame: {outcome_measure_time_frame or 'N/A'}")
        
        return '\n'.join(outcomes) if outcomes else 'N/A'
    
    def _format_participant_flow(self, trial_data: Dict[str, Any]) -> str:
        """Format participant flow information."""
        flow_info = []
        
        period_title = trial_data.get('period_title')
        milestone_title = trial_data.get('milestone_title')
        milestone_comment = trial_data.get('milestone_comment')
        num_periods = trial_data.get('num_of_periods')
        
        if any([period_title, milestone_title, milestone_comment, num_periods]):
            flow_info.append("Participant Flow Information:")
            flow_info.append(f"- Period Title: {period_title or 'N/A'}")
            flow_info.append(f"- Milestone Title: {milestone_title or 'N/A'}")
            flow_info.append(f"- Milestone Comment: {milestone_comment or 'N/A'}")
            flow_info.append(f"- Number of Periods: {num_periods or 'N/A'}")
        
        return '\n'.join(flow_info) if flow_info else 'N/A'
    
    def _format_baseline_characteristics(self, trial_data: Dict[str, Any]) -> str:
        """Format baseline characteristics information."""
        baseline_info = []
        
        population_desc = trial_data.get('baseline_analysis_population_description')
        arm_group_title = trial_data.get('arm_group_title')
        arm_group_desc = trial_data.get('arm_group_description')
        measure_title = trial_data.get('baseline_measure_title')
        
        if any([population_desc, arm_group_title, arm_group_desc, measure_title]):
            baseline_info.append("Baseline Characteristics:")
            baseline_info.append(f"- Population Description: {population_desc or 'N/A'}")
            baseline_info.append(f"- Arm Group Title: {arm_group_title or 'N/A'}")
            baseline_info.append(f"- Arm Group Description: {arm_group_desc or 'N/A'}")
            baseline_info.append(f"- Measure Title: {measure_title or 'N/A'}")
        
        return '\n'.join(baseline_info) if baseline_info else 'N/A'
    
    def _format_adverse_events(self, trial_data: Dict[str, Any]) -> str:
        """Format adverse events information."""
        ae_info = []
        
        arm_group_title = trial_data.get('adverse_events_arm_group_title')
        serious_num_affected = trial_data.get('num_affected_by_serious_adverse_event')
        serious_num_at_risk = trial_data.get('num_at_risk_for_serious_adverse_event')
        other_num_affected = trial_data.get('num_affected_by_other_adverse_event')
        other_num_at_risk = trial_data.get('num_at_risk_for_other_adverse_event')
        
        if any([arm_group_title, serious_num_affected, serious_num_at_risk, other_num_affected, other_num_at_risk]):
            ae_info.append("Adverse Events Information:")
            ae_info.append(f"- Arm Group Title: {arm_group_title or 'N/A'}")
            ae_info.append(f"- Serious Events: {serious_num_affected or 'N/A'} affected out of {serious_num_at_risk or 'N/A'} at risk")
            ae_info.append(f"- Other Events: {other_num_affected or 'N/A'} affected out of {other_num_at_risk or 'N/A'} at risk")
        
        return '\n'.join(ae_info) if ae_info else 'N/A'
    
    def process_trials_batch(self, trials_data: List[Dict[str, Any]]) -> List[Document]:
        """Process multiple clinical trials into documents."""
        all_documents = []
        for trial in trials_data:
            documents = self.process_trial(trial)
            all_documents.extend(documents)
        return all_documents