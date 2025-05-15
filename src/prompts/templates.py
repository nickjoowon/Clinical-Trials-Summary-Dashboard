from typing import List, Dict, Any

class PromptTemplates:
    @staticmethod
    def get_base_system_prompt() -> str:
        return """You are a clinical trials assistant. Your responses should be:
        - Accurate and based only on the provided clinical trial information
        - Clear and concise
        - Include relevant trial identifiers (NCT ID)
        - Cite specific details from the trials
        - If information is not available, clearly state that
        - Format responses in a structured, easy-to-read manner"""

    @staticmethod
    def get_status_query_prompt(trials: List[Dict[str, Any]]) -> str:
        context = "\n\n".join([
            f"Trial: {trial.get('title', 'N/A')} (NCT ID: {trial.get('nct_id', 'N/A')})\n"
            f"Status: {trial.get('status', 'N/A')}\n"
            f"Start Date: {trial.get('start_date', 'N/A')}\n"
            f"Completion Date: {trial.get('completion_date', 'N/A')}\n"
            f"Last Update: {trial.get('last_update', 'N/A')}\n"
            f"Why Stopped: {trial.get('why_stopped', 'N/A')}"
            for trial in trials
        ])
        
        return f"""System: {PromptTemplates.get_base_system_prompt()}
        When asked about trial status, include:
        - Current status
        - Start and completion dates
        - Last update date
        - Any termination reasons if applicable
        - Enrollment status if available

        Context:
        {context}

        User Query: [User's question about trial status]

        Assistant: [Structured response]"""

    @staticmethod
    def get_eligibility_query_prompt(trials: List[Dict[str, Any]]) -> str:
        context = "\n\n".join([
            f"Trial: {trial.get('title', 'N/A')} (NCT ID: {trial.get('nct_id', 'N/A')})\n"
            f"Eligibility Criteria:\n{trial.get('eligibility_criteria', 'N/A')}\n"
            f"Gender: {trial.get('eligibility_gender', 'N/A')}\n"
            f"Age: {trial.get('eligibility_age', 'N/A')}\n"
            f"Healthy Volunteers: {trial.get('eligibility_healthy_volunteers', 'N/A')}\n"
            f"Healthy Volunteers Description: {trial.get('eligibility_healthy_volunteers_description', 'N/A')}"
            for trial in trials
        ])
        
        return f"""System: {PromptTemplates.get_base_system_prompt()}
        When asked about eligibility, structure response as:
        - Inclusion criteria
        - Exclusion criteria
        - Age requirements
        - Gender requirements
        - Healthy volunteer status
        - Other specific requirements

        Context:
        {context}

        User Query: [User's question about eligibility]

        Assistant: [Structured response]"""

    @staticmethod
    def get_intervention_query_prompt(trials: List[Dict[str, Any]]) -> str:
        context = "\n\n".join([
            f"Trial: {trial.get('title', 'N/A')} (NCT ID: {trial.get('nct_id', 'N/A')})\n"
            f"Interventions:\n"
            f"Types: {', '.join(trial.get('intervention_types', ['N/A']))}\n"
            f"Names: {', '.join(trial.get('intervention_names', ['N/A']))}\n"
            f"Descriptions: {', '.join(trial.get('intervention_descriptions', ['N/A']))}"
            for trial in trials
        ])
        
        return f"""System: {PromptTemplates.get_base_system_prompt()}
        When asked about interventions, include:
        - Type of intervention
        - Name of intervention
        - Description of intervention
        - Dosage/administration if available
        - Duration if available
        - Comparator if applicable

        Context:
        {context}

        User Query: [User's question about interventions]

        Assistant: [Structured response]"""

    @staticmethod
    def get_outcome_query_prompt(trials: List[Dict[str, Any]]) -> str:
        context = "\n\n".join([
            f"Trial: {trial.get('title', 'N/A')} (NCT ID: {trial.get('nct_id', 'N/A')})\n"
            f"Primary Outcomes:\n{trial.get('primary_outcomes', 'N/A')}\n"
            f"Secondary Outcomes:\n{trial.get('secondary_outcomes', 'N/A')}"
            for trial in trials
        ])
        
        return f"""System: {PromptTemplates.get_base_system_prompt()}
        When asked about outcomes, include:
        - Primary outcomes
        - Secondary outcomes
        - Time frames
        - Measurement methods
        - Any specific outcome criteria

        Context:
        {context}

        User Query: [User's question about outcomes]

        Assistant: [Structured response]"""

    @staticmethod
    def get_general_query_prompt(trials: List[Dict[str, Any]]) -> str:
        context = "\n\n".join([
            f"Trial: {trial.get('title', 'N/A')} (NCT ID: {trial.get('nct_id', 'N/A')})\n"
            f"Official Title: {trial.get('official_title', 'N/A')}\n"
            f"Organization: {trial.get('organization_full_name', 'N/A')}\n"
            f"Status: {trial.get('status', 'N/A')}\n"
            f"Phase: {', '.join(trial.get('study_phase', ['N/A']))}\n"
            f"Study Type: {trial.get('study_type', 'N/A')}\n"
            f"Brief Summary: {trial.get('brief_summary', 'N/A')}\n"
            f"Conditions: {', '.join(trial.get('conditions', ['N/A']))}\n"
            f"Sponsor: {trial.get('sponsor', 'N/A')}\n"
            f"Collaborators: {', '.join(trial.get('collaborators', ['N/A']))}"
            for trial in trials
        ])
        
        return f"""System: {PromptTemplates.get_base_system_prompt()}

        Context:
        {context}

        User Query: [User's question]

        Assistant: [Structured response]""" 