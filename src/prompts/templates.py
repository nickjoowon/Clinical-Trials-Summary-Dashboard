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
    def get_general_query_prompt(query: str, context: str) -> str:
        return f"""You are a helpful assistant specialized in analyzing clinical trials. Use the following context to answer the user's question. 
        If the information is not available in the context, say so. Be concise but informative.

        Context:
        {context}

        User's question: {query}

        Please provide a clear and accurate response based on the clinical trial information provided."""

    @staticmethod
    def get_summary_prompt(query: str, context: str) -> str:
        return f"""You are a helpful assistant specialized in summarizing clinical trials. Create a concise summary of the following clinical trial information, 
        focusing on the key aspects relevant to the user's request. Be clear and organized in your summary.

        Context:
        {context}

        User's request: {query}

        Please provide a well-structured summary that captures the essential information."""

    @staticmethod
    def get_detailed_summary_prompt(query: str, context: str) -> str:
        return f"""You are a helpful assistant specialized in providing detailed summaries of clinical trials. Create a comprehensive summary of the following 
        clinical trial information, covering all important aspects while maintaining clarity and organization. Include specific details, numbers, and outcomes 
        where available.

        Context:
        {context}

        User's request: {query}

        Please provide a thorough and well-structured summary that includes all relevant details."""

    @staticmethod
    def get_eligibility_prompt(query: str, context: str) -> str:
        return f"""You are a helpful assistant specialized in analyzing clinical trial eligibility criteria. Review the following clinical trial information 
        and provide a clear explanation of the eligibility requirements, focusing on inclusion and exclusion criteria.

        Context:
        {context}

        User's question: {query}

        Please provide a detailed breakdown of the eligibility criteria, organized by inclusion and exclusion factors."""

    @staticmethod
    def get_outcome_prompt(query: str, context: str) -> str:
        return f"""You are a helpful assistant specialized in analyzing clinical trial outcomes. Review the following clinical trial information and provide 
        a clear explanation of the study outcomes, including primary and secondary endpoints, results, and their significance.

        Context:
        {context}

        User's question: {query}

        Please provide a detailed analysis of the trial outcomes, including any statistical significance and clinical relevance."""

    @staticmethod
    def get_trial_discovery_prompt(query: str, context: str) -> str:
        return f"""You are a helpful assistant specialized in finding relevant clinical trials. Based on the user's request, 
        identify and present the most relevant clinical trials from the provided context Make sure to not duplicate trials and if there are no relevant trials, say so.
        Format your response as follows:

        Start with a brief overview of what you found
        
        List up to 5 most relevant trials, each with:
           - Trial title
           - NCT ID
           - Brief description (1-2 sentences)
           - Current status
           - Key eligibility criteria (if available)
           - all the above in a bullet point list
        
        

        Context:
        {context}

        User's request: {query}

        Please provide a well-organized list of relevant clinical trials.""" 