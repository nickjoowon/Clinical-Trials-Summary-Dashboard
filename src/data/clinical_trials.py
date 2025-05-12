import requests
import pandas as pd
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

def fetch_clinical_trials(max_results: int = 100) -> Dict[str, Any]:
    """Fetch clinical trials data from ClinicalTrials.gov API v2"""
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    
    # Define the fields we want to retrieve - focusing on text-rich fields for LLM processing
    fields = [
        "NCTId",
        "BriefTitle",
        "OfficialTitle",
        "BriefSummary",
        "DetailedDescription",
        "Condition",
        "InterventionDescription",
        "EligibilityCriteria",
        "StudyType",
        "OverallStatus",
        "EnrollmentCount",
        "StartDate",
        "CompletionDate",
        "LastUpdatePostDate",
        "LeadSponsorName",
        "LocationFacility"
    ]
    
    params = {
        "query.term": "AREA[LastUpdatePostDate]RANGE[2023-01-01,MAX]",  # Recent trials
        "fields": ",".join(fields),
        "pageSize": min(max_results, 1000),  # API limits pageSize to 1000
        "format": "json",
        "markupFormat": "markdown",
        "sort": ["LastUpdatePostDate:desc"]  # Get most recent trials first
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}")
        
    try:
        return response.json()
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {str(e)}")
        print(f"Response content: {response.text}")
        raise

def preprocess_trial_data(trials_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Preprocess the clinical trials data for LLM processing"""
    # Extract studies from the response
    studies = trials_data.get('studies', [])
    
    # Create a list to store processed trials
    processed_trials = []
    
    for study in studies:
        # Extract the protocol section which contains most of the text data
        protocol = study.get('protocolSection', {})
        
        # Create a structured dictionary for each trial
        trial = {
            'nct_id': protocol.get('identificationModule', {}).get('nctId'),
            'title': protocol.get('identificationModule', {}).get('briefTitle'),
            'official_title': protocol.get('identificationModule', {}).get('officialTitle'),
            'brief_summary': protocol.get('descriptionModule', {}).get('briefSummary'),
            'detailed_description': protocol.get('descriptionModule', {}).get('detailedDescription'),
            'conditions': protocol.get('conditionsModule', {}).get('conditions', []),
            'interventions': [intervention.get('description') for intervention in protocol.get('armsInterventionsModule', {}).get('interventions', [])],
            'eligibility_criteria': protocol.get('eligibilityModule', {}).get('eligibilityCriteria'),
            'study_type': protocol.get('designModule', {}).get('studyType'),
            'status': protocol.get('statusModule', {}).get('overallStatus'),
            'enrollment': protocol.get('statusModule', {}).get('enrollmentCount'),
            'start_date': protocol.get('statusModule', {}).get('startDateStruct', {}).get('date'),
            'completion_date': protocol.get('statusModule', {}).get('completionDateStruct', {}).get('date'),
            'last_update': protocol.get('statusModule', {}).get('lastUpdatePostDateStruct', {}).get('date'),
            'sponsor': protocol.get('sponsorCollaboratorsModule', {}).get('leadSponsor', {}).get('name'),
            'facility': [location.get('facility') for location in protocol.get('contactsLocationsModule', {}).get('locations', [])],
            'primary_outcome': protocol.get('outcomesModule', {}).get('primaryOutcomes', [])
        }
        
        processed_trials.append(trial)
    
    return processed_trials 