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
        "IdentificationModule",
        "StatusModule",
        "SponsorCollaboratorsModule",
        "ConditionsModule",
        "ArmsInterventionsModule",
        "EligibilityModule",
        "DescriptionModule",
        "DesignModule",
        "OutcomesModule"
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

def preprocess_trial_data(trials_data):
    """Preprocess the clinical trials data for LLM processing"""
    # Extract studies from the response
    studies = trials_data.get('studies', [])
    
    # Create a list to store processed trials
    processed_trials = []
    
    for study in studies:
        # Extract the protocol section which contains most of the text data
        protocol = study.get('protocolSection', {})
        results = study.get('resultsSection', {})
        
        # Create a structured dictionary for each trial
        trial = {
            # Identification Module
            'nct_id': protocol.get('identificationModule', {}).get('nctId'),
            'organization_full_name': protocol.get('identificationModule', {}).get('organization', {}).get('fullName'),
            'title': protocol.get('identificationModule', {}).get('briefTitle'),
            'official_title': protocol.get('identificationModule', {}).get('officialTitle'),

            # Status Module
            'why_stopped': protocol.get('statusModule', {}).get('whyStopped'),
            'status': protocol.get('statusModule', {}).get('overallStatus'),
            'start_date': protocol.get('statusModule', {}).get('startDateStruct', {}).get('date'),
            'completion_date': protocol.get('statusModule', {}).get('completionDateStruct', {}).get('date'),
            'last_update': protocol.get('statusModule', {}).get('lastUpdatePostDateStruct', {}).get('date'),

            # Sponsor Collaborators Module
            'sponsor': protocol.get('sponsorCollaboratorsModule', {}).get('leadSponsor', {}).get('name'),
            'collaborators': [collaborator.get('name') for collaborator in protocol.get('sponsorCollaboratorsModule', {}).get('collaborators', [])],

            # Oversight Module
            'has_dmc': protocol.get('oversightModule', {}).get('oversightHasDmc'),
            'is_fda_regulated_drug': protocol.get('oversightModule', {}).get('isFdaRegulatedDrug'),
            'is_fda_regulated_device': protocol.get('oversightModule', {}).get('isFdaRegulatedDevice'), 
            'is_unapproved_device': protocol.get('oversightModule', {}).get('isUnapprovedDevice'),
            'is_ppsd': protocol.get('oversightModule', {}).get('isPpsd'),
            'is_us_export': protocol.get('oversightModule', {}).get('isUsExport'),

            # Description Module
            'brief_summary': protocol.get('descriptionModule', {}).get('briefSummary'),
            'detailed_description': protocol.get('descriptionModule', {}).get('detailedDescription'),

            # Conditions Module
            'conditions': protocol.get('conditionsModule', {}).get('conditions', []),
            
            # Design Module
            'study_type': protocol.get('designModule', {}).get('studyType'),
            'study_phase': protocol.get('designModule', {}).get('phases', []),
            'design_allocation': protocol.get('designModule', {}).get('designInfo', {}).get('allocation'),
            'intervention_study_design': protocol.get('designModule', {}).get('designInfo', {}).get('interventionModel'),
            'design_primary_purpose': protocol.get('designModule', {}).get('designInfo', {}).get('primaryPurpose'),
            'design_time_perspective': protocol.get('designModule', {}).get('designInfo', {}).get('timePerspective'),
            'enrollment': protocol.get('statusModule', {}).get('enrollmentCount'),
        
            # Arms Interventions Module
            'arm_group_label': [intervention.get('label') for intervention in protocol.get('armsInterventionsModule', {}).get('interventions', [])],
            'intervention_types': [intervention.get('type') for intervention in protocol.get('armsInterventionsModule', {}).get('interventions', [])],
            'intervention_names': [intervention.get('name') for intervention in protocol.get('armsInterventionsModule', {}).get('interventions', [])],
            'intervention_descriptions': [intervention.get('description') for intervention in protocol.get('armsInterventionsModule', {}).get('interventions', [])],
            
            # Outcomes Module
            'primary_outcomes': protocol.get('outcomesModule', {}).get('primaryOutcomes', []),
            'secondary_outcomes': protocol.get('outcomesModule', {}).get('secondaryOutcomes', []),
            
            # Eligibility Module
            'eligibility_criteria': protocol.get('eligibilityModule', {}).get('eligibilityCriteria'),
            'eligibility_gender': protocol.get('eligibilityModule', {}).get('gender'),
            'eligibility_age': protocol.get('eligibilityModule', {}).get('age'),
            'eligibility_healthy_volunteers': protocol.get('eligibilityModule', {}).get('healthyVolunteers'),
            'eligibility_healthy_volunteers_description': protocol.get('eligibilityModule', {}).get('healthyVolunteersDescription'),
            
            # Contacts Locations Module
            'facility': [location.get('facility') for location in protocol.get('contactsLocationsModule', {}).get('locations', [])]

            # Results Section
        }
        
        processed_trials.append(trial)
    
    return processed_trials