import requests
import pandas as pd
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

def fetch_clinical_trials(start_date: str = "2024-01-01", max_results: int = None) -> Dict[str, Any]:
    """
    Fetch all clinical trials from ClinicalTrials.gov API v2 from start_date onwards, handling pagination with nextPageToken.
    Args:
        start_date (str): Earliest LastUpdatePostDate to fetch (format: YYYY-MM-DD).
        max_results (int or None): Maximum number of results to fetch. If None, fetch all available.
    Returns:
        Dict[str, Any]: All studies fetched.
    """
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    fields = [
        "IdentificationModule", "StatusModule", "SponsorCollaboratorsModule",
        "ConditionsModule", "ArmsInterventionsModule", "EligibilityModule",
        "DescriptionModule", "DesignModule", "OutcomesModule"
    ]
    all_studies = []
    page_size = 1000  # API max
    next_page_token = None

    while True:
        params = {
            "query.term": f"AREA[LastUpdatePostDate]RANGE[{start_date},MAX]",
            "fields": ",".join(fields),
            "pageSize": page_size,
            "format": "json",
            "markupFormat": "markdown",
            "sort": ["LastUpdatePostDate:desc"]
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}")
        data = response.json()
        studies = data.get('studies', [])
        all_studies.extend(studies)
        if max_results and len(all_studies) >= max_results:
            all_studies = all_studies[:max_results]
            break
        next_page_token = data.get("nextPageToken")
        if not next_page_token or not studies:
            break

    return {"studies": all_studies}

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
            'facility': [location.get('facility') for location in protocol.get('contactsLocationsModule', {}).get('locations', [])],

            # Results Section
            # Participant Flow Module
            'period_title': results.get('participantFlowModule', {}).get('periods', {}).get('title'),
            'milestone_title': results.get('participantFlowModule', {}).get('periods', {}).get('milestones', {}).get('type'),
            'milestone_comment': results.get('participantFlowModule', {}).get('periods', {}).get('milestones', {}).get('comment'),
            'num_of_periods': results.get('participantFlowModule', {}).get('numFlowPeriods'),

            # Baseline Characteristics Module
            'baseline_analysis_population_description': results.get('baselineCharacteristicsModule', {}).get('populationDescription'),
            'arm_group_title': results.get('baselineCharacteristicsModule', {}).get('groups', {}).get('title'),
            'arm_group_description': results.get('baselineCharacteristicsModule', {}).get('groups', {}).get('description'),
            'baseline_measure_title': results.get('baselineCharacteristicsModule', {}).get('measures', {}).get('title'),
            'baseline_measure_title_for_study_specified_measure': results.get('baselineCharacteristicsModule', {}).get('measures', {}).get('description'),
            'baseline_measure_type': results.get('baselineCharacteristicsModule', {}).get('measures', {}).get('paramType'),
            'baseline_measure_dispersion_precision': results.get('baselineCharacteristicsModule', {}).get('measures', {}).get('dispersionType'),
            'baseline_unit_of_measure': results.get('baselineCharacteristicsModule', {}).get('measures', {}).get('unitOfMeasure'),

            # Outcome Measures Module
            'outcome_measure_type': results.get('outcomeMeasuresModule', {}).get('outcomeMeasures', {}).get('type'),
            'outcome_measure_title': results.get('outcomeMeasuresModule', {}).get('outcomeMeasures', {}).get('title'),
            'outcome_measure_time_frame': results.get('outcomeMeasuresModule', {}).get('outcomeMeasures', {}).get('timeFrame'),
            'outcome_group_title': results.get('outcomeMeasuresModule', {}).get('outcomeMeasures', {}).get('groups', {}).get('title'),
            'outcome_denom_count_value': results.get('outcomeMeasuresModule', {}).get('outcomeMeasures', {}).get('denoms', {}).get('counts', {}).get('value'),
            'outcome_measure_data_type': results.get('outcomeMeasuresModule', {}).get('outcomeMeasures', {}).get('paramType'),
            'outcome_measure_dispersion_precision': results.get('outcomeMeasuresModule', {}).get('outcomeMeasures', {}).get('dispersionType'),
            'outcome_measurement_value': results.get('outcomeMeasuresModule', {}).get('outcomeMeasures', {}).get('classes', {}).get('categories', {}).get('measurements', {}).get('value'),
            'outcome_measure_unit_of_measure': results.get('outcomeMeasuresModule', {}).get('outcomeMeasures', {}).get('unitOfMeasure'),

            # Adverse Events Module
            'adverse_events_arm_group_title': results.get('adverseEventsModule', {}).get('eventGroups', {}).get('title'),
            'num_affected_by_serious_adverse_event': results.get('adverseEventsModule', {}).get('eventGroups', {}).get('seriousNumAffected'),
            'num_affected_by_serious_adverse_event_description': results.get('adverseEventsModule', {}).get('eventGroups', {}).get('seriousNumAffectedDescription'),
            'num_at_risk_for_serious_adverse_event': results.get('adverseEventsModule', {}).get('eventGroups', {}).get('seriousNumAtRisk'),
            'num_affected_by_other_adverse_event': results.get('adverseEventsModule', {}).get('eventGroups', {}).get('otherNumAffected'),
            'num_at_risk_for_other_adverse_event': results.get('adverseEventsModule', {}).get('eventGroups', {}).get('otherNumAtRisk'),
            'adverse_event_term': results.get('adverseEventsModule', {}).get('seriousEvents', {}).get('term'),
            'organ_system': results.get('adverseEventsModule', {}).get('seriousEvents', {}).get('organSystem'),

            # More Info Module
            'point_of_contact_title': results.get('moreInfoModule', {}).get('pointOfContact', {}).get('title'),
            'point_of_contact_organization': results.get('moreInfoModule', {}).get('pointOfContact', {}).get('organization')
        }
        
        processed_trials.append(trial)
    
    return processed_trials