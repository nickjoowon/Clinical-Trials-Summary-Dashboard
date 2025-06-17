from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class ClinicalTrialSearch(BaseModel):
    """Search over a database of clinical trials."""
    
    content_search: str = Field(
        ...,
        description="Similarity search query applied to trial descriptions and titles."
    )
    
    title_search: str = Field(
        ...,
        description="Alternate version of the content search query to apply to trial titles."
    )
    
    conditions: Optional[str] = Field(
        None,
        description="Comma-separated medical conditions to filter by. Only use if explicitly specified."
    )
    
    phase: Optional[str] = Field(
        None,
        description="Comma-separated trial phases to filter by (e.g., 'PHASE1, PHASE2'). Only use if explicitly specified."
    )
    
    status: Optional[str] = Field(
        None,
        description="Trial status to filter by (e.g., 'RECRUITING, COMPLETED'). Only use if explicitly specified."
    )
    
    interventions: Optional[str] = Field(
        None,
        description="Comma-separated interventions to filter by. Only use if explicitly specified."
    )
    
    earliest_start_date: Optional[datetime] = Field(
        None,
        description="Earliest trial start date filter, inclusive. Only use if explicitly specified."
    )
    
    latest_start_date: Optional[datetime] = Field(
        None,
        description="Latest trial start date filter, exclusive. Only use if explicitly specified."
    )
    
    study_type: Optional[str] = Field(
        None,
        description="Comma-separated study types to filter by (e.g., 'INTERVENTIONAL, OBSERVATIONAL'). Only use if explicitly specified."
    )
    
    nct_id: Optional[str] = Field(
        None,
        description="NCT ID to filter by. Only use if explicitly specified."
    )

    def pretty_print(self) -> None:
        for field in self.__fields__:
            if getattr(self, field) is not None and getattr(self, field) != getattr(
                self.__fields__[field], "default", None
            ):
                print(f"{field}: {getattr(self, field)}")

def create_query_analyzer():
    """Create and return a query analyzer for clinical trials."""
    system = """You are an expert at converting user questions about clinical trials into structured database queries. \
                You have access to a database of clinical trials with the following metadata fields:
                - nct_id: unique identifier for each clinical trial
                - conditions: comma-separated medical conditions being studied
                - phase: comma-separated trial phases (PHASE1, PHASE2, etc.)
                - status: trial status (RECRUITING, COMPLETED, etc.)
                - interventions: comma-separated treatments or interventions being studied
                - start_date: when the trial started
                - study_type: comma-separated type of study (INTERVENTIONAL, OBSERVATIONAL, etc.)

                Given a question, return a database query optimized to retrieve the most relevant results.
                If there are medical terms or acronyms you are not familiar with, do not try to rephrase them."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{question}"),
    ])

    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    structured_llm = llm.with_structured_output(ClinicalTrialSearch)
    return prompt | structured_llm 