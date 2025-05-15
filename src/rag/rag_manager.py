from typing import List, Dict, Any
from .document_processor import ClinicalTrialProcessor
from .text_processor import TextProcessor
from .vector_store import VectorStoreManager
from ..prompts.templates import PromptTemplates
from dotenv import load_dotenv
import os
import re
from datetime import datetime
import requests
import json

load_dotenv()

class OllamaProvider:
    def __init__(self, model: str = "mistral", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    def _check_ollama_health(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False
    
    def generate_response(self, prompt: str, query: str, temperature: float = 0.7) -> str:
        """Generate a response from Ollama with proper error handling."""
        if not self._check_ollama_health():
            raise ConnectionError("Ollama is not running. Please start Ollama with 'ollama serve'")
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": f"{prompt}\n\nUser: {query}\nAssistant:",
                    "temperature": temperature,
                    "stream": False  # Ensure we get a complete response
                },
                timeout=30  # Add timeout
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
            
            try:
                result = response.json()
                if "response" not in result:
                    raise ValueError("Invalid response format from Ollama")
                return result["response"]
            except json.JSONDecodeError as e:
                # Try to extract response from raw text if JSON parsing fails
                text = response.text.strip()
                if text:
                    return text
                raise Exception(f"Failed to parse Ollama response: {str(e)}")
                
        except requests.exceptions.Timeout:
            raise Exception("Request to Ollama timed out. Please try again.")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error communicating with Ollama: {str(e)}")
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text.split()) * 1.3  # Rough estimate

class RAGManager:
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """Initialize the RAG system components."""
        self.text_processor = TextProcessor()
        self.document_processor = ClinicalTrialProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.vector_store = VectorStoreManager(persist_directory=persist_directory)
        self.llm = OllamaProvider(model="mistral")
        
        self.usage_stats = {
            "total_queries": 0,
            "total_tokens": 0,
            "queries_by_model": {"mistral": 0},
            "last_reset": datetime.now().isoformat()
        }
    
    def _determine_query_type(self, query: str) -> str:
        """Determine the type of query based on keywords."""
        query_lower = query.lower()
        
        # Define keyword patterns for each query type
        patterns = {
            'status': r'\b(status|phase|stage|current state|recruiting|completed|terminated|suspended)\b',
            'eligibility': r'\b(eligible|eligibility|criteria|requirements|inclusion|exclusion|age|gender)\b',
            'intervention': r'\b(intervention|treatment|drug|therapy|medication|dose|dosage|administration)\b',
            'outcome': r'\b(outcome|result|endpoint|measure|assessment|evaluation|primary|secondary)\b'
        }
        
        # Check each pattern
        for query_type, pattern in patterns.items():
            if re.search(pattern, query_lower):
                return query_type
        
        return 'general'
    
    def _get_relevant_trials(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Retrieve relevant trials for a query."""
        # Check if query contains an NCT ID
        nct_id_match = re.search(r'NCT\d{8}', query)
        
        if nct_id_match:
            # If NCT ID is found, try to get that specific trial
            nct_id = nct_id_match.group()
            doc = self.vector_store.get_document_by_id(nct_id)
            if doc:
                return [{
                    'nct_id': nct_id,
                    'title': doc.metadata.get('title'),
                    'status': doc.metadata.get('status'),
                    'phase': doc.metadata.get('phase'),
                    'content': doc.page_content
                }]
        
        # If no NCT ID or trial not found, perform semantic search
        documents = self.vector_store.similarity_search(query, k=k)
        
        # Convert documents back to trial format
        trials = []
        seen_nct_ids = set()
        
        for doc in documents:
            nct_id = doc.metadata.get('nct_id')
            if nct_id not in seen_nct_ids:
                seen_nct_ids.add(nct_id)
                trials.append({
                    'nct_id': nct_id,
                    'title': doc.metadata.get('title'),
                    'status': doc.metadata.get('status'),
                    'phase': doc.metadata.get('phase'),
                    'content': doc.page_content
                })
        
        return trials
    
    def _get_prompt_for_query(self, query: str, trials: List[Dict[str, Any]]) -> str:
        """Get the appropriate prompt template based on query type."""
        query_type = self._determine_query_type(query)
        
        prompt_methods = {
            'status': PromptTemplates.get_status_query_prompt,
            'eligibility': PromptTemplates.get_eligibility_query_prompt,
            'intervention': PromptTemplates.get_intervention_query_prompt,
            'outcome': PromptTemplates.get_outcome_query_prompt,
            'general': PromptTemplates.get_general_query_prompt
        }
        
        # Get the appropriate prompt template
        prompt_template = prompt_methods[query_type](trials)
        
        # Replace placeholder with actual query
        return prompt_template.replace("[User's question]", query)
    
    def get_response(
        self,
        query: str,
        k: int = 4,
        temperature: float = 0.7
    ) -> str:
        """Generate a response for a user query."""
        try:
            # Get relevant trials
            trials = self._get_relevant_trials(query, k=k)
            
            if not trials:
                return "I couldn't find any relevant clinical trials for your query."
            
            # Get appropriate prompt template
            prompt = self._get_prompt_for_query(query, trials)
            
            # Count input tokens
            input_tokens = self.llm.count_tokens(prompt) + self.llm.count_tokens(query)
            
            # Generate response
            response = self.llm.generate_response(prompt, query, temperature)
            
            # Update usage stats
            output_tokens = self.llm.count_tokens(response)
            self.usage_stats["total_queries"] += 1
            self.usage_stats["total_tokens"] += input_tokens + output_tokens
            self.usage_stats["queries_by_model"]["mistral"] += 1
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while processing your query. Please try again."
    
    def add_trials(self, trials_data: List[Dict[str, Any]]) -> None:
        """Add new clinical trials to the system."""
        # First, clean and preprocess the text
        processed_trials = self.text_processor.process_trials_batch(trials_data)
        
        # Then, process into documents
        documents = self.document_processor.process_trials_batch(processed_trials)
        
        # Finally, add to vector store
        self.vector_store.add_documents(documents)
    
    def clear_database(self) -> None:
        """Clear all data from the vector store."""
        self.vector_store.clear_collection()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the database."""
        return self.vector_store.get_collection_stats()
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return self.usage_stats
    
    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self.usage_stats = {
            "total_queries": 0,
            "total_tokens": 0,
            "queries_by_model": {"mistral": 0},
            "last_reset": datetime.now().isoformat()
        } 