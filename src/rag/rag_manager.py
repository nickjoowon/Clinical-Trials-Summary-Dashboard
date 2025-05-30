from typing import List, Dict, Any, Optional
from .document_processor import ClinicalTrialProcessor
from .text_processor import TextProcessor
from .vector_store import VectorStoreManager
from ..prompts.templates import PromptTemplates
from dotenv import load_dotenv
from langchain.schema import Document
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
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
            
            try:
                result = response.json()
                if "response" not in result:
                    raise ValueError("Invalid response format from Ollama")
                return result["response"]
            except json.JSONDecodeError as e:
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
        chunk_overlap: int = 200,
        model_name: str = "mistral"
    ):
        """Initialize the RAG system components."""
        self.text_processor = TextProcessor()
        self.document_processor = ClinicalTrialProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.vector_store = VectorStoreManager(persist_directory=persist_directory)
        self.llm = OllamaProvider(model=model_name)
        self.model_name = model_name
        
        self.usage_stats = {
            "total_queries": 0,
            "total_tokens": 0,
            "queries_by_model": {model_name: 0},
            "last_reset": datetime.now().isoformat()
        }
    
    def _get_documents_for_nct_id(self, nct_id: str, query: str) -> List[Document]:
        """Get all documents for a specific NCT ID."""
        # First try exact metadata match
        relevant_docs = self.vector_store.similarity_search(
            query,
            k=50,  # Get more documents to ensure we find all chunks
            filter_criteria={"nct_id": nct_id}
        )
        
        # If no results, try without filter to see if we can find it
        if not relevant_docs:
            print(f"Debug: No results found with filter for {nct_id}")
            # Try without filter to see what we get
            all_docs = self.vector_store.similarity_search(query, k=50)
            # Check if any of these docs have our NCT ID
            relevant_docs = [
                doc for doc in all_docs 
                if doc.metadata.get('nct_id') == nct_id
            ]
            if relevant_docs:
                print(f"Debug: Found {len(relevant_docs)} documents with NCT ID {nct_id} without filter")
            else:
                print(f"Debug: No documents found with NCT ID {nct_id} in any results")
                return []
        
        # Sort documents by their chunk index if available
        try:
            relevant_docs.sort(key=lambda doc: int(doc.metadata.get('chunk_index', 0)))
        except (ValueError, TypeError):
            # If chunk_index is not available or not a number, keep original order
            pass
        
        return relevant_docs

    def _get_documents_for_query(self, query: str) -> List[Document]:
        """Get all documents for a general query, ensuring complete trial information."""
        # First, get the most relevant documents to find the most relevant trials
        initial_docs = self.vector_store.similarity_search(query, k=5)
        
        if not initial_docs:
            return []
        
        # Extract unique NCT IDs from the initial results
        unique_nct_ids = set()
        for doc in initial_docs:
            nct_id = doc.metadata.get('nct_id')
            if nct_id:
                unique_nct_ids.add(nct_id)
        
        # For each unique NCT ID, get ALL documents for that trial
        all_relevant_docs = []
        for nct_id in unique_nct_ids:
            trial_docs = self._get_documents_for_nct_id(nct_id, query)
            all_relevant_docs.extend(trial_docs)
        
        # Sort documents by NCT ID and chunk index
        try:
            all_relevant_docs.sort(key=lambda doc: (
                doc.metadata.get('nct_id', ''),
                int(doc.metadata.get('chunk_index', 0))
            ))
        except (ValueError, TypeError):
            # If chunk_index is not available or not a number, keep original order
            pass
        
        return all_relevant_docs

    def _select_prompt_template(self, query: str, context: str) -> str:
        """Select the appropriate prompt template based on the query."""
        query_lower = query.lower()
        
        # Check for outcome-related queries
        outcome_keywords = [
            "outcome", "results", "outcome measure", "outcomes", "measures"
        ]
        
        if any(keyword in query_lower for keyword in outcome_keywords):
            return PromptTemplates.get_outcome_prompt(query, context)
        # Check for trial discovery queries
        elif any(keyword in query_lower for keyword in ["show me", "find", "list", "related to", "about", "for"]):
            return PromptTemplates.get_trial_discovery_prompt(query, context)
        # Check for summary requests
        elif any(keyword in query_lower for keyword in ["summarize", "summary", "summarise", "brief", "overview", "sum up"]):
            if any(keyword in query_lower for keyword in ["detailed", "comprehensive", "complete", "full"]):
                return PromptTemplates.get_detailed_summary_prompt(query, context)
            else:
                return PromptTemplates.get_summary_prompt(query, context)
        # Default to general query
        else:
            return PromptTemplates.get_general_query_prompt(query, context)

    def _verify_response(self, response: str, context: str) -> str:
        """Verify that the response only contains information from the context."""
        verification_prompt = f"""Verify if the following response contains ONLY information from the provided context. 
        If the response contains any made-up or hallucinated information, return "HALLUCINATION_DETECTED".
        If the response only contains information from the context, return "VERIFIED".

        Context:
        {context}

        Response to verify:
        {response}

        Verification result:"""

        verification = self.llm.generate_response(verification_prompt, "")
        
        if "HALLUCINATION_DETECTED" in verification:
            return "I apologize, but I need to correct my previous response. I was about to provide some information that wasn't fully supported by the available data. Let me provide a more accurate response based only on the verified information:\n\n" + self.llm.generate_response(prompt + "\n\nIMPORTANT: Only use information explicitly stated in the context. Do not make up or infer any details.", query)
        
        return response

    def get_response(self, query: str) -> str:
        """Generate a response for a user query."""
        try:
            # Check for NCT ID in query
            nct_pattern = r'NCT\d{8}'
            nct_matches = re.findall(nct_pattern, query.upper())
            
            if nct_matches:
                # Get specific trial by NCT ID
                nct_id = nct_matches[0]
                relevant_docs = self._get_documents_for_nct_id(nct_id, query)
                
                if not relevant_docs:
                    return f"I couldn't find any clinical trial with {nct_id}."
            else:
                # Get relevant documents for general query
                relevant_docs = self._get_documents_for_query(query)
                
                if not relevant_docs:
                    return "I couldn't find any relevant clinical trials for your query."
            
            # Combine relevant documents
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Select appropriate prompt template
            prompt = self._select_prompt_template(query, context)
            
            # Generate response
            response = self.llm.generate_response(prompt, query)
            
            # Verify response
            response = self._verify_response(response, context)
            
            # Update usage stats
            input_tokens = self.llm.count_tokens(prompt) + self.llm.count_tokens(query)
            output_tokens = self.llm.count_tokens(response)
            self.usage_stats["total_queries"] += 1
            self.usage_stats["total_tokens"] += input_tokens + output_tokens
            self.usage_stats["queries_by_model"][self.model_name] += 1
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while processing your query. Please try again."
    
    def add_trials(self, trials_data: List[Dict[str, Any]]) -> None:
        """Add new clinical trials to the system."""
        processed_trials = self.text_processor.process_trials_batch(trials_data)
        documents = self.document_processor.process_trials_batch(processed_trials)
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
            "queries_by_model": {self.model_name: 0},
            "last_reset": datetime.now().isoformat()
        } 