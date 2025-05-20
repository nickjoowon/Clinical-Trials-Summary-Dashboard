from typing import List, Dict, Any, Optional
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
    
    def get_response(self, query: str) -> str:
        """Generate a response for a user query."""
        try:
            # Get relevant documents
            relevant_docs = self.vector_store.similarity_search(query, k=5)
            
            if not relevant_docs:
                return "I couldn't find any relevant clinical trials for your query."
            
            # Combine relevant documents
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Determine query type and select appropriate prompt
            query_lower = query.lower()
            
            # Check for trial discovery queries
            discovery_keywords = ["show me", "find", "list", "related to", "about", "for"]
            if any(keyword in query_lower for keyword in discovery_keywords):
                prompt = PromptTemplates.get_trial_discovery_prompt(query, context)
            # Check for summary requests
            elif any(keyword in query_lower for keyword in ["summarize", "summary", "summarise", "brief", "overview", "sum up"]):
                if any(keyword in query_lower for keyword in ["detailed", "comprehensive", "complete", "full"]):
                    prompt = PromptTemplates.get_detailed_summary_prompt(query, context)
                else:
                    prompt = PromptTemplates.get_summary_prompt(query, context)
            # Default to general query
            else:
                prompt = PromptTemplates.get_general_query_prompt(query, context)
            
            # Generate response
            response = self.llm.generate_response(prompt, query)
            
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