from typing import List, Dict, Any, Optional
from .document_processor import ClinicalTrialProcessor
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import Document
from ..prompts import templates
from .query_analyzer import create_query_analyzer
import os
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from operator import itemgetter

class RAGManager:
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        """
        Initialize the RAG manager with vector store and LLM components.
        
        Args:
            persist_directory (str): Directory to persist the Chroma database
        """
        self.embeddings = OpenAIEmbeddings()
        self.persist_directory = persist_directory
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
        self.query_analyzer = create_query_analyzer()
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Number of documents to retrieve
        )
        self.multi_query_prompt = ChatPromptTemplate.from_template(
            """You are an AI language model assistant. Your task is to generate five \
            different versions of the given user question to retrieve relevant documents from a vector \
            database. By generating multiple perspectives on the user question, your goal is to help\
            the user overcome some of the limitations of the distance-based similarity search. \
            Provide these alternative questions separated by newlines. Original question: {question}"""
        )
        self.generate_queries = (
            self.multi_query_prompt
            | ChatOpenAI(temperature=0)
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )
    
    def add_trials(self, trials_data: List[Dict[str, Any]]) -> None:
        """Add new clinical trials to the vector store."""
        processor = ClinicalTrialProcessor()
        documents = processor.process_trials_batch(trials_data)
        
        batch_size = 500
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self.vector_store.add_documents(batch)
            print(f"Added batch {i//batch_size + 1} of {(len(documents) + batch_size - 1)//batch_size}")
        
        # No need to call persist() as Chroma automatically persists changes
    
    @staticmethod
    def get_unique_union(documents: list[list]):
        """Unique union of retrieved docs."""
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        unique_docs = list(set(flattened_docs))
        return [loads(doc) for doc in unique_docs]

    def get_response(self, query: str) -> str:
        """Generate a response for a user query using the final RAG chain."""
        try:
            # Step 1: Multi-query generation
            retrieval_chain = self.generate_queries | self.retriever.map() | RAGManager.get_unique_union
            # (Future steps: e.g., filtering, ranking, answer generation, etc.)

            template = """Answer the following question based on this context:

            {context}

            Question: {question}
            """
            prompt = ChatPromptTemplate.from_template(template)
            llm = ChatOpenAI(temperature=0)

            # Step 2: Answer generation (using retrieved context)
            final_rag_chain = (
                {
                    "context": retrieval_chain,
                    "question": itemgetter("question")
                }
                | prompt
                | llm
                | StrOutputParser()
            )
            response = final_rag_chain.invoke({"question": query})
            return response
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while processing your query. Please try again."
    
    def clear_database(self) -> None:
        """Clear all data from the vector store."""
        self.vector_store.delete_collection()
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        self.retriever = self.vector_store.as_retriever()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the database."""
        return {
            "total_documents": len(self.vector_store.get()["ids"]),
            "collection_name": self.vector_store._collection.name,
            "embedding_function": str(self.embeddings)
        }
    
    def generate_multi_queries(self, question: str, n: int = 5) -> list:
        """Generate multiple perspectives of the input question."""
        queries = self.generate_queries.invoke({"question": question})
        # Remove empty strings and limit to n
        return [q.strip() for q in queries if q.strip()][:n] 