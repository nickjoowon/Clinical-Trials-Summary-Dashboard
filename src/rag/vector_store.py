from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from langchain.schema import Document
import os
from datetime import datetime

class VectorStoreManager:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize the vector store with ChromaDB."""
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get the collection
        self.collection = self.client.get_or_create_collection(
            name="clinical_trials",
            metadata={"description": "Clinical trials documents and embeddings"}
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        try:
            # Prepare data for ChromaDB
            ids = [f"doc_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}" 
                  for i in range(len(documents))]
            cleaned_metadatas = []
            texts = []
            
            for doc in documents:
                # Clean metadata values
                cleaned_metadata = {}
                for key, value in doc.metadata.items():
                    if value is None:
                        cleaned_metadata[key] = "N/A"
                    elif isinstance(value, (str, int, float, bool)):
                        cleaned_metadata[key] = value
                    else:
                        cleaned_metadata[key] = str(value)
                
                cleaned_metadatas.append(cleaned_metadata)
                texts.append(doc.page_content)
            
            # Process in batches of 5000 (safe number below ChromaDB's limit)
            batch_size = 5000
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_texts = texts[i:i + batch_size]
                batch_metadatas = cleaned_metadatas[i:i + batch_size]
                
                # Add batch to collection
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_texts,
                    metadatas=batch_metadatas
                )
                print(f"Added batch {i//batch_size + 1} of {(len(ids) + batch_size - 1)//batch_size}")
            
        except Exception as e:
            print(f"Error adding documents to vector store: {str(e)}")
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search for similar documents."""
        try:
            # Convert filter criteria to ChromaDB's where clause format
            where = None
            if filter_criteria:
                where = {}
                for key, value in filter_criteria.items():
                    where[key] = {"$eq": value}
            
            # Perform the search
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=where
            )
            
            # Convert results to Document objects
            documents = []
            for i in range(len(results['documents'][0])):
                doc = Document(
                    page_content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i]
                )
                documents.append(doc)
            
            return documents
        except Exception as e:
            print(f"Error performing similarity search: {str(e)}")
            return []
    
    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """Retrieve a specific document by its ID."""
        try:
            result = self.collection.get(ids=[doc_id])
            if result and result['documents']:
                return Document(
                    page_content=result['documents'][0],
                    metadata=result['metadatas'][0]
                )
            return None
        except Exception as e:
            print(f"Error retrieving document: {str(e)}")
            return None

    
    def update_document(self, doc_id: str, document: Document) -> bool:
        """Update an existing document."""
        try:
            self.collection.update(
                ids=[doc_id],
                documents=[document.page_content],
                metadatas=[document.metadata]
            )
            return True
        except Exception as e:
            print(f"Error updating document: {str(e)}")
            return False
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the store."""
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception as e:
            print(f"Error deleting document: {str(e)}")
            return False
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        try:
            self.collection.delete(where={})
        except Exception as e:
            print(f"Error clearing collection: {str(e)}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection.name,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            print(f"Error getting collection stats: {str(e)}")
            return {} 