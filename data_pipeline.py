#!/usr/bin/env python3
"""
Clinical Trials Data Pipeline
Handles bulk ingestion of clinical trials data from ClinicalTrials.gov
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import json

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Get the absolute path of the project root directory (same as app.py)
current_file = Path(__file__).resolve()
project_root = current_file.parent  # data_pipeline.py is already in root
sys.path.insert(0, str(project_root))

from src.rag.rag_manager import RAGManager
from src.data.clinical_trials import fetch_clinical_trials, preprocess_trial_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ClinicalTrialsDataPipeline:
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        """Initialize the data pipeline."""
        self.rag_manager = RAGManager(persist_directory=persist_directory)
        self.persist_directory = persist_directory
        
    def check_existing_data(self) -> Dict[str, Any]:
        """Check if data already exists in the vector store."""
        try:
            stats = self.rag_manager.get_database_stats()
            logger.info(f"Current database stats: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Error checking database stats: {e}")
            return {"total_documents": 0}
    
    def fetch_trials_in_batches(self, start_date: str, max_results: Optional[int] = None, batch_size: int = 1000) -> List[Dict[str, Any]]:
        """
        Fetch trials in batches to avoid memory issues.
        Returns a list of trial batches.
        """
        logger.info(f"Starting data fetch from {start_date}")
        
        all_trials = []
        offset = 0
        
        while True:
            try:
                # Fetch a batch of trials
                batch_params = {
                    "start_date": start_date,
                    "max_results": batch_size
                }
                
                if max_results and len(all_trials) + batch_size > max_results:
                    remaining = max_results - len(all_trials)
                    batch_params["max_results"] = remaining
                
                logger.info(f"Fetching batch {len(all_trials)//batch_size + 1} (offset: {offset})")
                trials_data = fetch_clinical_trials(**batch_params)
                studies = trials_data.get('studies', [])
                
                if not studies:
                    logger.info("No more studies to fetch")
                    break
                
                all_trials.extend(studies)
                logger.info(f"Fetched {len(studies)} trials. Total: {len(all_trials)}")
                
                # Check if we've reached the limit
                if max_results and len(all_trials) >= max_results:
                    all_trials = all_trials[:max_results]
                    logger.info(f"Reached max results limit: {max_results}")
                    break
                
                # If we got fewer than batch_size, we're done
                if len(studies) < batch_size:
                    logger.info("Received fewer studies than batch size, ending fetch")
                    break
                
                offset += batch_size
                
            except Exception as e:
                logger.error(f"Error fetching batch: {e}")
                break
        
        logger.info(f"Total trials fetched: {len(all_trials)}")
        return all_trials
    
    def process_and_ingest_trials(self, trials: List[Dict[str, Any]], batch_size: int = 500) -> int:
        """
        Process and ingest trials in batches.
        Returns the number of successfully ingested trials.
        """
        logger.info(f"Processing {len(trials)} trials in batches of {batch_size}")
        
        total_processed = 0
        
        for i in range(0, len(trials), batch_size):
            batch = trials[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(trials) + batch_size - 1) // batch_size
            
            try:
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} trials)")
                
                # Preprocess the batch
                processed_batch = preprocess_trial_data({"studies": batch})
                
                # Add to vector store
                self.rag_manager.add_trials(processed_batch)
                
                total_processed += len(processed_batch)
                logger.info(f"Successfully processed batch {batch_num}. Total processed: {total_processed}")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                continue
        
        return total_processed
    
    def run_pipeline(self, start_date: str = "2024-01-01", max_results: Optional[int] = None, 
                    force_refresh: bool = False) -> bool:
        """
        Run the complete data pipeline.
        
        Args:
            start_date: Start date for fetching trials
            max_results: Maximum number of trials to fetch
            force_refresh: If True, clear existing data and re-ingest
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Starting Clinical Trials Data Pipeline")
            logger.info(f"Parameters: start_date={start_date}, max_results={max_results}, force_refresh={force_refresh}")
            
            # Check existing data
            stats = self.check_existing_data()
            existing_docs = stats.get('total_documents', 0)
            
            if existing_docs > 0 and not force_refresh:
                logger.info(f"Found {existing_docs} existing documents. Use force_refresh=True to re-ingest.")
                return True
            
            # Clear existing data if force_refresh is True
            if force_refresh and existing_docs > 0:
                logger.info("Clearing existing data...")
                self.rag_manager.clear_database()
            
            # Fetch trials
            logger.info("Fetching clinical trials data...")
            trials = self.fetch_trials_in_batches(start_date, max_results)
            
            if not trials:
                logger.warning("No trials found for the specified criteria")
                return False
            
            # Process and ingest
            logger.info("Processing and ingesting trials...")
            processed_count = self.process_and_ingest_trials(trials)
            
            # Final stats
            final_stats = self.rag_manager.get_database_stats()
            logger.info(f"Pipeline completed successfully!")
            logger.info(f"Final database stats: {final_stats}")
            logger.info(f"Total trials processed: {processed_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return False

def main():
    """Main function to run the data pipeline."""
    import argparse
    
    # Debug information
    print(f"Current working directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path[:3]}")  # Show first 3 entries
    print(f"OPENAI_API_KEY set: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
    
    parser = argparse.ArgumentParser(description="Clinical Trials Data Pipeline")
    parser.add_argument("--start-date", default="2024-01-01", 
                       help="Start date for fetching trials (YYYY-MM-DD)")
    parser.add_argument("--max-results", type=int, default=None,
                       help="Maximum number of trials to fetch")
    parser.add_argument("--force-refresh", action="store_true",
                       help="Clear existing data and re-ingest")
    parser.add_argument("--batch-size", type=int, default=500,
                       help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = ClinicalTrialsDataPipeline()
    
    # Run pipeline
    success = pipeline.run_pipeline(
        start_date=args.start_date,
        max_results=args.max_results,
        force_refresh=args.force_refresh
    )
    
    if success:
        logger.info("Data pipeline completed successfully!")
        sys.exit(0)
    else:
        logger.error("Data pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 