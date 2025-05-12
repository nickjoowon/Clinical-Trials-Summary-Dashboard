from data.clinical_trials import fetch_clinical_trials, preprocess_trial_data
from processing.text_processor import prepare_trials_for_embedding, combine_trial_text
from embeddings.embedding_manager import EmbeddingManager

def main():
    # Test data fetching
    print("Fetching clinical trials data...")
    trials_data = fetch_clinical_trials(max_results=1)  # Get just 1 trial for example
    
    # Test preprocessing
    print("\nPreprocessing data...")
    processed_trials = preprocess_trial_data(trials_data)
    
    # Show example of combined text
    print("\nExample of combined trial text:")
    print("-" * 80)
    combined_text = combine_trial_text(processed_trials[0])
    print(combined_text)
    print("-" * 80)
    
    # Test text processing
    print("\nPreparing for embedding...")
    embedding_data = prepare_trials_for_embedding(processed_trials)
    print(f"Created {len(embedding_data)} text chunks")
    
    # Test embedding creation
    print("\nCreating embeddings...")
    embedding_manager = EmbeddingManager()
    texts = [item['text'] for item in embedding_data]
    embeddings = embedding_manager.create_embeddings(texts)
    print(f"Created embeddings with shape: {embeddings.shape}")
    
    # Build and test search
    print("\nBuilding search index...")
    embedding_manager.build_index(embeddings)
    embedding_manager.add_metadata(embedding_data)
    
    # Test search
    print("\nTesting search functionality...")
    test_queries = [
        "pain management",
        "cancer treatment",
        "clinical study"
    ]
    
    for query in test_queries:
        print(f"\nSearching for: '{query}'")
        results = embedding_manager.search(query, k=2)
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Title: {result['metadata']['title']}")
            print(f"Score: {result['score']:.2f}")
            print(f"Text snippet: {result['text'][:200]}...")

if __name__ == "__main__":
    main() 