# Improved Image Search System with Better Accuracy

import chromadb
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
from azure.storage.blob import BlobServiceClient
from PIL import Image
from io import BytesIO
import redis
import hashlib
import json

# Initialize CLIP model and processor
print("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("✅ CLIP model loaded")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="chromadb_persist")
collection = client.get_or_create_collection("image_embeddings")

def get_normalized_image_embedding(image):
    """Generate normalized CLIP embedding for better similarity search"""
    try:
        # Ensure image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process image
        inputs = clip_processor(images=image, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            image_embedding = clip_model.get_image_features(**inputs)
            
            # Normalize the embedding for better cosine similarity
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        
        return image_embedding.cpu().numpy().flatten()  # Flatten to 1D array
    except Exception as e:
        print(f"Error generating image embedding: {e}")
        return None

def get_normalized_query_embedding(query):
    """Generate normalized CLIP text embedding"""
    try:
        inputs = clip_processor(text=[query], return_tensors="pt", padding=True)
        
        with torch.no_grad():
            query_embedding = clip_model.get_text_features(**inputs)
            
            # Normalize the embedding
            query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
        
        return query_embedding.cpu().numpy().flatten()  # Flatten to 1D array
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return None

def process_and_store_embeddings_improved():
    """Improved embedding processing with better accuracy"""
    global collection, container_client  # Make these accessible
    
    print("🔄 Processing images for improved embeddings...")
    
    processed_count = 0
    error_count = 0
    
    # Processing images 
    max_images = 1000 
    
    try:
        for idx, blob in enumerate(container_client.list_blobs()):
            if processed_count >= max_images:
                break
                
            try:
                image_name = blob.name
                
                # Skip non-image files
                if not any(image_name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']):
                    continue
                
                print(f"Processing {processed_count + 1}/{max_images}: {image_name}")
                
                # Load image
                image = load_image_from_blob(image_name)
                
                # Generate normalized embedding
                image_embedding = get_normalized_image_embedding(image)
                
                if image_embedding is not None:
                    # Check if embedding already exists
                    try:
                        existing = collection.get(ids=[str(processed_count)])
                        if existing['ids']:
                            print(f"Skipping {image_name} - already processed")
                            processed_count += 1
                            continue
                    except:
                        pass  # ID doesn't exist, continue with adding
                    
                    # Add to ChromaDB with proper metadata
                    collection.add(
                        documents=[image_name],
                        embeddings=[image_embedding.tolist()],
                        metadatas=[{
                            "source": image_name,
                            "processed_date": str(idx),
                            "embedding_model": "clip-vit-base-patch32"
                        }],
                        ids=[str(processed_count)]
                    )
                    
                    processed_count += 1
                    
                    # Progress update
                    if processed_count % max_images == 0:
                        print(f"✅ Processed {processed_count} images successfully")
                else:
                    error_count += 1
                    print(f"❌ Failed to process {image_name}")
                    
            except Exception as e:
                error_count += 1
                print(f"❌ Error processing {blob.name}: {e}")
                continue
    
    except Exception as e:
        print(f"❌ Error accessing blob storage: {e}")
        print("Make sure your Azure connection is properly configured")
        return 0
    
    print(f"🎉 Embedding processing complete!")
    print(f"✅ Successfully processed: {processed_count} images")
    print(f"❌ Errors: {error_count} images")
    return processed_count

def improved_search_images(query, max_results=5):
    """Improved search with better accuracy"""
    try:
        print(f"🔍 Searching for: '{query}'")
        
        # Check cache first
        cached_results = get_cached_results(query)
        if cached_results:
            print("📋 Using cached results")
            return cached_results
        
        # Generate normalized query embedding
        query_embedding = get_normalized_query_embedding(query)
        
        if query_embedding is None:
            print("❌ Failed to generate query embedding")
            return []
        
        # Search in ChromaDB with better parameters
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=max_results,
            include=['documents', 'distances', 'metadatas']
        )
        
        # Extract image names and similarity scores
        if results['documents'] and len(results['documents'][0]) > 0:
            image_names = results['documents'][0]
            distances = results['distances'][0] if 'distances' in results else [0] * len(image_names)
            
            # Convert distances to similarity scores (lower distance = higher similarity)
            similarity_scores = [max(0, 1 - dist) for dist in distances]
            
            # Create structured results
            structured_results = []
            for i, (image_name, similarity) in enumerate(zip(image_names, similarity_scores)):
                structured_results.append({
                    'rank': i + 1,
                    'image_name': image_name,
                    'similarity_score': similarity
                })
            
            # Cache the results
            cache_search_results(query, image_names)
            
            print(f"✅ Found {len(structured_results)} results")
            for result in structured_results:
                print(f"  {result['rank']}. {result['image_name']} (similarity: {result['similarity_score']:.3f})")
            
            return structured_results
        else:
            print("❌ No results found")
            return []
            
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return []

def test_search_accuracy():
    """Test search accuracy with various queries"""
    test_queries = [
        "people",
        "sunset",
        "mountain",
        "ocean",
        "city",
        "forest",
        "children playing",
        "beautiful landscape",
        "urban street",
        "natural scenery"
    ]
    
    print("🧪 Testing search accuracy...")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        results = improved_search_images(query, max_results=3)
        
        if results:
            for result in results:
                print(f"  ✅ {result['rank']}. {result['image_name']} ({result['similarity_score']:.3f})")
        else:
            print("  ❌ No results found")
        
        print("-" * 40)

def rebuild_embeddings_if_needed():
    """Check if embeddings need to be rebuilt for better accuracy"""
    global collection  # Make collection accessible
    
    try:
        # Check current collection size
        collection_info = collection.count()
        print(f"📊 Current collection size: {collection_info} embeddings")
        
        if collection_info == 0:
            print("🔄 No embeddings found. Building embeddings...")
            return process_and_store_embeddings_improved()
        else:
            print("✅ Embeddings exist. Testing search quality...")
            
            # Test with a simple query
            test_result = improved_search_images("people", max_results=1)
            
            if not test_result:
                print("⚠️ Search quality poor. Consider rebuilding embeddings.")
                rebuild = input("Rebuild embeddings? (y/n): ")
                if rebuild.lower() == 'y':
                    # Clear existing collection
                    try:
                        client.delete_collection("image_embeddings")
                        collection = client.create_collection("image_embeddings")
                        print("🗑️ Cleared existing embeddings")
                    except Exception as e:
                        print(f"Warning: Could not clear collection: {e}")
                    return process_and_store_embeddings_improved()
            else:
                print("✅ Search quality acceptable")
                return collection_info
                
    except Exception as e:
        print(f"❌ Error checking embeddings: {e}")
        return 0

# Main execution
if __name__ == "__main__":
    print("🚀 Starting Improved Image Search System")
    
    # Step 1: Check and rebuild embeddings if needed
    embedding_count = rebuild_embeddings_if_needed()
    
    if embedding_count > 0:
        # Step 2: Test search accuracy
        test_search_accuracy()
        
        # Step 3: Interactive search
        print("\n🎯 Interactive Search Mode")
        print("Enter 'quit' to exit")
        
        while True:
            query = input("\n🔍 Enter search query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
                
            if query:
                results = improved_search_images(query, max_results=5)
                
                if results:
                    print(f"\n📊 Top {len(results)} results for '{query}':")
                    for result in results:
                        print(f"  {result['rank']}. {result['image_name']}")
                        print(f"     Similarity: {result['similarity_score']:.3f}")
                else:
                    print("❌ No relevant images found")
    
    print("\n👋 Search system testing complete!")
