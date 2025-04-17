from mem0 import Memory
import os

# Custom instructions for memory processing
# These aren't being used right now but Mem0 does support adding custom prompting
# for handling memory retrieval and processing.
CUSTOM_INSTRUCTIONS = """
Extract the Following Information:  

- Key Information: Identify and save the most important details.
- Context: Capture the surrounding context to understand the memory's relevance.
- Connections: Note any relationships to other topics or memories.
- Importance: Highlight why this information might be valuable in the future.
- Source: Record where this information came from when applicable.
"""

def get_mem0_client():
    # Get LLM provider and configuration
    llm_provider = os.getenv('LLM_PROVIDER')
    llm_api_key = os.getenv('LLM_API_KEY')
    llm_model = os.getenv('LLM_CHOICE')
    embedding_model = os.getenv('EMBEDDING_MODEL_CHOICE')
    
    # Initialize config dictionary
    config = {}
    
    # Determine embedding dimensions based on provider
    if llm_provider == 'mistral':
        embedding_dims = 1024  # Mistral default
    elif llm_provider == 'ollama':
        embedding_dims = 768 # Default for nomic-embed-text with Ollama
    else: # Default to OpenAI/OpenRouter dimensions
        embedding_dims = 1536 # Default for text-embedding-3-small
        
    # Configure LLM based on provider
    if llm_provider == 'openai' or llm_provider == 'openrouter' or llm_provider == 'mistral':
        # Treat mistral as openai compatible for the LLM part
        config["llm"] = {
            "provider": "openai", # Use openai provider setting for Mistral API compatibility
            "config": {
                "model": llm_model,
                "temperature": 0.2,
                "max_tokens": 2000,
            }
        }
        
        # Set API key in environment if not already set
        if llm_api_key and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = llm_api_key
            
        # For OpenRouter, set the specific API key
        if llm_provider == 'openrouter' and llm_api_key:
            os.environ["OPENROUTER_API_KEY"] = llm_api_key
        # For Mistral, set the specific API key if needed (using OPENAI_API_KEY env var)
        elif llm_provider == 'mistral' and llm_api_key:
             os.environ["OPENAI_API_KEY"] = llm_api_key # Mem0 uses OPENAI_API_KEY for Mistral too
             # Optionally set MISTRAL_API_KEY if other parts of your system need it
             # os.environ["MISTRAL_API_KEY"] = llm_api_key
             
        # Set base URL if provided (useful for Mistral, OpenRouter, or self-hosted OpenAI compatible)
        llm_base_url = os.getenv('LLM_BASE_URL')
        if llm_base_url:
            config["llm"]["config"]["base_url"] = llm_base_url
            
    elif llm_provider == 'ollama':
        config["llm"] = {
            "provider": "ollama",
            "config": {
                "model": llm_model,
                "temperature": 0.2,
                "max_tokens": 2000,
            }
        }
        
        # Set base URL for Ollama if provided
        llm_base_url = os.getenv('LLM_BASE_URL')
        if llm_base_url:
            config["llm"]["config"]["ollama_base_url"] = llm_base_url
            
    # Configure embedder based on provider
    if llm_provider == 'openai' or llm_provider == 'openrouter':
        config["embedder"] = {
            "provider": "openai",
            "config": {
                "model": embedding_model or "text-embedding-3-small",
                "embedding_dims": embedding_dims
            }
        }
        # Set API key in environment if not already set
        if llm_api_key and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = llm_api_key
        # For OpenRouter, set the specific API key
        if llm_provider == 'openrouter' and llm_api_key:
            os.environ["OPENROUTER_API_KEY"] = llm_api_key
            
    elif llm_provider == 'mistral':
         config["embedder"] = {
            "provider": "openai", # Use openai provider setting for Mistral API compatibility
            "config": {
                "model": embedding_model or "mistral-embed",
                "embedding_dims": embedding_dims
            }
        }
         # Set API key in environment if not already set
         if llm_api_key and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = llm_api_key # Mem0 uses OPENAI_API_KEY for Mistral too
            # Optionally set MISTRAL_API_KEY if other parts of your system need it
            # os.environ["MISTRAL_API_KEY"] = llm_api_key
            
         # Handle Mistral API base URL using the new OPENAI_BASE_URL variable
         embedding_base_url = os.getenv('LLM_BASE_URL')
         if embedding_base_url and llm_provider == 'mistral':
            os.environ["OPENAI_BASE_URL"] = embedding_base_url
            # Clean up old deprecated variable if it exists
            if "OPENAI_API_BASE" in os.environ:
                del os.environ["OPENAI_API_BASE"]

    elif llm_provider == 'ollama':
        config["embedder"] = {
            "provider": "ollama",
            "config": {
                "model": embedding_model or "nomic-embed-text",
                "embedding_dims": embedding_dims
            }
        }
        # Set base URL for Ollama embedder if provided
        embedding_base_url = os.getenv('LLM_BASE_URL') # Use same base URL for embeddings
        if embedding_base_url:
            config["embedder"]["config"]["ollama_base_url"] = embedding_base_url
            
    # Configure Supabase vector store
    config["vector_store"] = {
        "provider": "supabase",
        "config": {
            "connection_string": os.environ.get('DATABASE_URL', ''),
            "collection_name": "mem0_memories",
            "embedding_model_dims": embedding_dims # Use dynamically determined dimensions
        }
    }

    # config["custom_fact_extraction_prompt"] = CUSTOM_INSTRUCTIONS
    
    # Create and return the Memory client
    return Memory.from_config(config)
