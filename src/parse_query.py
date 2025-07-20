# src/parse_query.py

"""
A simple utility for parsing and potentially enriching the user's query.
For this version, it's a placeholder for more advanced NLP tasks like
named entity recognition (NER) to extract fields like age, location, etc.
Currently, it performs basic cleaning.
"""

from utils.logging_config import get_logger

logger = get_logger(__name__)

def parse_and_clean_query(query: str) -> str:
    """
    Cleans and prepares the user query.

    In a more advanced implementation, this function would:
    - Use a library like SpaCy or an LLM call to extract structured entities
      (e.g., age, procedure, location, policy duration).
    - These entities could then be used for more precise metadata filtering
      in the hybrid search step.
    
    For now, it performs basic stripping of whitespace.

    Args:
        query (str): The raw user query.

    Returns:
        str: The cleaned query.
    """
    logger.info(f"Parsing raw query: '{query}'")
    cleaned_query = query.strip()
    # Future enhancement: Add NER here
    # entities = extract_entities(cleaned_query)
    # logger.info(f"Extracted entities: {entities}")
    return cleaned_query


if __name__ == '__main__':
    print("--- Testing Query Parser ---")
    raw_query = "  46-year-old male, knee surgery in Pune, 3-month-old insurance policy   "
    cleaned = parse_and_clean_query(raw_query)
    
    print(f"Raw query: '{raw_query}'")
    print(f"Cleaned query: '{cleaned}'")
    
    # Example of what an advanced version would do:
    # expected_entities = {
    #     "age": 46,
    #     "gender": "male",
    #     "procedure": "knee surgery",
    #     "location": "Pune",
    #     "policy_duration_months": 3
    # }
    # print(f"\nFuture implementation would extract entities like: {expected_entities}")
