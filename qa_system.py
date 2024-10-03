import cohere
from config import API_KEY as cohere_api_key


# Initialize the Cohere client with your API key

cohere_client = cohere.Client(cohere_api_key)

def answer_question(question, context, input_tokens1):
    """
    Uses Cohere API to generate an answer to the provided question based on the given context.
    """
    try:
        # Concatenate the context and question for a better prompt
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        
        # Call Cohere's generate function to get an answer
        response = cohere_client.generate(
            model='command-xlarge-nightly',
            prompt=prompt,
            max_tokens=input_tokens1,  # Limit the response length
            temperature=0.7  # Adjust for creativity
        )
        
        # Extract and return the answer from the response
        if response.generations:
            return response.generations[0].text.strip()
        else:
            return "Sorry, I couldn't find that information in the document."
    except Exception as e:
        return f"Error using Cohere: {e}"
