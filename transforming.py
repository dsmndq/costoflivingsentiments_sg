import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

def generate_transformer_embeddings(corpus: list[str]) -> np.ndarray:
    """
    Generates a matrix of document embeddings for a given corpus of text.
    
    This function processes a list of documents, generates token-level embeddings
    using a pre-trained transformer model, and then aggregates these tokens into
    a single, fixed-size vector for each document by averaging them.

    Args:
        corpus: A list of strings, where each string is a document.

    Returns:
        A 2D NumPy array where each row is the embedding for a document.
    """
    # 1. Initialize a pre-trained transformer model and tokenizer
    # Using a smaller model like 'distilbert-base-uncased' for efficiency
    model_name = 'distilbert-base-uncased'
    print(f"Loading tokenizer and model for '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    document_embeddings = []
    print("Generating embeddings for each document...")

    # Process each document in the corpus
    for document in corpus:
        # 2. Tokenize the document
        # padding=True and truncation=True ensure all inputs are handled,
        # but the key is how we handle the output later.
        inputs = tokenizer(document, return_tensors='pt', padding=True, truncation=True, max_length=512)

        # 3. Get model outputs (no gradient calculation needed)
        with torch.no_grad():
            outputs = model(**inputs)

        # 4. Get the token embeddings from the last hidden state
        # The shape is (batch_size, num_tokens, embedding_dim)
        token_embeddings = outputs.last_hidden_state

        # --- THIS IS THE CRITICAL FIX ---
        # 5. Aggregate token embeddings into a single document vector.
        # We average the embeddings of all tokens in the document.
        # The .mean(dim=1) operation collapses the token dimension,
        # resulting in a single vector per document.
        # The .squeeze() removes any leftover batch dimension.
        document_vector = token_embeddings.mean(dim=1).squeeze()
        # --- END OF FIX ---

        # 6. Append the fixed-size vector to our list
        # We detach it from the computation graph and convert to a NumPy array
        document_embeddings.append(document_vector.detach().numpy())

    # 7. Create the final NumPy matrix
    # This now works because every element in document_embeddings is a
    # 1D array of the same fixed size (e.g., 768).
    embedding_matrix = np.array(document_embeddings)
    
    print("Embedding matrix generated successfully.")
    return embedding_matrix

# --- Example Usage ---
if __name__ == '__main__':
    # A sample corpus with documents of different lengths
    # This would have caused the original error
    sample_corpus = [
        "This is the first document.",
        "This one is a little bit longer.",
        "Transformers are a powerful tool in modern natural language processing.",
        "Short.",
        "Another sentence to process."
    ]

    print(f"Processing a corpus of {len(sample_corpus)} documents.")

    # Generate the embedding matrix
    transformer_embedding_matrix = generate_transformer_embeddings(sample_corpus)

    # Print the shape to verify the result
    # The shape should be (number of documents, embedding dimension)
    # For distilbert-base-uncased, the embedding dimension is 768.
    # So, the expected shape is (5, 768).
    print("\nShape of the final embedding matrix:", transformer_embedding_matrix.shape)
    
    # Verify that the output is a valid NumPy array with floating-point numbers
    print("Data type of the matrix:", transformer_embedding_matrix.dtype)
    print("\nFirst 5 elements of the first document's embedding:")
    print(transformer_embedding_matrix[0, :5])

    # --- Improvement: Save the output matrix to a file ---
    output_filename = 'transformer_embeddings.npy'
    np.save(output_filename, transformer_embedding_matrix)
    print(f"\nEmbedding matrix successfully saved to '{output_filename}'")
