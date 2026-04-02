import sys
import numpy as np

try:
    import ollama
except ImportError:
    print("FATAL: The 'ollama' Python package is required. Install it using: pip install ollama")
    sys.exit(1)

def main():
    print("\n" + "="*70)
    print("VECTOR AND BIASES DEMONSTRATION WITH OLLAMA")
    print("="*70 + "\n")

    # 1. The Vector (Embedding)
    # We use Ollama to convert human text into a numeric vector (embedding).
    text_input = "Artificial Intelligence is transforming software development."
    print(f" Requesting Vector embedding for text: '{text_input}'")

    try:
        # We are using llama3.2:3b, but you can swap this with a dedicated 
        # embedding model like 'nomic-embed-text' or 'mxbai-embed-large'
        response = ollama.embed(model='llama3.2:3b', input=text_input)
        
        # The API returns a 2D array, we extract the first sequence
        input_vector = np.array(response['embeddings'])
        vector_length = len(input_vector)
        
        print(f"✓ Success! Ollama returned a semantic vector of length {vector_length}.")
        print(f"✓ First 5 dimensions of the vector: {input_vector[:5]}\n")
        
    except Exception as e:
        print(f"✗ Error connecting to Ollama: {e}")
        print("Please ensure the Ollama daemon is running and the model is pulled.")
        sys.exit(1)

    # 2. Weights and Biases (Simulating a Neural Network Layer)
    # We will simulate passing this vector into a simple neural network classification layer.
    print(" Initializing Weights and Biases for Neural Network Layer")

    # Let's assume this layer outputs 3 values (e.g., classifying the text into 3 categories)
    output_neurons = 3

    # Weights: A matrix that multiplies our input vector
    # We generate random weights scaled down by 0.01 for realistic initialization
    weights = np.random.randn(vector_length, output_neurons) * 0.01

    # Biases: A separate vector added to the result of the weight multiplication
    # This shifts the activation curve so the network can fit the data better
    biases = np.array([0.5, -0.2, 0.8])

    print(f"✓ Generated Weight matrix of shape: {weights.shape}")
    print(f"✓ Generated Bias vector of shape: {biases.shape}")
    print(f"✓ Bias values: {biases}\n")

    # 3. The Mathematical Execution
    print(" Executing Neural Network Math: Output = (Input * Weights) + Biases")

    # A: Multiply the vector by the weights (Dot Product)
    weighted_sum = np.dot(input_vector, weights)
    print(f"→ Result after applying Weights (Input * Weights): {weighted_sum}")

    # B: Add the Bias vector
    final_output = weighted_sum + biases
    print(f"→ Final Output after adding Biases: {final_output}\n")

    print("="*70)
    print("CONCEPTUAL SUMMARY:")
    print("="*70)
    print("1. VECTOR: Ollama successfully converted your text into an array of numbers")
    print("   representing its semantic meaning.")
    print("2. WEIGHTS: Multiplied against the vector to determine which semantic")
    print("   features were most important.")
    print("3. BIASES: Shifted the final numbers (e.g., adding 0.8 to the third output),")
    print("   allowing the network to adjust its baseline predictions mathematically.")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()