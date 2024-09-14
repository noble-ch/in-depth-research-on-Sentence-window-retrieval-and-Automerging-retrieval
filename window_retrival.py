import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# nltk.data.path.append('/home/path/custom_nltk_data')

nltk.download('punkt')

# nltk.download('punkt')  

# Sample text document
document = """
Retrieval-Augmented Generation (RAG) is a hybrid approach in artificial intelligence, combining retrieval mechanisms and generation models. It focuses on how machines can retrieve relevant documents or data from large corpora and then generate coherent responses or insights based on that information. RAG systems excel in tasks involving large amounts of information, addressing challenges such as producing accurate, informative, and contextually relevant content.
"""

# Function to split document into sentence windows
def create_sentence_windows(text, window_size=2):
    sentences = sent_tokenize(text)
    windows = []
    for i in range(len(sentences) - window_size + 1):
        window = ' '.join(sentences[i:i + window_size])
        windows.append(window)
    return windows

# Function to retrieve the best matching sentence window for a query
def retrieve_best_window(query, sentence_windows):
    vectorizer = TfidfVectorizer().fit_transform(sentence_windows + [query])
    vectors = vectorizer.toarray()
    
    # Compute cosine similarity between query and each sentence window
    cosine_similarities = cosine_similarity(vectors[-1].reshape(1, -1), vectors[:-1]).flatten()
    
    # Get the index of the most similar window
    best_window_index = cosine_similarities.argmax()
    
    return sentence_windows[best_window_index], cosine_similarities[best_window_index]

# Create sentence windows from the document
sentence_windows = create_sentence_windows(document, window_size=2)

# Example query
query = "retrieval-augmented generation combines retrieval and generation models to process and generate responses based on large corpora"
    
# Retrieve the best matching window
best_window, similarity = retrieve_best_window(query, sentence_windows)

# Display the result
print("Best Matching Sentence Window:")
print(best_window)
print(f"Similarity Score: {similarity}")
