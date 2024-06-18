import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

queries = [
    "Hello",
    "How are you?",
    "What is your name?",
    "Tell me a joke",
    "Goodbye"
]

responses = [
    "Hi there!",
    "I'm a bot, so I don't have feelings, but I'm here to help you!",
    "I'm a chatbot created by you.",
    "Why don't scientists trust atoms? Because they make up everything!",
    "Goodbye! Have a great day!"
]

vectorizer = TfidfVectorizer()

all_texts = queries
vectorizer.fit(all_texts)


query_vectors = vectorizer.transform(queries)

def get_response(user_input):
    user_input_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_input_vector, query_vectors)
    most_similar_idx = np.argmax(similarities)
    return responses[most_similar_idx]

def chatbot():
    print("Welcome to the chatbot. Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        
        response = get_response(user_input)
        print(f"Chatbot: {response}")

chatbot()
