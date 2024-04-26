import sys
import json
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Your existing Python code for preprocessing, recommendation etc.

# load data from pickle file
data = pd.read_pickle('/Users/abhijaysingh/Desktop/IR_Final/travall2/backend/final_data.pkl')


# Load the BERT model
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Function to get the embeddings of a sentence
def get_embeddings(sentence):
    return model.encode(sentence)

def recommend_places(user_query, city, wheelchair_accessibility, visual_accessibility, top_n=5):
    # Assuming the dataset is loaded as `data` and preprocessing functions are defined and imported
    # Preprocess the user query
    user_query = preprocess_text(user_query)
    user_query = remove_stopwords(user_query)
    user_query = lemmatize_text(user_query)

    # Compute embeddings for user query
    query_embedding = get_embeddings(user_query)
    
    # Filter data based on city
    if city.lower() == 'india':
        city_data = data
    else:
        city_data = data[data['City'].str.lower() == city.lower()]

    # Calculate similarity between user query and reviews in the specified city
    embeddings = np.array(city_data['Relevant Review Embeddings'].tolist())

    # print embeddings type

    similarity_scores = cosine_similarity(embeddings, query_embedding.reshape(1, -1)).flatten()
    
    # Normalize the similarity scores to be between 0 and 1
    similarity_scores = (similarity_scores - similarity_scores.min()) / (similarity_scores.max() - similarity_scores.min())
    
    # Calculate combined scores based on accessibility preferences
    city_data = city_data.copy()  # Avoid SettingWithCopyWarning
    city_data['Similarity'] = similarity_scores

    if wheelchair_accessibility:
        city_data['Normalized_Wheelchair_Score'] = city_data['Wheelchair_Accessibility_Score_x'] / 10
    else:
        city_data['Normalized_Wheelchair_Score'] = 0
    
    if visual_accessibility:
        city_data['Normalized_Visual_Score'] = city_data['Visual_Accessibility_Score_x'] / 10
    else:
        city_data['Normalized_Visual_Score'] = 0

    # Calculate the combined score
    city_data['Combined_Score'] = 0.8 * (city_data['Normalized_Wheelchair_Score'] + city_data['Normalized_Visual_Score']) + 0.2 * city_data['Similarity']

    # Sort data by the combined score in descending order
    top_recommendations = city_data.sort_values(by='Combined_Score', ascending=False).head(top_n)
    
    return top_recommendations

# Function to preprocess the text
def preprocess_text(text):
    # Convert the text to lowercase
    text = text.lower()
    # Remove the special characters
    text = re.sub(r'\W', ' ', text)
    # Remove the digits
    text = re.sub(r'\d', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove leading and trailing spaces
    text = text.strip()
    return text

# Function to remove the stopwords
def remove_stopwords(text):
    # Tokenize the text
    words = word_tokenize(text)
    # Remove the stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)  

# Function to lemmatize the text
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    # Tokenize the text
    words = word_tokenize(text)
    # Lemmatize the text
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

if __name__ == '__main__':
    city = sys.argv[1]
    search_query = sys.argv[2]
    wheelchair_accessibility = sys.argv[3].lower() == 'true'
    visual_accessibility = sys.argv[4].lower() == 'true'
    top_n = 3

    recommendations = recommend_places(search_query, city, wheelchair_accessibility, visual_accessibility, top_n)
    recommendations_list = recommendations['Place'].tolist()
    
    # Add line breaks between recommendations
    recommendations_with_line_breaks = []
    for i, recommendation in enumerate(recommendations_list, start=1):
        recommendations_with_line_breaks.append(f"{i}.{recommendation}  ")
        if i < len(recommendations_list):  # Add a blank line if not the last recommendation
            recommendations_with_line_breaks.append("")  # Add a blank line
    
    # Convert recommendations to a JSON string with line breaks
    print(json.dumps(recommendations_with_line_breaks))

    
# import sys
# import json
# import pandas as pd
# import numpy as np
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# import torch

# # Load the GPT-2 tokenizer and model
# tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
# model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")

# # Load the BERT model
# model = SentenceTransformer('bert-base-nli-mean-tokens')

# # Load data from pickle file
# data = pd.read_pickle('/Users/abhijaysingh/Downloads/travall2/backend/final_data.pkl')

# # Function to get the embeddings of a sentence
# def get_embeddings(sentence):
#     return model.encode(sentence)

# # Function to preprocess the text
# def preprocess_text(text):
#     # Convert the text to lowercase
#     text = text.lower()
#     # Remove the special characters
#     text = re.sub(r'\W', ' ', text)
#     # Remove the digits
#     text = re.sub(r'\d', ' ', text)
#     # Remove extra spaces
#     text = re.sub(r'\s+', ' ', text)
#     # Remove leading and trailing spaces
#     text = text.strip()
#     return text

# # Function to remove the stopwords
# def remove_stopwords(text):
#     # Tokenize the text
#     words = word_tokenize(text)
#     # Remove the stopwords
#     words = [word for word in words if word not in stopwords.words('english')]
#     return ' '.join(words)  

# # Function to lemmatize the text
# def lemmatize_text(text):
#     lemmatizer = WordNetLemmatizer()
#     # Tokenize the text
#     words = word_tokenize(text)
#     # Lemmatize the text
#     words = [lemmatizer.lemmatize(word) for word in words]
#     return ' '.join(words)

# # Function to generate recommendations using GPT-2
# def generate_recommendations_gpt2(user_query, city, top_n=3):
#     # Preprocess the user query
#     user_query = preprocess_text(user_query)
#     user_query = remove_stopwords(user_query)
#     user_query = lemmatize_text(user_query)

#     # Generate recommendations using GPT-2
#     recommendations = []
#     for _ in range(top_n):
#         input_text = f"The best places to visit in {city} if you want {user_query}."
#         input_ids = tokenizer_gpt2.encode(input_text, return_tensors="pt")
#         output = model_gpt2.generate(input_ids, max_length=100, num_return_sequences=1)  # Generate only one sequence at a time
#         recommendation = tokenizer_gpt2.decode(output[0], skip_special_tokens=True)
#         recommendations.append(recommendation)

#     return recommendations

# # Function to recommend places based on similarity
# def recommend_places(user_query, city, wheelchair_accessibility, visual_accessibility, top_n=5):
#     # Preprocess the user query
#     user_query = preprocess_text(user_query)
#     user_query = remove_stopwords(user_query)
#     user_query = lemmatize_text(user_query)

#     # Compute embeddings for user query
#     query_embedding = get_embeddings(user_query)
    
#     # Filter data based on city
#     if city.lower() == 'india':
#         city_data = data
#     else:
#         city_data = data[data['City'].str.lower() == city.lower()]

#     # Calculate similarity between user query and reviews in the specified city
#     embeddings = np.array(city_data['Relevant Review Embeddings'].tolist())
#     similarity_scores = cosine_similarity(embeddings, query_embedding.reshape(1, -1)).flatten()
    
#     # Normalize the similarity scores to be between 0 and 1
#     similarity_scores = (similarity_scores - similarity_scores.min()) / (similarity_scores.max() - similarity_scores.min())
    
#     # Calculate combined scores based on similarity and accessibility preferences
#     city_data = city_data.copy()  # Avoid SettingWithCopyWarning
#     city_data['Similarity'] = similarity_scores

#     if wheelchair_accessibility:
#         city_data['Normalized_Wheelchair_Score'] = city_data['Wheelchair_Accessibility_Score_x'] / 10
#     else:
#         city_data['Normalized_Wheelchair_Score'] = 0
    
#     if visual_accessibility:
#         city_data['Normalized_Visual_Score'] = city_data['Visual_Accessibility_Score_x'] / 10
#     else:
#         city_data['Normalized_Visual_Score'] = 0

#     # Calculate the combined score
#     city_data['Combined_Score'] = 0.8 * (city_data['Normalized_Wheelchair_Score'] + city_data['Normalized_Visual_Score']) + 0.2 * city_data['Similarity']

#     # Sort data by the combined score in descending order
#     top_recommendations = city_data.sort_values(by='Combined_Score', ascending=False).head(top_n)
    
#     return top_recommendations

# if __name__ == '__main__':
#     city = sys.argv[1]
#     search_query = sys.argv[2]
#     wheelchair_accessibility = sys.argv[3].lower() == 'true'
#     visual_accessibility = sys.argv[4].lower() == 'true'
#     use_gpt2 = sys.argv[5].lower() == 'true'
#     top_n = 3

#     if use_gpt2:
#         recommendations = generate_recommendations_gpt2(search_query, city, top_n)
#     else:
#         recommendations = recommend_places(search_query, city, wheelchair_accessibility, visual_accessibility, top_n)
    
#     # Convert recommendations to a JSON string
#     recommendations_json = recommendations.to_json(orient='records')
#     print(recommendations_json)
