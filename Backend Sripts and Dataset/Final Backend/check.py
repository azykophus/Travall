import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Download the necessary resources  
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

# load data from pickle file
data = pd.read_pickle('C:/Users/amilb/Desktop/Sem 6/IR/Project/Travall/Final Project/Dataset/Combined_datasets/final_data.pkl')


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

# if python file is run directly
if __name__ == '__main__':
    user_query = "food shopping clothes market"
    city = "india"
    wheelchair_accessibility = True
    visual_accessibility = False
    top_n = 3
    recommendations = recommend_places(user_query, city, wheelchair_accessibility, visual_accessibility, top_n)
    # getting the top 5 recommendations in a list
    recommendations = recommendations['Place'].tolist()
    print("Your top recommendations are:")
    for i, recommendation in enumerate(recommendations):
        print(f"{i + 1}. {recommendation}")