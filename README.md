# Travall: Travel for All

Travall is a cutting-edge travel planning application designed to enhance travel experiences for all individuals, especially those with special needs, by providing personalized, accessible travel recommendations.

## Project Overview

Travall stands out with its innovative integration of detailed accessibility features, advanced machine learning models, and user-centered design to offer a unique travel planning tool. This project addresses the gap in the current travel and tourism landscape by providing a unified platform with comprehensive, inclusive destination recommendations.

## Key Features

- **Accessibility-Focused**: Integrates detailed accessibility data for travelers with special needs.
- **Personalized Recommendations**: Uses natural language processing to tailor travel suggestions based on user preferences.
- **Real-Time Data**: Leverages crowdsourced reviews and updates to ensure current and relevant information.
- **Advanced Data Processing**: Utilizes machine learning and NLP for sophisticated data analysis and recommendation logic.

## Methodology

The Travall system utilizes a multi-step process to provide personalized and accessible travel recommendations. Here's a brief overview of our methodology:

### Data Collection and Preprocessing
We begin by compiling a dataset of user-generated reviews from various cities, focusing on the most reviewed places. This dataset undergoes thorough preprocessing to ensure clean and structured data for analysis.

### Feature Extraction Using TF-IDF
TF-IDF analysis is performed to identify key themes and descriptors within the reviews, which helps in highlighting critical features for further analysis.

### Embedding Generation with Sentence-BERT
We use Sentence-BERT to generate embeddings from the identified key words. These embeddings capture the semantic nuances of the reviews, which are essential for accurate recommendation.

### Fine-Tuning BERT for Accessibility Ratings
A BERT model is fine-tuned to derive accessibility ratings for travel destinations, focusing on aspects like wheelchair access and visual accessibility.

### Combining Datasets
The embeddings and the outputs from the fine-tuned BERT model are combined into a comprehensive dataset that serves as the foundation for our recommendation engine.

### Implementation of Retrieval-Augmented Generation (RAG)
A Retrieval-Augmented Generation model is utilized to dynamically retrieve and generate content relevant to the user's queries, enriching the contextual information available for recommendations.

### User Query Processing and Recommendation Generation
User queries are transformed into embeddings and assessed for similarity with destination embeddings. Recommendations are then filtered based on accessibility scores and user preferences.

Through this methodology, Travall aims to revolutionize the travel industry by providing a highly accessible, personalized travel planning tool that caters to the needs of all travelers, especially those with special requirements.


## Prerequisites

- Node.js
- npm (Node Package Manager)
- Python 3.x
- Git (for version control)

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/your-username/travall.git
cd travall
```

## Set up the Backend
Navigate to the backend directory, install dependencies, and start the server.

```bash
cd backend
npm install
npm start
```
In a separate terminal, start the development server for the backend.

```bash
npm run dev
```
The backend servers will be available at http://localhost:3002 and http://localhost:3003 respectively.

## Set up the Frontend
Navigate to the frontend directory from the project root, install dependencies, and start the development server.

```bash
cd frontend
npm install
npm run dev
```
The application will open in your default browser at  http://localhost:5173/.

## Running Python Scripts
Ensure your Python scripts are within the backend directory and are set to handle input parameters correctly.

## Usage
- Use the application via the browser interface.
- Enter username, 'username' as a sample and login.
- Enter search queries as to where you want to go while selecting a city of your choice (or whole India) and an accessibility option.
- The backend handles data processing with Python scripts to return personalized travel recommendations.

## API Endpoints
- POST /submit: Accepts search parameters and returns processed data.
- GET /message: Provides initial data for the frontend.
- POST /authenticate: Handles user authentication.
- POST /run-python: Runs the Python script with the given arguments and returns its output.


## Evaluation
Our system's effectiveness has been rigorously evaluated through various machine learning techniques and model tuning to ensure high accuracy and user satisfaction in our travel recommendations.
