import kagglehub
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

# Required NLTK downloads
nltk.download('stopwords')
nltk.download('wordnet')

# 1. Data Acquisition
def load_data():
    path = kagglehub.dataset_download("goyaladi/twitter-dataset")
    csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
    full_df = pd.read_csv(os.path.join(path, csv_file))
    # Requirement: Using 1,000 entries for the development phase
    return full_df.sample(n=1000, random_state=42).reset_index(drop=True)

# 2. Preprocessing
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text) # Remove URLs
    text = re.sub(r'[^a-z\s]', '', text) # Remove special characters/numbers
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Tokenization, Stopword removal, and Lemmatization
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return tokens # Returning as list for Coherence calculation

# 3. ENTITY ANALYSIS (Hashtags & Users) 
def entity_analysis(df):
    # Analyzing most active users
    # Adjust column name based on your dataset inspection (e.g., 'user_name')
    print("\n--- TOP 5 ACTIVE USERS ---")
    print(df['user_name'].value_counts().head(5))
    
    # Extracting hashtags from the original text
    all_hashtags = df['text'].apply(lambda x: re.findall(r"#(\w+)", str(x)))
    flat_hashtags = [tag for sublist in all_hashtags for tag in sublist]
    print("\n--- TOP 5 HASHTAGS ---")
    print(pd.Series(flat_hashtags).value_counts().head(5))

# 4. TOPIC EXTRACTION (Extract 5 Topics) [cite: 120, 147]
def calculate_coherence(model, feature_names, tokenized_texts):
    # Convert model topics into a list of words for Gensim
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topics.append([feature_names[i] for i in topic.argsort()[:-11:-1]])
    
    # Use Gensim to calculate Coherence Score
    common_dictionary = Dictionary(tokenized_texts)
    coherence_model = CoherenceModel(topics=topics, 
                                     texts=tokenized_texts, 
                                     dictionary=common_dictionary, 
                                     coherence='c_v')
    return coherence_model.get_coherence()

if __name__ == "__main__":
    df = load_data()
    
    # Apply preprocessing
    df['tokens'] = df['text'].apply(preprocess_text)
    df['clean_text'] = df['tokens'].apply(lambda x: " ".join(x))
    
    # Perform Entity Analysis 
    entity_analysis(df)
    
    # Vectorization (Comparing TF-IDF and CountVectorizer) [cite: 75]
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['clean_text'])
    
    # Topic Modeling: LDA (Latent Dirichlet Allocation) [cite: 120, 148]
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(tfidf_matrix)
    
    # Evaluation with Coherence Score
    score = calculate_coherence(lda, tfidf_vectorizer.get_feature_names_out(), df['tokens'].tolist())
    print(f"\nLDA Coherence Score: {score:.4f}")
    
    print("\n--- TOP 5 TOPICS (LDA) ---")
    for idx, topic in enumerate(lda.components_):
        print(f"Topic {idx}: {[tfidf_vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-5:]]}")

# Datensatz laden
df = load_data()

# 1. Die ersten Zeilen betrachten
print("--- FIRST ROWS ---")
print(df.head())

# 2. Struktur und Datentypen prüfen (Wichtig für die 'Quality of Implementation')
print("\n--- DATA INFO ---")
print(df.info())

# 3. Spaltennamen identifizieren
print("\n--- COLUMNS ---")
print(df.columns)

# 4. Fehlende Werte prüfen
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())