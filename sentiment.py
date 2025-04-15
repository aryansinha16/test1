import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

def preprocess_text(text):
    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    return ' '.join(lemmatized_tokens)

def analyze_sentiment(text):
    cleaned_text = preprocess_text(text)

    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(cleaned_text)

    compound_score = scores['compound']
    if compound_score >= 0.05:
        sentiment = 'Positive'
    elif compound_score <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    print(f"Original Text: {text}")
    print(f"Cleaned Text: {cleaned_text}")
    print(f"Sentiment: {sentiment}")
    print(f"Scores: {scores}")

def main():
    print("Sentiment Analyzer (type 'exit' or 'quit' to stop)")
    while True:
        user_input = input("Enter text to analyze: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting...")
            break
        analyze_sentiment(user_input)

if __name__ == "__main__":
    main()