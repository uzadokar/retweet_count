import re
import string
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import TextBlob

# 1
def extract_tweet_features(tweet):
    count_words = len(re.findall(r'\w+', tweet))
    count_mentions = len(re.findall(r'@\w+', tweet))
    count_hashtags = len(re.findall(r'#\w+', tweet))
    count_urls = len(re.findall(r'http.?://[^\s]+[\s]?', tweet))
    count_emojis = len(re.findall(r':[a-z_&]+:', emoji.demojize(tweet)))

    return count_words, count_mentions, count_hashtags, count_urls, count_emojis


#  2

def clean(data):

    stopwords_list = set(stopwords.words('english'))
    porter = PorterStemmer()

    def remove_mentions(data):
        return re.sub(r'@\w+', '', data)

    def remove_urls(data):
        return re.sub(r'http.?://[^\s]+[\s]?', '', data)

    def emoji_oneword(data):
        return data.replace('_', '')

    def remove_punctuation(data):
        punct = string.punctuation
        trantab = str.maketrans(punct, len(punct) * ' ')
        return data.translate(trantab)

    def remove_digits(data):
        return re.sub(r'\d+', '', data)

    def to_lower(data):
        return data.lower()

    def remove_stopwords(data):
        words = data.split()
        clean_words = [word for word in words if (word not in stopwords_list) and len(word) > 1]
        return " ".join(clean_words)

    def stemming(data):
        words = data.split()
        stemmed_words = [porter.stem(word) for word in words]
        return " ".join(stemmed_words)

    def clean_text(data):
        clean_data = remove_mentions(data)
        clean_data = remove_urls(clean_data)
        clean_data = emoji_oneword(clean_data)
        clean_data = remove_punctuation(clean_data)
        clean_data = remove_digits(clean_data)
        clean_data = to_lower(clean_data)
        clean_data = remove_stopwords(clean_data)
        clean_data = stemming(clean_data)
        return clean_data

    cleaned_data = clean_text(data)
    return cleaned_data

# 3
from textblob import TextBlob

def get_sentiment_score(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score

# 4
def extract_features(tweet_posted_time):
    # Define a mapping for month abbreviations
    month_mapping = {
        'Jan': 1,
        'Feb': 2,
        'Mar': 3,
        'Apr': 4,
        'May': 5,
        'Jun': 6,
        'Jul': 7,
        'Aug': 8,
        'Sep': 9,
        'Oct': 10,
        'Nov': 11,
        'Dec': 12
    }

    # Extract hour, year, and month from tweet_posted_time
    TweetPostedTime_hour = int(tweet_posted_time.split()[3].split(':')[0])
    year_of_signup = int(tweet_posted_time.split()[-1])
    month_abbreviation = tweet_posted_time.split()[1]
    month_of_signup = month_mapping.get(month_abbreviation)

    return TweetPostedTime_hour,year_of_signup,month_of_signup