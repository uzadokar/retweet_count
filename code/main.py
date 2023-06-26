from functions import extract_tweet_features,clean,get_sentiment_score,extract_features

from flask import Flask, jsonify, request, render_template
import pickle
import numpy as np
import pandas as pd
from datetime import datetime


app = Flask(__name__)

# Load the necessary models and data
# extract_time_f = pickle.load(open("./extract_time_f.pkl", "rb"))
# Tweet_body = pickle.load(open("./Tweet_Body.pkl", "rb"))
# clean = pickle.load(open("./clean.pkl", "rb"))
# get_sentiment_score = pickle.load(open("./get_sentiment_score.pkl", "rb"))
# model = pickle.load(open("./DT.pkl", "rb"))
model = pickle.load(open("dt.pkl", "rb"))



@app.route("/")
def status():
    return "API is working."


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    TweetRetweetFlag = data["TweetRetweetFlag"]
    print("TweetRetweetFlag = " + str(TweetRetweetFlag))

    TweetFavoritesCount = data["TweetFavoritesCount"]
    print("TweetFavoritesCount = " + str(TweetFavoritesCount))

    UserFollowersCount = data["UserFollowersCount"]
    print("UserFollowersCount = " + str(UserFollowersCount))

    UserFriendsCount = data["UserFriendsCount"]
    print("UserFriendsCount = " + str(UserFriendsCount))

    UserTweetCount = data["UserTweetCount"]
    print("UserTweetCount = " + str(UserTweetCount))

    MacroIterationNumber = data["MacroIterationNumber"]
    print("MacroIterationNumber = " + str(MacroIterationNumber))

    TweetPostedTime = data["TweetPostedTime"]
    print("TweetPostedTime = " + str(TweetPostedTime))

    Tweet_Body = data["Tweet_Body"]
    print("Tweet_Body = " + str(Tweet_Body))

    UserSignupDate = data["UserSignupDate"]
    print("UserSignupDate = " + str(UserSignupDate))

    if TweetRetweetFlag:
        TweetRetweetFlag = 1
        print("TweetRetweetFlag = " + str(TweetRetweetFlag))
    else:
        TweetRetweetFlag = 0
        print("TweetRetweetFlag = " + str(TweetRetweetFlag))

    # Extract Time Features
    
    TweetPostedTime_hour, year_of_signup, month_of_signup = extract_features(TweetPostedTime)
    print(TweetPostedTime_hour, year_of_signup, month_of_signup)

    # Derived features
    Total_Activity = TweetFavoritesCount + UserTweetCount
    print(Total_Activity)

    current_year = datetime.now().year
    signup_year = datetime.strptime(UserSignupDate, "%a %b %d %H:%M:%S %z %Y").year
    Age = current_year - signup_year
    print(Age)
    Average_Total_Activity_per_year = Total_Activity / Age
    print(Average_Total_Activity_per_year)

    count_words, count_mentions, count_hashtags, count_urls, count_emojis = extract_tweet_features(Tweet_Body)
    print(count_words, count_mentions, count_hashtags, count_urls, count_emojis)

    cleaned_tweet = clean(Tweet_Body)

    Senti_score = get_sentiment_score(cleaned_tweet)

    print("Senti_score:",Senti_score)

    

    # Model: Decision tree
    array = np.zeros(18)

    array[0] = TweetRetweetFlag
    array[1] = TweetFavoritesCount
    array[2] = UserFollowersCount
    array[3] = UserFriendsCount
    array[4] = UserTweetCount
    array[5] = MacroIterationNumber
    array[6] = TweetPostedTime_hour
    array[7] = year_of_signup
    array[8] = month_of_signup
    array[9] = Total_Activity
    array[10] = Age
    array[11] = Average_Total_Activity_per_year
    array[12] = count_words
    array[13] = count_mentions
    array[14] = count_hashtags
    array[15] = count_urls
    array[16] = count_emojis
    array[17] = Senti_score

    prediction = model.predict([array])
    prediction = prediction[0]

    return f"prediction =  {prediction}"


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)


# #{
#     "TweetRetweetFlag": "True",
#     "TweetFavoritesCount": 0,
#     "UserFollowersCount": 811,
#     "UserFriendsCount": 77,
#     "UserTweetCount": 10579,
#     "MacroIterationNumber": 0,
#     "TweetPostedTime": "Tue Dec 20 10:56:48 +0000 2016",
#     "Tweet_Body": "RT @hocais: #Rize #Turkey\n#Ayder\n#CityOfAllSeasons\n#HerMevsiminKenti\n#Travel\nGörmelisin https://t.co/6gJQObKA8y",
#     "UserSignupDate": "Wed Jan 11 11:12:20 +0000 2012"