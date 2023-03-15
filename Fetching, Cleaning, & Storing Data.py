import praw
import snowflake.connector
import re
from textblob import TextBlob
from datetime import datetime

# Set up Reddit API credentials
reddit = praw.Reddit(
    client_id="CLIENT_ID",
    client_secret="CLIENT_SECRET",
    username="USERNAME",
    password="PASSWORD",
    user_agent="Sentiment Analysis/0.0.1"
)

# Set up Snowflake credentials and connection
ctx = snowflake.connector.connect(
    user='USERNAME',
    password='PASSWORD',
    account='DATABASE_ACCOUNT',
    warehouse='COMPUTE_WH',
    database='REDDIT',
    schema='PUBLIC',
)
cur = ctx.cursor()

subreddit_name = "datascience"
num_posts = 10000

# Retrieve information from Reddit
subreddit = reddit.subreddit(subreddit_name)
new_posts = subreddit.new(limit=num_posts)

create_table_sql = "CREATE TABLE data_science(" \
                   "subreddit VARCHAR(500)," \
                   "title VARCHAR(1000)," \
                   "selftext VARCHAR(50000)," \
                   "upvote_ratio INT," \
                   "ups INT," \
                   "downs INT," \
                   "score INT," \
                   "sentiment FLOAT," \
                   "link_flair_text VARCHAR(500)," \
                   "created_utc NUMBER," \
                   "created_date DATE" \
                   "); "

cur.execute(create_table_sql)

def clean_text(text):
    """
    Utility function to clean text by removing links, special characters, and converting text to lowercase.
    """
    # remove links
    text = re.sub(r'http\S+', '', text)
    # remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    # convert text to lowercase
    text = text.lower()
    return text

for post in new_posts:
    post_dict = vars(post)
    # combine the title and body of the post
    text = post_dict['title'] + ' ' + post_dict['selftext']
    # clean the text
    cleaned_text = clean_text(text)
    # use TextBlob to perform sentiment analysis on the cleaned text
    blob = TextBlob(cleaned_text)
    # get the sentiment polarity of the text
    sentiment = blob.sentiment.polarity
    # get the created date from the created_utc timestamp
    created_utc = datetime.utcfromtimestamp(post_dict['created_utc'])
    created_date = created_utc.date()
    # insert the data into the database
    cur.execute(
        "INSERT INTO data_science (subreddit, title, selftext, upvote_ratio, ups, downs, score, sentiment, link_flair_text, created_utc, created_date) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (post_dict['subreddit'].display_name, post_dict['title'], post_dict['selftext'], post_dict['upvote_ratio'],
         post_dict['ups'], post_dict['downs'], post_dict['score'], sentiment,
         post_dict['link_flair_text'], post_dict['created_utc'], created_date),
    )
    ctx.commit()
