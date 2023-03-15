import snowflake.connector
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# Load the data from the Snowflake database
ctx = snowflake.connector.connect(
    user='USER_ID',
    password='PASSWORD',
    account='DATABASE_ACCOUNT',
    warehouse='COMPUTE_WH',
    database='REDDIT',
    schema='PUBLIC',
)
df = pd.read_sql_query("SELECT * FROM data_science", ctx)

# histogram of the sentiment scores
sentiment_counts = df['SENTIMENT'].value_counts()

# histogram of sentiment scores with count on y-axis
plt.hist(df['SENTIMENT'], bins=20)
plt.xlabel('Sentiment Score')
plt.ylabel('Count')
plt.title('Distribution of Sentiment Scores')
plt.show()

###Mean Sentiment by Day###

df['CREATED_DATE'] = pd.to_datetime(df['CREATED_DATE'], format='%Y-%m-%d')

# set "CREATED_DATE" as the index
df.set_index('CREATED_DATE', inplace=True)

# calculate daily mean sentiment
daily_sentiment = df.resample('D')['SENTIMENT'].mean()
plt.ylabel('Mean Sentiment Score')
plt.xlabel('Date')

# plot time series of daily mean sentiment
daily_sentiment.plot()
plt.show()

### Mean Sentiment by Day, W/line representing Mondays###
# Convert the date string to a datetime object
df['CREATED_DATE'] = pd.to_datetime(df['CREATED_DATE'], format='%Y-%m-%d')

# Set "created_date" as the index
df.set_index('CREATED_DATE', inplace=True)

# Plot a time series of sentiment scores
df.resample('D')['SENTIMENT'].mean().plot()

# Add vertical lines for Mondays
mondays = df[df.index.weekday == 0].index
for monday in mondays:
    plt.axvline(x=monday, color='r', linestyle='--', alpha=0.5)
    plt.ylabel('Mean Sentiment Score')
    plt.xlabel('Date')
plt.show()

#Define a threshold for positive and negative sentiment
pos_threshold = 0.2
neg_threshold = -0.2

# Create a new column for sentiment classification
df['sentiment_class'] = pd.cut(df['SENTIMENT'], bins=[-1, neg_threshold, pos_threshold, 1], labels=['negative', 'neutral', 'positive'])

# # Plot a bar chart of sentiment classifications
df['sentiment_class'].value_counts().plot(kind='bar')
plt.ylabel('Post Count')
plt.xlabel('Category')
plt.show()

# Create a TF-IDF matrix of the cleaned text

# Train an NMF model with 10 topics
# Get the top 10 words for each topic
for LINK_FLAIR_TEXT, group in df.groupby('LINK_FLAIR_TEXT'):
    print(f"Link Flair Text: {LINK_FLAIR_TEXT}\n")
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(group['SELFTEXT'])

    # Train an NMF model with 10 topics
    nmf = NMF(n_components=10, random_state=1)
    nmf.fit(tfidf_matrix)

    # Get the top 10 words for each topic
    for i, topic in enumerate(nmf.components_):
        print(f"Topic {i}:")
        print([tfidf.get_feature_names()[j] for j in topic.argsort()[:-11:-1]])
    print('\n')

