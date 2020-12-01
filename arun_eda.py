import pandas as pd
import numpy as np

import streamlit as st
import plotly

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

st.title("Arun District Travel Review Data 2019")
st.sidebar.subheader("Dashboards")
dashboard_choice = st.sidebar.selectbox("Please Choose Dashboard",("Exploratory Data Analysis","Machine Learning Models","Neural Network Models","Sentiment Analyser Tool"),key = "main")
st.markdown("This application is a Streamlit Dashboard to analyse Tourist Reviews in Arun District️ in 2019")
st.text("Reviews are classed as postive relating to user review ratings of 4 and 5, neutral - ratings of 3, and negative - ratings of 1 and 2. For more information on the features please refer to the main report.")
#st.sidebar.title("Arun District Travel Review Data 2019")
st.sidebar.subheader("Arun District Travel Review Data")

# Load dataset and cache the output
DATA_URL = ("GitHub/Auckland68/Travel-Reviews-EDA-App/new_data.csv")

@st.cache(persist = True)
def load_data():
    data = pd.read_csv(DATA_URL)
    return data

data = load_data()

# Allow user to view raw data
if dashboard_choice =="Exploratory Data Analysis":
    if st.sidebar.checkbox('Show Raw Data'):
        st.subheader('Arun District Review Data')
        st.write(data)

# Location of Users and time of review on a map
st.sidebar.subheader("Visitor Locations")
st.sidebar.markdown("Location of Visitors posting reviews for each category")
select_users = st.sidebar.selectbox('Visualisation Type',['Map','Pie chart'], key = "loc")
cat = st.sidebar.selectbox("Choose Category of Review",("Accommodation","Food","Attractions"),key = "cat_Loc")
modified_data = data[data.category == cat]
review_locs = modified_data["location"].value_counts()
review_locs = pd.DataFrame({"Visitor Location":review_locs.index,"Number of Reviews":review_locs.values})
if select_users == "Map":
    st.subheader("Map of Visitor Location")
    st.markdown("Zoom in for detail")
    st.markdown(cat)
    st.map(modified_data)
else:
    fig =plotly.pie(review_locs, values = "Number of Reviews", names = "Visitor Location")
    st.subheader("Visitor Location")
    st.markdown(cat)
    st.plotly_chart(fig)

## How many reviews posted by month
st.sidebar.subheader("Number of Reviews Posted by Month")
if st.sidebar.checkbox("Show",True, key = "r"):
    st.subheader("Reviews posted by Month")
    test_series = pd.DataFrame(data["post month"].value_counts().reset_index())
    test_series.columns = ["Month","Reviews"]
    fig = plotly.bar(test_series, x = "Month", y = "Reviews")
    st.plotly_chart(fig)

# Overall sentiment for Arun District By category
st.sidebar.subheader("User Sentiment by Category for Arun District")
select_chart = st.sidebar.selectbox('Visualisation Type',['Histogram','Pie chart'], key = "sel_char")
cat_sentiment = st.sidebar.selectbox("Choose Category of Review",("Accommodation","Food","Attractions"),key = "cat_sent")
sent_count = data[data["category"] == cat_sentiment]["sentiment"].value_counts()
sent_count = pd.DataFrame({"Sentiment":sent_count.index,"Number of Reviews":sent_count.values})
if st.sidebar.checkbox("Show",True, key = "s"):
    if select_chart == "Histogram":
        st.subheader("Sentiment By Category for Arun District")
        st.subheader("%s" % (cat_sentiment))
        fig = plotly.bar(sent_count, x = "Sentiment", y = "Number of Reviews", color = "Number of Reviews", height = 500)
        st.plotly_chart(fig)
    else:
        fig =plotly.pie(sent_count, values = "Number of Reviews", names = "Sentiment")
        st.subheader("Sentiment By Category for Arun District")
        st.subheader("%s" % (cat_sentiment))
        st.plotly_chart(fig)

# Most highly reviewed establishments
st.header("Most Reviewed Establishments By Town, Category and Sentiment")
st.sidebar.subheader("Most Reviewed Establishments By Town,Category and Sentiment")
cat_choices = st.sidebar.selectbox("Choose Category", ("Accommodation","Food","Attractions"),key = "cats")
town_choices = st.sidebar.selectbox("Choose Town", ("Arundel","Bognor","Littlehampton"), key = "towns")
sentiment = st.sidebar.selectbox("Choose Sentiment",("positive","negative","neutral"),key = "sent_choice")
name_counts =data[(data["town"] == town_choices) & (data["category"] == cat_choices) &(data["sentiment"]==sentiment)]["name"].value_counts()
name_counts = pd.DataFrame({"Name":name_counts.index,"Number of Reviews":name_counts.values})
if st.sidebar.checkbox("Show",True, key = "m"):
    st.subheader("Highest Number of Reviews By Town,Category,Establishment and Sentiment")
    st.subheader("%s %s %s" % (town_choices,cat_choices,sentiment))
    fig = plotly.bar(name_counts, x="Name", y="Number of Reviews",color = "Number of Reviews", width = 800, height = 500)
    st.plotly_chart(fig)


# Detail Analysis Sentiment By Town, Category and estblishment type
st.sidebar.header("Detail Analysis: Sentiment By Category, Town and Establishment Type")
if st.sidebar.checkbox("Show", True, key = "tce"):
    select = st.sidebar.selectbox('Visualisation Type',['Histogram','Pie chart'], key = "sel")
    town_choice = st.sidebar.selectbox("Choose a Town",("Arundel","Bognor","Littlehampton"), key = "tc")
    category_choice = st.sidebar.selectbox("Please Choose a Category",('Accommodation','Food','Attractions'),key = "cc")
    if category_choice == "Accommodation":
        type_choice = st.sidebar.multiselect("Please Choose at least one Establishment Type",['Hotel', 'B&B/Inn', 'AccomOther'])
    elif category_choice == "Food":
        type_choice = st.sidebar.multiselect("Please Choose at least one Establishment Type",['Restaurant', 'Pub/Bar', 'Café', 'Steakhouse/Diner',
           'Fast Food/Takeaway', 'Gastropub'])
    else:
        type_choice = st.sidebar.multiselect("Choose at least one Establishment Type",['Historical/Culture', 'Nature/Gardens', 'Amusements/Fun', 'Nightlife',
           'Beach/Outdoor', 'Shopping', 'Spas/Leisure Centres','Classes/Workshops'])

    st.header('Sentiment By Category, Town and Establishment Type')
    st.subheader("%s %s %s" % (town_choice, category_choice, type_choice))
    sentiment_counts = data[(data["town"] == town_choice) & (data["category"] == category_choice) & (data["type"].isin(type_choice))]["sentiment"].value_counts()
    sentiment_counts = pd.DataFrame({"Sentiment":sentiment_counts.index,"Number of Reviews":sentiment_counts.values})
    if select == "Histogram":
        fig = plotly.bar(sentiment_counts, x = "Sentiment", y = "Number of Reviews", color = "Number of Reviews", height = 500)
        st.plotly_chart(fig)
    else:
        fig = plotly.pie(sentiment_counts, values = "Number of Reviews", names = "Sentiment")
        st.plotly_chart(fig)


# WordClouds for Postive and Negative sentiment_count
st.sidebar.header("Word Cloud")
fig, ax = plt.subplots()
if st.sidebar.checkbox("Show", True, key='7'):
    town_choice2 = st.sidebar.selectbox("Choose a Town",("Arundel","Bognor","Littlehampton"), key = "WC")
    category_choice2 = st.sidebar.selectbox("Choose a Category",('Accommodation','Food','Attractions'))
    word_sentiment = st.sidebar.radio('Choose sentiment for Word Cloud', ('positive', 'neutral', 'negative'))
    word_cloud = data[(data['category']== category_choice2) & (data["town"] == town_choice2) & (data['sentiment'] == word_sentiment)]
    st.subheader('Word cloud for %s sentiment' % (word_sentiment))
    st.subheader("%s %s" % (town_choice2, category_choice2))
    words = " ".join(title for title in word_cloud.title)
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=640).generate(words)
    plt.imshow(wordcloud)
    plt.xticks([])
    plt.yticks([])
    st.pyplot(fig)

# Allow random review to be shown
st.sidebar.subheader("Show Random Reviews for Chosen Categories and Town")
random = word_cloud[["title","review","post month"]].sample(1)
st.sidebar.markdown(random.iat[0,0])
if st.sidebar.checkbox("Show Full Review", True, key = "8"):
    st.sidebar.markdown(random.iat[0,1])
    st.sidebar.markdown("Posted in Month:")
    st.sidebar.markdown(random.iat[0,2])
