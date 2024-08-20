#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

file_path = 'training.1600000.processed.noemoticon.csv'
df = pd.read_csv(file_path, encoding='latin-1', header=None)


# In[2]:


df.head(), df.info()


# In[3]:


df.isnull().sum()


# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt

# Rename columns for clarity
df.columns = ['Sentiment', 'Tweet_ID', 'Date', 'Query', 'User', 'Tweet']

# Plot the sentiment distribution
sns.countplot(x=df['Sentiment'], palette='coolwarm')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment (0 = Negative, 4 = Positive)')
plt.ylabel('Number of Tweets')
plt.show()


# In[5]:


import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stop words
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['Cleaned_Tweet'] = df['Tweet'].apply(preprocess_text)
df['Cleaned_Tweet'].head()


# In[6]:


from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def tokenize_and_lemmatize(text):
    tokens = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(token) for token in tokens])

df['Lemmatized_Tweet'] = df['Cleaned_Tweet'].apply(tokenize_and_lemmatize)
df['Lemmatized_Tweet'].head()


# In[ ]:


from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def tokenize_and_lemmatize(text):
    tokens = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(token) for token in tokens])

df['Lemmatized_Tweet'] = df['Cleaned_Tweet'].apply(tokenize_and_lemmatize)
df['Lemmatized_Tweet'].head()


# In[ ]:


from sklearn.model_selection import train_test_split

X = df['Lemmatized_Tweet']
y = df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_vect, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test_vect)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')


# In[ ]:


plt.show()


# In[ ]:




