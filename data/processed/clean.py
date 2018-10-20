import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#import data into dataframe
df = pd.read_csv('WGS_abstracts.txt', sep = '\t', names = ['Project ID','Title','Abstract','Runs'])
df.set_index('Project ID', inplace=True)

#address missing values
df = df.replace(r'^\s+$', 'None', regex=True)
df = df.replace(r'none provided', 'None', regex=True)
df = df.replace(r'â€˜', '', regex=True)
df = df.replace(r'â€™', '', regex=True)
df = df.replace(r'\.(?!\d)', '', regex=True)
df = df.replace(r',', '', regex=True)
df = df.replace(r'\(', '', regex=True)
df = df.replace(r'\)', '', regex=True)
df = df.replace(np.nan, '')

#extract abstracts into series
abstracts = df['Abstract']
abstracts.size

#mapreduce abstracts ID-IDF
cvec = CountVectorizer(stop_words='english', min_df=1, max_df=.25, ngram_range=(1,2))

#calculate all n grams found in all documents
from itertools import islice
cvec.fit(abstracts)
list(islice(cvec.vocabulary_.items(), 20))

#transform into bag of words
cvec_counts = cvec.transform(abstracts)
cvec_counts.shape
print(cvec_counts)
