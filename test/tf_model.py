import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import keras
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

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

#word count occurences and normalize
cvec = TfidfVectorizer(stop_words='english', min_df=0.0020, max_df=.5, ngram_range=(1,2)i, norm=’l2’, use_idf=False)

#learn vocabulary dictionary
from itertools import islice
cvec.fit(abstracts)
list(islice(cvec.vocabulary_.items(), 20))

#transform into bag of words
cvec_counts = cvec.transform(abstracts)

#data
x_train = transformed_weights

#PCA
mu = x_train.mean(axis=0)
U,s,V = np.linalg.svd(x_train - mu, full_matrices=False)
Zpca = np.dot(x_train - mu, V.transpose())

Rpca = np.dot(Zpca[:,:2], V[:2,:]) + mu    # reconstruction
err = np.sum(np.asarray(x_train-Rpca)**2)/Rpca.shape[0]/Rpca.shape[1]
print('PCA reconstruction error with 2 PCs: ' + str(round(err,3)));

#Autoencoder
m = Sequential()
m.add(Dense(500,  activation='sigmoid', input_shape=(x_train.shape[1],)))
m.add(Dense(250,  activation='sigmoid'))
m.add(Dense(2,    activation='linear', name="bottleneck"))
m.add(Dense(250,  activation='sigmoid'))
m.add(Dense(500,  activation='sigmoid'))
m.add(Dense(x_train.shape[1],  activation='sigmoid'))
m.compile(loss='categorical_crossentropy', optimizer = Adam())
history = m.fit(x_train, x_train, batch_size=128, epochs=100, verbose=1, validation_split=0.1)

encoder = Model(m.input, m.get_layer('bottleneck').output)
Zenc = encoder.predict(x_train)  # bottleneck representation
Renc = m.predict(x_train)        # reconstruction

#Plot
plt.subplot(121)
plt.title('PCA')
plt.scatter([Zpca[:1000,0]], [Zpca[:1000,1]])

plt.subplot(122)
plt.title('Autoencoder')
plt.scatter([Zenc[:1000,0]], [Zenc[:1000,1]])

plt.tight_layout()
plt.savefig("tf.pdf")

#Reconstruction
