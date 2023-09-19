# https://medium.com/@nisha.imagines/nlp-with-python-text-clustering-based-on-content-similarity-cae4ecffba3c

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import cosine_similarity



if __name__ == '__main__':
    data = [
        'login to URL and navigate to Contacts',
        'login to URL and navigate to About US',
        'login to URL and navigate to History page',
        'Provide incorrect password and validate failure',
        'incorrect id and see if it fails',
        'Application crashed',
        'Application availability is low',
        'Latency is poor',
        'incorrect URL should fail',
        'login to URL and click shopping cart'
    ]

    df = pd.DataFrame(data, columns=['Scenarios'])

    # Convert sentences into vectors using TFIDF - Start
    vec = TfidfVectorizer(stop_words="english", ngram_range = (1, 3))
    vec.fit(df.Scenarios.values)
    features = vec.transform(df.Scenarios.values)
    # Convert sentences into vectors using TFIDF - End

    # Make K-Means - Start
    clust = KMeans(init='k-means++', n_clusters=2, n_init=10)
    clust.fit(features)
    yhat = clust.predict(features)
    df['Cluster Labels'] = clust.labels_

    print(df)
    # Make K-Means - End

    df_1 = df.loc[df['Cluster Labels'] == 1]
    df_2 = df.loc[df['Cluster Labels'] == 0]

    # count similarity - Start
    cosine_sim = cosine_similarity(features)

    print('看看features')
    print(features)


    print('看看相似度')
    print(len(cosine_sim))
    print(cosine_sim[0])


    def getind(c):
        return df[df.Scenarios==c].index.tolist()

    def getscene(i):
        return df[df.index==i].Scenarios.tolist()

    similar = list(enumerate(cosine_sim[0]))
    # print(similar)
    # count similarity - End





    
