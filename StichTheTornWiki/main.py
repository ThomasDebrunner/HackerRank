import re
import numpy as np
import sys


def read_data(file):
    n_articles = int(next(file))

    # Create containers to hold the sets
    set_a = []
    set_b = []

    # read in the sets
    for line in file:
        if line.startswith('*****'):
            break
        set_a.append(line)

    for line in file:
        set_b.append(line)

    return set_a, set_b


def prepare_word(word):
    return re.sub('[^A-Za-z]+', '', word)


def get_normalized_vectors(articles):
    n_articles = int(len(articles)/2)

    # build array with every word that occurs in every article
    words = []
    for article in articles:
        for word in article.split():
            # sanitize word
            word = prepare_word(word)
            # append word, if we have not seen it yet
            if word not in words:
                words.append(word)

    n_words = len(words)
    vectors = []

    # establish numpy vectors and fill them with the term frequencies
    for article in articles:
        vector = np.zeros(n_words)
        # count term frequency
        for word in article.split():
            # sanitize word
            word = prepare_word(word)
            vector[words.index(word)] += 1
        # add vector to our vectors
        vectors.append(vector)

    # Calculate the document frequencies for the words (df)
    df = np.zeros(n_words)
    for vector in vectors:
        df += (vector > 0).astype(int)

    # calculate the tf-idf values
    for vector in vectors:
        for i in np.nonzero(vector):
            vector[i] = (1 + np.log(vector[i])) * np.log10(2*n_articles / df[i])

    # normalize the vectors
    for i in range(len(vectors)):
        vectors[i] /= np.linalg.norm(vectors[i])

    return vectors


def main():
    # load data
    set_a, set_b = read_data(sys.stdin)
    # combine data sets, as we need to process them equally
    articles = set_a + set_b
    n_articles = int(len(articles)/2)

    vectors = get_normalized_vectors(articles)

    # calculate the cos() correlations
    corr = []
    for i in range(int(n_articles)):
        for j in range(int(n_articles), n_articles*2):
            corr.append(np.dot(vectors[i], vectors[j]))

    # determine results
    results = [-1]*int(n_articles)
    # while we still have something to assign
    for i in range(len(corr)):
        max_index = corr.index(max(corr))
        # mark the maximum as used
        corr[max_index] = -1
        part_a = int(max_index / n_articles)
        part_b = int(max_index % n_articles)
        # assign result, if we don't have a result for this yet. Continue otherwise
        if results[part_a] == -1 and part_b not in results:
            results[part_a] = part_b

    # return result
    for result in results:
        print(result+1)

if __name__ == '__main__':
    main()
