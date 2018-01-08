# Simple Natual Language Processing
# Deep Learning with our own Data
# URL: https://pythonprogramming.net/using-our-own-data-tensorflow-deep-learning-tutorial/?completed=/tensorflow-neural-network-session-machine-learning-tutorial/
# Youtube Video: https://www.youtube.com/watch?v=7fcWfUavO7E&index=48&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v
# Use a neural network to correctly identify sentiment, training with this data.

# The plan and theory for applying our deep neural network to some sentiment training data, and now we're going to be
# working on the pre-processing script for that.
# This code will take our string sample data and convert it to vectors.

# Prerequisite:
# run Python:
# > import nltk
# > nltk.download()
# installed nltk_data in /usr/local/share

import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import random  # shuffle the data
import pickle  # save the process so that we dont need to do it every time
from collections import Counter # Counter will be used for sorting most common lemmas
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
# We define the lemmatizer, and then we set the hm_lines value. 100,000 will do all of the lines, there are just
# over 10,000 lines. If you want to test something new, or shrink the total data size for a smaller computer/processor,
# you can set a smaller number here
hm_lines = 100000

# Data files are downloaded from:
# Positive data: https://pythonprogramming.net/static/downloads/machine-learning-data/pos.txt
# Negative data: https://pythonprogramming.net/static/downloads/machine-learning-data/neg.txt
pos_file = 'data/pos.txt'
neg_file = 'data/neg.txt'
pickle_file = 'data/sentiment_set.pickle'

def create_lexicon(pos,neg):
    # This function which takes a path to the positive file and the negative file. From here, we open the files,
    # read the lines, tokenize the words, and add them to the lexicon.
    lexicon = []
    with open(pos, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            all_words = word_tokenize(l)
            lexicon += list(all_words)

    with open(neg, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            all_words = word_tokenize(l)
            lexicon += list(all_words)

    # At this point, our lexicon is just a list of every word in our training data. If you had a huge dataset,
    # too large to fit into your memory, then you'd need to adjust the hm_lines value here, to just go through
    # the first hm_lines number of lines per file.

    # Now we still need to lemmatize and remove duplicates. We also don't really need super common words,
    # nor very uncommon words. For example, words like "a", "and", or "or" aren't going to give us much value
    # in this simple "bag of words" model, so we don't want them. Uncommon words aren't going to be very useful either,
    # since they'd likely be so rare that their very presence would skew the results. We can try to play with this
    # to see if we're correct in this belief.

    # We lemmatize, then count the word occurance. If the word occurs less than 1,000 times, but more than 50 times,
    # we want to include it in our lexicon. These two values are definitely something you may want to tweak,
    # and really ought to be some sort of % of the entire dataset.
    # None of this code is optimized or meant to be used in production. This is just conceptual code,
    # with tons of room for improvement.
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    l2 = []
    for w in w_counts:
        #print(w_counts[w])
        if 1000 > w_counts[w] > 50:
            l2.append(w)
    print('[create_lexicon] Result =',len(l2))
    return l2

def sample_handling(sample,lexicon,classification):
    # Now we can take this lexicon, and use it as our bag of words that we will look for in a string.
    # Each time we find a lemma in our lexicon that exists in the lemmatized and word tokenized sample sentence,
    # the index of that lemma in the lexicon is turned "on" in our previously numpy zeros array that is the same
    # length as the lexicon.

    # Now we begin to iterate through the lemmatized words, adding 1 to the index value in the features array that
    # is the same index of the word in the lexicon. From here, we apply this to our total featureset.
    # When done, we return that whole thing. This function will be run twice; once for the positives and once for
    # the negatives.
    featureset = []
    with open(sample,'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            features = list(features)
            featureset.append([features,classification])

    return featureset


def create_feature_sets_and_labels(pos,neg,test_size = 0.1):
    # The create_feature_sets_and_labels function is where everything comes together.
    # We create the lexicon here based on the raw sample data that we have, then we build the full features
    # based on their associated files, the lexicon, and then the classifications.
    lexicon = create_lexicon(pos,neg)
    features = []
    features += sample_handling(pos_file, lexicon, [1, 0])
    features += sample_handling(neg_file,lexicon,[0,1])
    random.shuffle(features)
    features = np.array(features)

    testing_size = int(test_size*len(features))

    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])

    return train_x,train_y,test_x,test_y


if __name__ == '__main__':
    train_x,train_y,test_x,test_y = create_feature_sets_and_labels(pos_file, neg_file)
    # if you want to pickle this data:
    with open(pickle_file,'wb') as f:
        pickle.dump([train_x,train_y,test_x,test_y],f)