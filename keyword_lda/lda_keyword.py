#sg
#This script runs LDA on a given set of sentences with the hope that it will help in uncovering keywords to be used with topics.

def readInList(file_list):
    res = []
    for file_name in file_list:
        for line in open(file_name, 'r'):
            res.append(line.rstrip('\n'))
    return res


'''
Creates a document term matrix with countvectorizer from sklearn
'''
def createDTMat(fileList):
    from sklearn.feature_extraction.text import CountVectorizer
    cvec = CountVectorizer(stop_words = 'english')
    lines_list = readInList(fileList)
    X = cvec.fit_transform(lines_list).toarray()
    vocab = cvec.get_feature_names()
    return (X, vocab)


'''
Takes the document term matrix and runs LDA
'''
def run_lda(docTermMat, num_topics = 20, iter = 500):
    from lda import lda
    model = lda.LDA(n_topics = num_topics, n_iter = iter, random_state = 1)
    model.fit(docTermMat)
    topic_word_distributions = model.topic_word_ 
    return topic_word_distributions


def print_results(topic_word_distributions, vocab, topk):
    import numpy as np
    for i, topic_dist in enumerate(topic_word_distributions):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-topk:-1]
        print topic_words



if __name__ == '__main__':
    import sys
    if(len(sys.argv) < 4):
        print 'Usage lda_keyword.py num_topics num_iterations file1 file2 ... fileN'
        sys.exit(0)

    #step 1: create document term matrix
    (X, vocab) = createDTMat(sys.argv[3:])

    #step 2: run lda
    topic_word_distributions = run_lda(X, int(sys.argv[1]), int(sys.argv[2]))

    #step 3: print results from running LDA
    print_results(topic_word_distributions, vocab, 10);
    

    

