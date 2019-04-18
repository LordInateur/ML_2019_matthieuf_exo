import glob
import errno

import math
from sklearn.feature_extraction.text import TfidfVectorizer

"""
    Le but est de trouver un article qui "ressemble"
    affiche pour un seul test pour l'instant
"""
if __name__ == "__main__":
    path = './data/SimpleText_train/*.txt'
    documents = []
    documents_names = []

    files = glob.glob(path)
    for name in files:
        try:
            with open(name, "r", encoding="utf8") as f:
                documents.append(f.read())
                documents_names.append(name)
        except IOError as exc:
            if exc.errno != errno.EISDIR:
                raise


    path_test = './data/SimpleText_test/*.txt'
    documents_test = []
    documents_names_test = []

    files = glob.glob(path_test)
    for name in files:
        try:
            with open(name, "r", encoding="utf8") as f:
                documents_test.append(f.read())
                documents_names_test.append(name)
        except IOError as exc:
            if exc.errno != errno.EISDIR:
                raise




    print("nb_document : ", len(documents))

    vectorizer = TfidfVectorizer()
    sklearn_representation = vectorizer.fit_transform(documents)
    # print("nb_mot_dic:", len(vectorizer.get_feature_names()))
    print(vectorizer.get_feature_names())

    print(sklearn_representation.shape)


    def cosine_similarity(vector1, vector2):
        dot_product = sum(p*q for p,q in zip(vector1, vector2))
        magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
        if not magnitude:
            return 0
        return dot_product/magnitude

    # sklearn_representation.transform(documents_test)

    skl_tfidf_comparisons = []
    for count_0, doc_0 in enumerate(sklearn_representation.toarray()):
        # for count_1, doc_1 in enumerate(sklearn_representation.toarray()):
        # boucler pour comparer tout les documents entre eux
        count_1, doc_1 = 0, sklearn_representation.toarray()[0]
        skl_tfidf_comparisons.append((cosine_similarity(doc_0, doc_1), count_0, count_1))


    for x in sorted(skl_tfidf_comparisons, reverse = True):
        print(documents_names[x[1]], ' -  ', documents_names[x[2]], ' : ', x[0] )




