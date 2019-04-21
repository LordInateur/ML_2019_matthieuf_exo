import datetime

from sklearn import svm

def svm_classifier(x_train, y_train, x_test, shape='ovo'):
    start_time = datetime.datetime.now()
    clf = svm.SVC(gamma='scale', decision_function_shape=shape)
    clf.fit(x_train, y_train)
    cl_construct_time = datetime.datetime.now() - start_time
    return clf.predict(x_test), cl_construct_time.microseconds, (datetime.datetime.now() - start_time - cl_construct_time).microseconds
