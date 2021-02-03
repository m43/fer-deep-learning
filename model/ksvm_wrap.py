from sklearn.svm import SVC


class KSVMWrap:
    def __init__(self, X, Y_, c=1, gamma="auto", decision_function_shape="ovo"):
        self.clf = SVC(kernel='rbf', decision_function_shape=decision_function_shape, C=c, gamma=gamma,
                       probability=True)
        self.c = c
        self.gamma = gamma
        self.clf.fit(X, Y_)
        self.support = self.clf.support_vectors_
        self.support_indices = self.clf.support_

    def predict(self, X):
        return self.clf.predict(X)

    def get_scores(self, X):
        return self.clf.predict_proba(X)
