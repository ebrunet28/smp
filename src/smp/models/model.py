from sklearn.base import BaseEstimator


class Model(BaseEstimator):
    """
    This class should be used to load and invoke the serialized model and
    any other required model artifacts for pre/post-processing.
    """

    def __init__(self, estimator_classifier=None, estimator_regressor=None):
        self.estimator_classifier = estimator_classifier
        self.estimator_regressor = estimator_regressor

    def fit(self, X, y=None, **fit_params):
        self.estimator_classifier.fit(X, y=y==0, **fit_params)
        print(self.estimator_classifier.best_params_)
        self.estimator_regressor.fit(X, y=y, **fit_params)
        return self

    def predict(self, data):
        """
        Returns model predictions.
        """
        # Add any required pre/post-processing steps here.
        mask = self.estimator_classifier.predict(data)
        like = self.estimator_regressor.predict(data)
        like[mask] = 0
        print(like[mask].shape)
        return like
