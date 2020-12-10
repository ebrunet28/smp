from sklearn.base import BaseEstimator
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.base import clone


class Model(BaseEstimator):
    """
    This class should be used to load and invoke the serialized model and
    any other required model artifacts for pre/post-processing.
    """

    def __init__(
        self,
        estimator_regressor=None,
        n_clusters=None,
        linkage=None,
        n_neighbors=None,
        weight=None,
    ):
        self.linkage = linkage
        self.n_clusters = n_clusters
        self.estimator_regressor = estimator_regressor
        self.n_neighbors = n_neighbors
        self.weight = weight
        self._estimator_type = "regressor"

    def fit(self, X, y=None, **fit_params):
        self.estimator_regressors = clone(
            [self.estimator_regressor for i in range(self.n_clusters)]
        )
        for i, estimator in enumerate(self.estimator_regressors):
            estimator.random_state = i

        self.custers = AgglomerativeClustering(
            linkage=self.linkage, n_clusters=self.n_clusters
        )
        self.neighbors = KNeighborsClassifier(n_neighbors=self.n_neighbors, p=4)

        labels = self.custers.fit_predict(X)
        self.neighbors.fit(X, labels)
        self.max_clusters = []
        for i, estimator_regressor in enumerate(self.estimator_regressors):
            self.max_clusters.append(y[labels == i].max())
            weigths = np.ones(y.shape)
            weigths[labels == i] = self.weight
            estimator_regressor.fit(X, y=y, sample_weight=weigths, **fit_params)

        return self

    def predict(self, data):
        """
        Returns model predictions.
        """
        # Add any required pre/post-processing steps here.

        proba = self.neighbors.predict_proba(data)
        preds = []
        for estimator_regressor, _max in zip(
            self.estimator_regressors, self.max_clusters
        ):
            pred = estimator_regressor.predict(data).reshape(-1, 1)
            pred[pred > _max*1.15] = _max*1.15  # clip predictions that dont make sense
            pred[pred < 0] = 0  # clip predictions that dont make sense
            preds.append(pred)

        preds = np.concatenate(preds, axis=1)
        preds = np.diag(np.matmul(preds, proba.T))

        return preds
