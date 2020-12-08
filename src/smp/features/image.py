from smp.features.features import Feature, Base
import pandas as pd
from sklearn.pipeline import Pipeline
import numpy as np
from PIL import Image
from os import listdir
from sklearn.decomposition import PCA
from pathlib import Path
from smp import data_dir
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import FeatureAgglomeration, KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


class ImageToArray(Base):
    def __init__(self, var_name, offset=2, n_components=175, n_clusters=3):
        self.var_name = var_name

        self.train_dir = data_dir / "raw" / "train_profile_images" / "profile_images_train"
        self.train_files = listdir(self.train_dir)
        self.test_dir = data_dir / "raw" / "test_profile_images" / "profile_images_test"
        self.test_files = listdir(self.test_dir)

        self.offset = offset
        self.pca = PCA(n_components=n_components, whiten=False)
        self.cluster = KMeans(n_clusters=n_clusters)
        self.enc = OneHotEncoder()
        self.scaler = StandardScaler()

    def fit(self, X: pd.Series, y=None):
        im = np.array([self.im_to_array(x) for x in X])
        im = self.scaler.fit_transform(im)
        im = self.pca.fit_transform(im)
        self.cluster.fit(im)
        self.enc.fit(self.cluster.predict(im).reshape(-1, 1))

    def transform(self, X: pd.Series):
        im = np.array([self.im_to_array(x) for x in X])
        im = self.scaler.transform(im)
        pca_fit = self.pca.transform(im)

        # return pd.DataFrame(
        #     pca_fit,
        #     index=X.index,
        #     columns=[self.var_name + " {}".format(i) for i in range(0, pca_fit.shape[1])]
        # )

        labels = self.enc.transform(self.cluster.predict(pca_fit).reshape(-1, 1))
        return pd.DataFrame(
            labels.toarray(),
            index=X.index,
            columns=[self.var_name + " {}".format(i) for i in range(0, labels.shape[1])]
        )

    @property
    def description(self):
        return "Convert image"

    def im_to_array(self, im_id):
        if im_id in self.train_files:
            file = self.train_dir / im_id
        else:
            file = self.test_dir / im_id

        im = Image.open(file)
        return np.array(
            im.crop((self.offset, self.offset, im.width - self.offset, im.height - self.offset))
            .convert('RGB')
        ).flatten() / 256


class ImagePreprocess(Feature):
    def __init__(self, var_name, offset, n_components):
        super().__init__(var_name)
        self._pipe = Pipeline(
            [
                ImageToArray(var_name, offset, n_components).to_step(),
            ],
            verbose=True
        )


class ProfileImage(ImagePreprocess):
    def __init__(self, offset=10, n_components=10):
        super().__init__("Profile Image", offset, n_components)


if __name__ == "__main__":
    from smp.features.features import Loader, Dataset

    loader = Loader()
    pipe = Pipeline(
        [
            Dataset(
                [
                    ProfileImage(offset=10, n_components=10),
                ]
            ).to_step()
        ],
        verbose=True,
    )

    train_data = pipe.fit_transform(loader.train)
    test_data = pipe.transform(loader.test)

    print(train_data.head())
    assert loader.train.shape[0] == train_data.shape[0]
    assert loader.train.shape[0] == train_data.dropna().shape[0]
    assert loader.test.shape[0] == test_data.shape[0]
    assert loader.test.shape[0] == test_data.dropna().shape[0]
