from smp.features.features import Feature, Base
import pandas as pd
from sklearn.pipeline import Pipeline
import numpy as np
from PIL import Image
from os import listdir
from sklearn.decomposition import PCA
from pathlib import Path
from smp import data_dir


class ImageToArray(Base):
    def __init__(self, var_name, offset=10, n_components=10):
        self.var_name = var_name

        self.train_dir = data_dir / "train_profile_images" / "profile_images_train"
        self.train_files = listdir(self.train_dir)
        self.test_dir = data_dir / "test_profile_images" / "profile_images_test"
        self.test_files = listdir(self.test_dir)

        self.offset = offset
        self.pca = PCA(n_components=n_components)

    def fit(self, X: pd.Series, y=None):
        im = np.array([self.im_to_array(x) for x in X])
        self.pca.fit(im)

    def transform(self, X: pd.Series):
        im = np.array([self.im_to_array(x) for x in X])
        pca_fit = self.pca.transform(im)
        return pd.DataFrame(
            pca_fit,
            index=X.index,
            columns=[self.var_name + " {}".format(i) for i in range(0, pca_fit.shape[1])]
        )

    @property
    def description(self):
        return "Convert image"

    def im_to_array(self, im_id):
        if im_id in self.train_files:
            file = self.train_dir / im_id
        else:
            file = self.test_dir / im_id

        return np.array(
            Image.open(file)
            .crop((self.offset, self.offset, 32 - self.offset, 32 - self.offset))
            .convert('L')
        ).flatten()


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
