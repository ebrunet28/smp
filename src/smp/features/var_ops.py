from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from smp.features.features import Feature, Base, ToLog, ToVector
import numpy as np
# from smp.features.float import AvgDailyProfileClicks, AvgDailyProfileVisitDuration
from sklearn.pipeline import Pipeline
from smp.features.features import Loader, Dataset
from sklearn.preprocessing import StandardScaler


class ToSqrt(Base):
    def transform(self, X):
        return np.sqrt(X)

    @property
    def description(self):
        return "Convert to sqrt(x)"

class DoProduct(Base):
    def transform(self, X):
        return X.prod(axis=1)[:, None]

    @property
    def description(self):
        return "Product of cols"


class DoDivide(Base):
    def transform(self, X):
        return X[:, 0].reshape(-1, 1) / X[:, 1].reshape(-1, 1)

    @property
    def description(self):
        return "Divide vars"


class DoClip(Base):
    def transform(self, X):
        clipval = 3
        X[X>=clipval] = clipval
        return X

    @property
    def description(self):
        return "Clip car"

class DoSubstract(Base):
    def transform(self, X):
        return X[:, 0].reshape(-1, 1) - X[:, 1].reshape(-1, 1)

    @property
    def description(self):
        return "Substract vars"


# class Var1(Feature):
#     def __init__(self, var_name="Avg Daily Profile Visit Duration in seconds"):
#         super().__init__(var_name)
#         self._pipe = Pipeline(
#             [
#                 ToVector().to_step(),
#                 ("SimpleImputer", SimpleImputer(strategy="median"),),
#             ],
#             verbose=True,
#         )
#
#
# class Var2(Feature):
#     def __init__(self, var_name="Avg Daily Profile Clicks"):
#         super().__init__(var_name)
#         self._pipe = Pipeline(
#             [
#                 ToVector().to_step(),
#                 ("SimpleImputer", SimpleImputer(strategy="median"),),
#             ],
#             verbose=True,
#         )


class ToLog0(Base):
    def transform(self, X):
        return np.log(X)

    @property
    def description(self):
        return "Convert to log(x)"


class DoExp(Base):
    def transform(self, X):
        return np.exp(-X)

    @property
    def description(self):
        return "Convert to exp(-x)"


class Var1(Feature):
    def __init__(self, var_name="Num of Status Updates"):
        super().__init__(var_name)
        self._pipe = Pipeline(
            [
                ToVector().to_step(),
                ("SimpleImputer", SimpleImputer(strategy="median"),),
            ],
            verbose=True,
        )


class Var2(Feature):
    def __init__(self, var_name="Avg Daily Profile Visit Duration in seconds"):
        super().__init__(var_name)
        self._pipe = Pipeline(
            [
                ToVector().to_step(),
                ("SimpleImputer", SimpleImputer(strategy="median"),),
            ],
            verbose=True,
        )


class RatioClick(Feature):
    def __init__(self, var_name=["Num of Status Updates",  "Avg Daily Profile Visit Duration in seconds"]):
        super().__init__(var_name)
        self._pipe = Pipeline(
            [
                Dataset(
                    [
                        Var2(),
                        Var1(),
                    ]
                ).to_step(),
                DoDivide().to_step(),
                ToLog0().to_step(),
                # ToSqrt().to_step(),
                # ToLog0().to_step(),
                # DoExp().to_step(),
                # DoClip().to_step(),
                # ("Std Scaler", StandardScaler())
            ],
            verbose=True,
        )
#
# class RatioClick(Feature):
#     def __init__(self, var_name=["Avg Daily Profile Clicks",  "Avg Daily Profile Visit Duration in seconds"]):
#         super().__init__(var_name)
#         self._pipe = Pipeline(
#             [
#                 Dataset(
#                     [
#                         Var2(),
#                         Var1(),
#                     ]
#                 ).to_step(),
#                 DoDivide().to_step(),
#                 ToLog0().to_step(),
#                 # ToSqrt().to_step(),
#                 # ToLog0().to_step(),
#                 # DoExp().to_step(),
#                 # DoClip().to_step(),
#                 # ("Std Scaler", StandardScaler())
#             ],
#             verbose=True,
#         )


class Var3(Feature):
    def __init__(self, var_name='Num of Status Updates'):
        super().__init__(var_name)
        self._pipe = Pipeline(
            [
                ToVector().to_step(),
                ("SimpleImputer", SimpleImputer(strategy="mean"),),
                ToLog().to_step()
            ],
            verbose=True,
        )


class Var4(Feature):
    def __init__(self, var_name='Num of Followers'):
        super().__init__(var_name)
        self._pipe = Pipeline(
            [
                ToVector().to_step(),
                ("SimpleImputer", SimpleImputer(strategy="mean"),),
                ToLog().to_step()
            ],
            verbose=True,
        )


class Ratio1(Feature):
    def __init__(self, var_name=['Num of Status Updates', 'Num of Followers']):
        super().__init__(var_name)
        self._pipe = Pipeline(
            [
                Dataset(
                    [
                        Var3(),
                        Var4(),
                    ]
                ).to_step(),
                DoSubstract().to_step(),
                ("Std Scaler", StandardScaler())
            ],
            verbose=True,
        )

