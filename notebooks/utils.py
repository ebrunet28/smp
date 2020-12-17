import matplotlib.pyplot as plt

from smp.features.boolean import PersonalURL
from smp.features.categorical import ProfileVerificationStatus
from smp.features.discrete import NumOfStatusUpdates, CustomFeature


def plot_target(y):
    plt.hist(y, bins=50)
    plt.title("Num of Profile Likes Distribution")
    plt.xlabel("Log")
    plt.ylabel("Count")


def plot_feature(feature, X, y=None):

    if isinstance(feature, PersonalURL):
        plt.hist(y[X.reshape(-1)], bins=100, alpha=0.5, density=True, label="missing")
        plt.hist(y[~X.reshape(-1)], bins=100, alpha=0.5, density=True, label="not missing")
        plt.title(feature.var_name)
        plt.xlabel("Num of Profile Likes (Log)")
        plt.ylabel("Density")
        plt.legend()

    if isinstance(feature, ProfileVerificationStatus):
        plt.hist(y[X == 'Verified'], bins=100, alpha=0.5, density=True, label="Verified")
        plt.hist(y[X == 'Not verified'], bins=100, alpha=0.5, density=True, label="Not verified")
        plt.hist(y[X == 'Pending'], bins=100, alpha=0.5, density=True, label="Pending")
        plt.title(feature.var_name)
        plt.xlabel("Num of Profile Likes (Log)")
        plt.ylabel("Density")
        plt.legend()

    if isinstance(feature, NumOfStatusUpdates):
        if y is None:
            plt.hist(X, bins=50, )
            plt.title(feature.var_name)
            plt.xlabel("Log")
            plt.ylabel("Count")
        else:
            plt.scatter(X, y)
            plt.title(feature.var_name)
            plt.xlabel("Num of Status Updates (Log)")
            plt.ylabel("Num of Profile Likes (Log)")

    if isinstance(feature, CustomFeature):
        if y is None:
            plt.hist(X, bins=50, )
            plt.title("Custom Feature")
            plt.xlabel("Log")
            plt.ylabel("Count")
        else:
            plt.scatter(X, y)
            plt.title("Custom Feature")
            plt.xlabel("Custom Feature (Log)")
            plt.ylabel("Num of Profile Likes (Log)")

    else:
        pass
