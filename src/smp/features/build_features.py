from smp.features.features import Loader, Preprocessor
from smp.features.rgb import RGB
from smp.features.float import Float


if __name__ == "__main__":

    loader = Loader()
    preprocessor = Preprocessor(loader)
    train_data, test_data = preprocessor.preprocess(
        [
            RGB("Profile Text Color"),
            RGB("Profile Page Color"),
            RGB("Profile Theme Color"),
            Float("UTC Offset"),
            Float("Num of Followers"),
            Float("Num of People Following"),
            Float("Num of Status Updates"),
            Float("Num of Direct Messages"),
            Float("Avg Daily Profile Visit Duration in seconds"),
            Float("Avg Daily Profile Clicks"),
            # Float("Num of Profile Likes"),
        ]
    )

    print(train_data.head(10))
    print(train_data.shape)