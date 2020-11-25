import unittest

from main import predict


class BaseTestCase(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    @staticmethod
    def test_predict():
        df = predict()
        assert len(df) == 2500
