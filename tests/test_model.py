#!/usr/bin/env python
import pytest
from smp.models.model import Model

def test_predict():
    model = Model()
    model.predict([[1,2],[3,4]])
