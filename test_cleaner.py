import pytest
from cleaner import clean_data


def test_clean_data():
    result = clean_data("testdata.csv")
    assert "cleaned_data" in result
    assert result["quality_score"] <= 100


pytest.main(["-v"])  # pip install pytest ; pytest
