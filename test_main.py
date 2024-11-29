import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from sklearn.linear_model import LogisticRegression
from preprocess import Preprocessor
from clustering_model import ClusteringModel
from main import main, map_clusters_to_categories, save_models
import joblib

# Test main() with mock dependencies
def test_main():
    # Mock file path and data
    mock_file_path = r"C:\Users\YAKUMAR\Documents\Hackathon\bbc_data.csv"
    mock_data = pd.DataFrame({
        'text': ["sample text 1", "sample text 2", "sample text 3"],
        'labels': ["Sport", "Business", "Politics"]
    })

    # Mocking Preprocessor
    Preprocessor.clean_text = MagicMock(side_effect=lambda x: f"cleaned_{x}")
    Preprocessor.fit_transform = MagicMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    Preprocessor.transform = MagicMock(return_value=[[0.1, 0.2, 0.3]])
    
    # Mocking ClusteringModel
    ClusteringModel.fit = MagicMock()
    ClusteringModel.predict = MagicMock(return_value=[0, 1, 2])
    
    # Mock LogisticRegression
    LogisticRegression.fit = MagicMock()
    LogisticRegression.predict = MagicMock(return_value=[0])
    
    # Mock joblib.dump
    with patch('joblib.dump') as mock_dump:
        # Mock user input
        with patch('builtins.input', return_value="Test text to classify"):
            # Mock pandas read_csv
            with patch('pandas.read_csv', return_value=mock_data):
                # Run the main function
                main()

                # Assertions
                Preprocessor.clean_text.assert_called()
                Preprocessor.fit_transform.assert_called_once()
                ClusteringModel.fit.assert_called_once()
                LogisticRegression.fit.assert_called_once()
                mock_dump.assert_called()

