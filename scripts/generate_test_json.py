"""
In this script we generate json files for our test data to evaluate at Nuwe.
We will receive as input the test data and the model we want to use.
We will return a json file with the predictions.
"""

import os
import pandas as pd
import sys
import torch
import json

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_data(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data to be used in the model.
    """
    # One-hot encoding
    data_frame = pd.get_dummies(
        data_frame,
        columns=data_frame.columns[data_frame.dtypes == 'object'],
    )

    # Normalize the data (except binary columns)
    binary_columns = data_frame.columns[data_frame.nunique() == 2]
    binary_cols_data = data_frame[binary_columns]
    data_frame = (data_frame - data_frame.mean()) / data_frame.std()
    data_frame[binary_columns] = binary_cols_data
    return data_frame

def main(model_name:str, dataset_name:str):
    # Load the model
    model_relative_path = f'../model/{model_name}_{dataset_name}.pt'
    model_path = os.path.join(os.path.dirname(__file__), model_relative_path)
    model = torch.jit.load(model_path)
    model.to(DEVICE)
    model.eval()

    # Load the test data
    if dataset_name == 'balanced':  # The balanced test data is the same as the cleaned one
        test_data_relative_path = f'../data/supply_chain_test_cleaned.csv'
    else:
        test_data_relative_path = f'../data/supply_chain_test_{dataset_name}.csv'
    test_data_path = os.path.join(os.path.dirname(__file__), test_data_relative_path)
    test_data_df = pd.read_csv(test_data_path, index_col=0)

    # Preprocess the test data (same as in model training)
    test_data_df = preprocess_data(test_data_df)
    test_data = torch.tensor(test_data_df.to_numpy())
    test_data = test_data.type(torch.float32) 
    test_data.reshape(-1, 1)

    # Get the binary predictions (test_data has no target column)
    print('Predicting test data...')
    test_data = test_data.to(DEVICE)
    predictions = model(test_data)
    predictions = torch.round(predictions)  # Round to get the binary predictions
    predictions = predictions.cpu().detach().numpy()
    predictions = predictions.astype(int).flatten().tolist()

    # Print the percentage of positive and negative predictions
    print(f'Positive predictions: {predictions.count(1)} ({predictions.count(1)/len(predictions)*100:.2f}%)')
    print(f'Negative predictions: {predictions.count(0)} ({predictions.count(0)/len(predictions)*100:.2f}%)')

    # Create the json file
    json_dict = {'target': {str(i): predictions[i] for i in range(len(predictions))}}
    json_filename = f'../data/{model_name}_{dataset_name}_predictions.json'
    json_path = os.path.join(os.path.dirname(__file__), json_filename)

    # Save the json file
    with open(json_path, 'w') as f:
        json.dump(json_dict, f)

if __name__ == '__main__':
    model_name = sys.argv[1]
    dataset_name = sys.argv[2]
    main(model_name, dataset_name)