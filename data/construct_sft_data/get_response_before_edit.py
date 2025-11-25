"""
Get response before editing for a query.
"""
import sys
sys.path.append('..')
from tqdm import tqdm
from utils.call_llm import call_llm
from utils.util import read_json, save_json


def get_response_before_edit(data_point, model_name):
    """
    Get response before editing for a data point.
    
    Args:
        data_point: Data point containing query
        model_name: Name of the LLM model
        
    Returns:
        Data point with response_before_edit added
    """
    response_before_edit = call_llm(message=data_point['query'], llm_model_name=model_name)
    data_point['response_before_edit'] = response_before_edit
    return data_point


def main(read_data_path, save_data_path, model_name, start_id=None, end_id=None):
    """
    Main function to process data and get responses before editing.
    
    Args:
        read_data_path: Path to input data file
        save_data_path: Path to save output data file
        model_name: Name of the LLM model
        start_id: Start index for data slicing (optional)
        end_id: End index for data slicing (optional)
    """
    data = read_json(read_file_path=read_data_path)[start_id:end_id]
    new_data = []
    for i in tqdm(range(len(data))):
        new_data_point = get_response_before_edit(data_point=data[i], model_name=model_name)
        new_data.append(new_data_point)
        if len(new_data) % 10 == 0:
            save_json(save_file_path=save_data_path, data=new_data, mode='w')
    save_json(save_file_path=save_data_path, data=new_data, mode='w')


