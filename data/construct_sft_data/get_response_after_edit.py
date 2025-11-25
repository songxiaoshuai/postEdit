"""
Get response after editing for a query.
"""
import sys
sys.path.append('..')
from tqdm import tqdm
from utils.call_llm import call_llm
from utils.util import read_json, save_json



def construct_edit_prompt(data_point):
    """
    Construct edit prompt from data point.
    
    Args:
        data_point: Data point containing edit information
        
    Returns:
        Formatted prompt string
    """
    requested_rewrite=data_point['requested_rewrite']
    target_old=data_point['target_old']
    target_new=data_point['target_new']
    query=data_point['query']
    response_before_edit=data_point['response_before_edit']
    # prompt=f"For the following query and original response, you first need to identify every span related to {target_old} in original response and then modify them according to New Fact. Finally, output the modified response. \n\nNew fact:\nThe ground truth of '{requested_rewrite}' has been updated from '{target_old}' to '{target_new}'.\n\nThe query:\n{query}\n\nOriginal response:\n{response_before_edit}\n\nEdited response:\n"
    # prompt=f"You will assume the role of an editor. For the following query and original response. Use the new fact to modify the content related to '{target_old}' in original reply and output modified response. \n\nNew fact:\nThe answer of '{requested_rewrite}' has been updated from '{target_old}' to '{target_new}'.\n\nThe query:\n{query}\n\nOriginal response:\n{response_before_edit}\n\nEdited response:\n"
    prompt=f"For the following query and original response, you need to follow in order:\nFirstly, locate all spans related to the **old fact:{requested_rewrite} {target_old}** in original reply;\nSecondly, modify these spans according to **new fact: {requested_rewrite} {target_new}**.\nThirdly, output the edited response based on the modified spans (Do not output other content).\n\nThe query:\n{query}\n\nOriginal response:\n{response_before_edit}\n\nEdited response:\n"
    return prompt


def get_response_after_edit(data_point, model_name):
    """
    Get response after editing for a data point.
    
    Args:
        data_point: Data point containing edit information
        model_name: Name of the LLM model
        
    Returns:
        Data point with chatgpt_response_after_edit added
    """
    prompt = construct_edit_prompt(data_point)
    response = call_llm(message=prompt, llm_model_name=model_name)
    data_point['chatgpt_response_after_edit'] = response
    return data_point


def main(read_data_path, save_file_path, model_name, start_id=None, end_id=None):
    """
    Main function to process data and get responses after editing.
    
    Args:
        read_data_path: Path to input data file
        save_file_path: Path to save output data file
        model_name: Name of the LLM model
        start_id: Start index for data slicing (optional)
        end_id: End index for data slicing (optional)
    """
    data = read_json(read_file_path=read_data_path)[start_id:end_id]
    new_data = []
    for i in tqdm(range(len(data))):
        data_point = data[i]
        if 'chatgpt_response_after_edit' in data_point and len(data_point['chatgpt_response_after_edit']) > 0:
            new_data.append(data_point)
        else:
            new_data_point = get_response_after_edit(data_point, model_name)
            new_data.append(new_data_point)
        if len(new_data) % 10 == 0:
            save_json(save_file_path=save_file_path, data=new_data, mode='w')
    save_json(save_file_path=save_file_path, data=new_data, mode='w')