"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp


class Prompter(object):
    def __init__(self, template_name: str = ""):
        if 'editor' in template_name:
            self.prompt_mode='editor'
        elif 'serac' in template_name:
            self.prompt_mode='serac'
        elif 'mquake' in template_name:
            self.prompt_mode='mquake'
        else:
            raise 
        self.template_name = template_name
        file_name = osp.join("data", "template", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
            print(f"Using prompt template {template_name}: {self.template['description']}")

    def generate_train_prompt(self, data_point):
        res = self.template["prompt_input"].format(
            requested_rewrite=data_point['requested_rewrite'],
            target_old=data_point['target_old'],
            target_new=data_point['target_new'],
            query=data_point['query'],
            response_before_edit=data_point['response_before_edit']
        )
        return res

    def generate_prompt(self, edit_data_point, query_data_point):
        # Generate input prompt for training or inference
        return getattr(self, f"generate_{self.prompt_mode}_prompt")(edit_data_point, query_data_point)
    
    def generate_prompt_target(self, query_data_point):
        # Generate target for training (only called during training)
        return getattr(self, f"generate_{self.prompt_mode}_prompt_target")(query_data_point)       

    def generate_editor_prompt(self, edit_data_point, query_data_point) -> str:
        res = self.template["prompt_input"].format(
            requested_rewrite=edit_data_point['requested_rewrite'],
            target_old=edit_data_point['target_old'],
            target_new=edit_data_point['target_new'],
            query=query_data_point['query'],
            response_before_edit=query_data_point['response_before_edit']
        )
        return res
    
    def generate_serac_prompt(self, edit_data_point, query_data_point) -> str:
        res = self.template["prompt_input"].format(
            requested_rewrite=edit_data_point['requested_rewrite'],
            target_old=edit_data_point['target_old'],
            target_new=edit_data_point['target_new'],
            query=query_data_point['query'],
        )
        return res
    
    def generate_mquake_prompt(self, edit_data_point, query_data_point) -> str:
        edits = query_data_point["edits"].copy()
        edits_str = ""
        for i in range(len(edits)):
            edit = edits[i]
            edits_str += f"{i+1}. {edit['requested_rewrite']} '{edit['target_old']}' (old) -> '{edit['target_new']}' (new)\n"
        res = self.template["prompt_input"].format(
            editing_facts=edits_str,
            query=query_data_point['query'],
            response_before_edit=query_data_point['response_before_edit']
        )
        return res

    def generate_editor_prompt_target(self, query_data_point) -> str:
        if query_data_point['query_type'] in ['naive', 'paraphrase']:
            return query_data_point['gpt4_response_after_edit']
        elif query_data_point['query_type'] == 'neighbor':
            return "retain"
        else:
            raise ValueError(f"Unknown query_type: {query_data_point['query_type']}")

    def generate_serac_prompt_target(self, query_data_point) -> str:
        return query_data_point['gpt4_response_after_edit']


    def get_response(self, output, query_data_point):
        if len(output.split(self.template["response_split"])) > 1:
            # Remove input prefix from output during inference
            output = output.split(self.template["response_split"])[1].strip().replace('</s>', '')
            if 'retain' in output:
                # Output "retain" means no editing
                return query_data_point['response_before_edit']
            else:
                return output.replace('</s>', '')
        else:
            return output.strip().replace('</s>', '')


