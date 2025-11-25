"""
Edit Memory Module

Functionality:
- edit_memory: Stores edit_knowledge and can retrieve top_K similar edit_knowledge based on query
- memory_mode: single, batch, sequence
- Knowledge injection types: add, delete, update (currently only update is considered)
- Embedding model: all-MiniLM-L6-v2
"""
import torch
import json
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import normalize_embeddings, semantic_search, dot_score

def read_json(read_file_path):
    with open(read_file_path,encoding='utf-8') as file:
        raw_data=json.load(file)
    return raw_data

class EditMemory(object):
    def __init__(self, memory_mode,embedding_model):
        assert memory_mode in ['single_edit','batch_edit','sequence_edit']
        self.memory_mode = memory_mode
        self.edit_knowledge_dict={}
        self.edit_knowledge_ids=[]
        self.memory_size=0
        self.edit_knowledge_embedding=None
        self.embedding_model=embedding_model
    
    def print_memory_info(self):
        print('edit memory size:',self.memory_size)

    def __encode_knowledge(self,edit_knowledge):
            edit_embedding = self.embedding_model.encode(edit_knowledge)
            edit_embedding=torch.tensor(edit_embedding)  
            edit_embedding =edit_embedding.unsqueeze(0)  # [1,embedding_size]
            edit_embedding = normalize_embeddings(edit_embedding)
            return edit_embedding
    
    def __reset_memory(self):
        self.edit_knowledge_dict={} 
        self.edit_knowledge_ids=[]
        self.edit_knowledge_embedding=None
        self.memory_size=0

    
    
    def add_edit(self, edit_id, requested_rewrite, target_old=None, target_new=None, edit_mode='update'):
        """
        Add new edit_knowledge to memory.
        """
        if self.memory_mode == 'single_edit':
            # For single_edit, clear existing memory before adding new knowledge
            self.__reset_memory()
        
        assert edit_mode in ['update', 'add', 'delete']
        if edit_id not in self.edit_knowledge_ids:  # Prevent injecting duplicate edit_knowledge
            self.edit_knowledge_dict.update({self.memory_size: {'edit_id':edit_id,
                                                        'requested_rewrite':requested_rewrite,
                                                        'target_old':target_old,
                                                        'target_new':target_new,
                                                        'edit_mode':edit_mode}})
            self.edit_knowledge_ids.append(edit_id)
            self.memory_size+=1
            if edit_mode=='update':
                # edit_knowledge=f"\n'{requested_rewrite}' has been updated from {target_old} to {target_new}."
                edit_knowledge=f"{requested_rewrite} {target_old}."
                edit_embedding=self.__encode_knowledge(edit_knowledge)
                # update self.edit_knowledge_embedding
                if self.edit_knowledge_embedding==None:
                    self.edit_knowledge_embedding=edit_embedding
                else:
                    self.edit_knowledge_embedding = torch.cat((self.edit_knowledge_embedding, edit_embedding), dim=0)

    def retrival(self, query_data_point=None, query=None, top_k=1):
        """
        Retrieve top_k edit_knowledge from edit_memory based on query.
        """
        # Get query embedding
        if query_data_point is not None:
            query_embedding = self.embedding_model.encode(query_data_point['query'])
        elif query is not None:
            query_embedding = self.embedding_model.encode(query)
        else:
            raise ValueError("Either query_data_point or query must be provided")
        query_embedding = torch.tensor(query_embedding)
        query_embedding = query_embedding.unsqueeze(0)
        query_embedding = normalize_embeddings(query_embedding)
        # Retrieve and recall
        retrival_result = semantic_search(query_embeddings=query_embedding, 
                                corpus_embeddings=self.edit_knowledge_embedding, 
                                score_function=dot_score, 
                                top_k=top_k)[0]
        retrival_ids=[item['corpus_id'] for item in retrival_result]
        similarity_scores=[round(item['score'],4) for item in retrival_result]
        retrival_knowledges=[self.edit_knowledge_dict[id] for id in retrival_ids]
        if top_k==1:
            retrival_ids=retrival_ids[0]
            similarity_scores=similarity_scores[0]
            retrival_knowledges=retrival_knowledges[0]
        return retrival_ids,similarity_scores, retrival_knowledges

