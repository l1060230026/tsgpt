import torch

from transformers import AutoConfig, StoppingCriteria


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = [tokenizer(keyword).input_ids for keyword in keywords]
        self.keyword_ids = [keyword_id[0] for keyword_id in self.keyword_ids if type(keyword_id) is list and len(keyword_id) == 1]
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids
        self.stopped_mask = torch.zeros(input_ids.shape[0], dtype=torch.bool)  # 使用mask替代set
        
    def reset(self):
        """重置停止条件的状态"""
        self.start_len = None
        self.stopped_mask = torch.zeros(self.input_ids.shape[0], dtype=torch.bool)
        
    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        
        # 检查最后一个token是否匹配任何keyword_id
        last_tokens = output_ids[:, -1]
        keyword_matches = torch.tensor([token in self.keyword_ids for token in last_tokens])
        
        # 更新停止状态
        self.stopped_mask = self.stopped_mask | keyword_matches
        
        # 如果所有样本都已停止，返回True
        return self.stopped_mask.all()