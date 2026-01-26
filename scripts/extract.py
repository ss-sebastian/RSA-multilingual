from transformers import BertTokenizer, BertForSequenceClassification
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from transformers import GPT2Tokenizer, GPT2Model
from transformers import AutoTokenizer, AutoModel
import torch

class extract_bert_hidden_states:
    def __init__(self, model_name="google-bert/bert-base-multilingual-cased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, output_hidden_states=True)

    def get_hidden_states(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        hidden_states = outputs.hidden_states
        return hidden_states

class extract_xlm_hidden_states:
    def __init__(self, model_name="xlm-roberta-base"):
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        self.model = XLMRobertaModel.from_pretrained(model_name, output_hidden_states=True)

    def get_hidden_states(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        hidden_states = outputs.hidden_states
        return hidden_states
    
class extract_gpt2_hidden_states:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name, output_hidden_states=True)

    def get_hidden_states(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        hidden_states = outputs.hidden_states
        return hidden_states

class extract_microsoft_hidden_states: 
    def __init__(self, model_name="microsoft/mdeberta-v3-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    def get_hidden_states(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        hidden_states = outputs.hidden_states
        return hidden_states
    
class extract_qwen_hidden_states:
    def __init__(self, model_name="Qwen/Qwen-7B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    def get_hidden_states(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        hidden_states = outputs.hidden_states
        return hidden_states

class extract_llama_hidden_states:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    def get_hidden_states(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        hidden_states = outputs.hidden_states
        return hidden_states

class extract_aya_hidden_states:
    def __init__(self, model_name="CohereLabs/aya-23-8B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    def get_hidden_states(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        hidden_states = outputs.hidden_states
        return hidden_states

class extract_transformer_hidden_states:
    def __init__(self, model_name):
        if "bert" in model_name:
            self.extractor = extract_bert_hidden_states(model_name)
        elif "gpt2" in model_name:
            self.extractor = extract_gpt2_hidden_states(model_name)
        elif "qwen" in model_name:
            self.extractor = extract_qwen_hidden_states(model_name)
        elif "llama" in model_name:
            self.extractor = extract_llama_hidden_states(model_name)
        elif "aya" in model_name:
            self.extractor = extract_aya_hidden_states(model_name)
        elif "xlm" in model_name:
            self.extractor = extract_xlm_hidden_states(model_name)
        elif "microsoft" in model_name:
            self.extractor = extract_microsoft_hidden_states(model_name)
        else:
            raise ValueError(f"Model {model_name} not supported.")

    def get_hidden_states(self, text):
        return self.extractor.get_hidden_states(text)
    