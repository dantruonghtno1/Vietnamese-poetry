import  torch
from transformers import AutoTokenizer, GPT2LMHeadModel, pipeline

def infer(text, model_path = '/home/truong/Documents/poem/save_model'):
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    tokenizer.add_tokens('\n')
    model = pipeline('text-generation', model = model_path, tokenizer = tokenizer)

    text = tokenizer.bos_token +' '+  text 
    return model(text)[0]['generated_text']

# if __name__ == "__main__":
#     text = "yêu"
#     print(infer(text, model_path='save_model'))
