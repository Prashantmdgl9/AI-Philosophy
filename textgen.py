from aitextgen import aitextgen
from aitextgen.utils import GPT2ConfigCPU
from aitextgen.TokenDataset import TokenDataset
from aitextgen.tokenizers import train_tokenizer

config = GPT2ConfigCPU()

file_name = 'philo.txt'
train_tokenizer(file_name)
vocab_file = "aitextgen-vocab.json"
merges_file = "aitextgen-merges.txt"

ai = aitextgen(tf_gpt2="124M", config = config)
data = TokenDataset(file_name, block_size=64)
data

ai.train(data, batch_size=16, num_steps=200, save_every=10)






ai = aitextgen('trained_model/pytorch_model.bin', config = 'trained_model/config.json')
prompt_text = "What is life?"
prompt_text = "I hate you"

gpt_text = ai.generate(prompt = prompt_text, top_p = 0.9)



'''
str = '7.5, edss, 5, control'
element = str.split(',')
len(element)
element

numbers = []
for item in element:
    for subitem in item.split('.'):
        if((subitem.isdigit()) or (subitem == 'float')):
            numbers.append(subitem)

numbers

for item in element:
    for inner_item in item.split(' '):
        if (inner_item.isdigit()):
            numbers.append(inner_item)

k = '7.5'
k.isnumeric()


s = '7.5, edss, 5, control'
results = [t for t in s.split(',')
           if t.lstrip('+-').replace('.', '', 1).isdigit()]
print(results)  #1.5

import re
import numpy as np
str = '7.5, edss, 5, control'

def compute_mean(str):
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", str)
    res = [float(ele) for ele in matches]
    mean_val = np.mean(res)
    return mean_val

compute_mean(str)
'''
