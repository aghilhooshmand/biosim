import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import sys,os,tqdm
from tqdm.auto import tqdm

# Load model from HuggingFace Hub

#Func to calculate similarity
#############################
def sentence_similarity_by_torch_BioLORD(s1:list,s2:list)->pd.DataFrame:
  #Load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained('FremyCompany/BioLORD-STAMB2-v1', cache_dir="bioSim_model/")
    model = AutoModel.from_pretrained('FremyCompany/BioLORD-STAMB2-v1', cache_dir="bioSim_model/").to(device)

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    # Sentences we want sentence embeddings for
    result=pd.DataFrame(columns=['s1','s2','similarity'])
    print("Similarity calculations should be performed for the two list of sentences [{}, {}].Therefore, it will calculate overall: {}".format(len(s1),len(s2),len(s1)*len(s2)))
    # k=len(s1)*len(s2)
    for i in tqdm(s1):
        for j in s2:
            sentences = [i, j]
            # Tokenize sentences
            encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
            # Compute token embeddings
            with torch.no_grad():
                model_output = model(**encoded_input)
            # Perform pooling
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            # Normalize embeddings
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            r=util.cos_sim(sentence_embeddings[0], sentence_embeddings[1]).cpu().data.numpy()[0][0]
            result.loc[len(result)]=[i,j,np.round(r,2)]
    return(result)

def get_similarity(data:pd.DataFrame):
    result=sentence_similarity_by_torch_BioLORD(np.array(data.iloc[:,0:1].dropna().values).flatten().tolist(), np.array(data.iloc[:,1:2].dropna().values).flatten().tolist())
    return(result)


def main():
    current_working_directory = os.getcwd()
    data=pd.read_csv(current_working_directory+'/'+sys.argv[1],delimiter='\t',header=None)
    result=get_similarity(data)
    result.to_csv(current_working_directory+'/'+'result.csv',sep='\t',index=False)
  

if __name__ == '__main__':
    main()
