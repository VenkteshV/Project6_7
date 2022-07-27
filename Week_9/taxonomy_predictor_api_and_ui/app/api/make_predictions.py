import torch
import torch.nn as nn
import os
import joblib
import numpy as np
import torch.nn.functional as F
from transformers import BertModel, AdamW, BertConfig, BertTokenizer
from sentence_transformers import SentenceTransformer
dir_path = os.path.dirname(os.path.realpath(__file__))

sent_model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')
model_path_recommend = os.path.join(dir_path, '../models/model_euclidean_SENT_BERT_cos_1')
pathData = os.path.join(dir_path, '../data')

test_labels = joblib.load(pathData+'/test_labels')
test_labels = np.array(test_labels)

LE = joblib.load(pathData+'/label_encoder_reduced')

labels_with_board = joblib.load(pathData+'/labels_formatted')

class MulticlassClassifier(nn.Module):
    def __init__(self,bert_model_path):
        super(MulticlassClassifier,self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_path,output_hidden_states=False,output_attentions=False)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(768, 384)
        self.fc2 = nn.Linear(384, 1024)

    def forward(self,tokens,masks):
        _, pooled_output = self.bert(tokens, attention_mask=masks)
        x = self.fc1(pooled_output)
        x = self.fc2(x)
        return x


model = MulticlassClassifier('bert-base-uncased')
model.load_state_dict(torch.load(model_path_recommend+'/model_weights.zip',map_location=torch.device('cpu')))
recommender_tokenizer = BertTokenizer.from_pretrained(model_path_recommend, do_lower_case=True)

def get_labels(prediction):
    predicted_label =  LE.inverse_transform([prediction])
    return predicted_label[0]

def get_cleaned_taxonomy(taxonomy):
  cleaned_taxonomy = []
  for value in taxonomy:
      value = ' '.join(value.split(">>"))
      cleaned_taxonomy.append( value )
  return cleaned_taxonomy


def get_taxonomy_embeddings():
    cleaned_taxonomy = get_cleaned_taxonomy(test_labels)
    taxonomy_vectors = sent_model.encode(cleaned_taxonomy)
    taxonomy_vectors = np.vstack(taxonomy_vectors)
    test_poincare_tensor = torch.tensor(taxonomy_vectors,dtype=torch.float)
    return test_poincare_tensor
test_poincare_tensor = get_taxonomy_embeddings()
print("test_labels",test_poincare_tensor.shape)

cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)


def recommend_taxonomy(text):

    encoded_dict = recommender_tokenizer.encode_plus(
                        text,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 128,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.    
    input_ids = encoded_dict['input_ids']
    
    attention_masks = encoded_dict['attention_mask']

    # Tracking variables 
    predictions , true_labels = [], []
    with torch.no_grad():
        outputs = model(input_ids.reshape(1,-1),attention_masks.reshape(1,-1))
        
    # distances = (F.normalize(outputs,p=2,dim=1) - F.normalize(test_poincare_tensor,p=2,dim=1)).pow(2).sum(1)
    distances = cos(outputs,test_poincare_tensor)
    distances,indices = torch.topk(distances,3,largest=True)

    top_k_labels = test_labels[indices.cpu().numpy()]
    top_k_labels = list(top_k_labels)
    print("top_k_labels are", test_labels[indices.cpu().numpy()])

    final_list = []
    results = []
    for label,distance in zip(top_k_labels,distances):
        for formatted_label in labels_with_board:
            if label in formatted_label and len(label.split(">>"))>1:
                final_list.append((formatted_label,distance))
    for (prediction,distance)  in final_list:
        results.append({
            "taxonomy": prediction,
            "confidence": distance       })


    return results