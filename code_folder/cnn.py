#import mods
!pip install transformers
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
max_len=160
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
device = torch.device("cpu")



class RtwtDataset(Dataset):
    def __init__(self, content, retweet, tokenizer, max_len):
        self.content = content
        self.retweet= retweet
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return (len(self.content))

    def __getitem__(self,item):

        content =  str(self.content[item])
        retweet = self.retweet[item]

        encoding = self.tokenizer.encode_plus(
            content,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            )

        return {
        'text_content': content,
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'retweet': torch.tensor(retweet, dtype=torch.float)
        }

df_x = pd.read_csv("/content/drive/MyDrive/50.021-AI-Project-2022/datasets/working_dataset/tweet_content_only/train/X_train.csv")
df_y = pd.read_csv("/content/drive/MyDrive/50.021-AI-Project-2022/datasets/working_dataset/tweet_content_only/train/y_train.csv")
df_train = pd.concat([df_x,df_y], axis=1)
 
#df_train, df_test = train_test_split(df, test_size=0.1, )
#df_val, df_test = train_test_split(df_test, test_size=0.5)

def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = RtwtDataset(
    content=df.original_text.to_numpy(),
    retweet=df.retweet_count.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
  )

batch_size = 16

train_data_loader = create_data_loader(df_train, tokenizer, max_len, batch_size)

data = next(iter(train_data_loader))
data.keys()


bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)



class Predictor(nn.Module):

  def __init__(self):
    super(Predictor, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    

    self.conv = nn.Conv1d(in_channels = 768, out_channels = 100, kernel_size=1)

    
    
    self.fc = nn.Linear( 100, 1)
        
    
  
  def forward(self, input_ids, attention_mask):
    


    
   #hidden_state shape = batch, seq_len, feature_dim
   #conv1d  accepts batch, dim, len

    hidden_state = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )[0]
    hs = hidden_state.permute(0,2,1)

    print('hs shape', hs.size())
    
    conved = F.relu(self.conv(hs))
        
    #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
            
    pooled = F.max_pool1d(conved,1)
    print('pooled size',pooled.size())
    
    #pooled_n = [batch size, n_filters]
    #print('pooled view', pooled.view(-1,100).size())
    #output = self.fc(pooled.view(-1,100))
    pool_permute = pooled.permute(0, 1, 2)[:, :, -1]
    #print(pool_permute.size())
    output = self.fc(pool_permute)
    #output = self.fc(output)
    print('output size', output.size())
    
        
    return output

model = Predictor()
model = model.to(device)

input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)

EPOCHS = 10

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

#loss_fn = F.mse_loss().to(device)

def train_epoch(
  model, 
  data_loader, 
  
  optimizer, 
  device, 
  scheduler, 
  n_examples
):
  model = model.train()

  losses = []
  correct_predictions = 0
  
  for d in data_loader:
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    retweet = d["retweet"].to(device)

    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    print(outputs)
    print(outputs.size())
    holder, preds = torch.max(outputs, dim=1)
    print('torch max',torch.max(outputs, dim=1))
    print('holder',holder)
    print('preds',preds)
    print('first',torch.max(outputs, dim=1)[0])
    retweet = retweet.view(holder.size()[0],-1)
    print('retweet SIZE', retweet.size())
    loss = F.mse_loss(outputs, retweet).to(device)
    #loss = loss.to(device)

    print('preds',preds)
    print('pred size', preds.size())
    print('retweet',retweet)
    print('retweet size',retweet.size())
    correct_predictions += torch.sum(holder == retweet)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions.float() / n_examples, np.mean(losses)


for epoch in range(EPOCHS):

    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,    
       
        optimizer, 
        device, 
        scheduler, 
        len(df_train)
    )
    print('train acc: ', train_acc, "train_loss: ", train_loss)