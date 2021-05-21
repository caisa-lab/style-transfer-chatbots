import pandas as pd
import numpy as np
import gzip
import json
import torch
from tqdm import tqdm
from profanity_check import predict
from transformers import RobertaForSequenceClassification, RobertaTokenizer


class MyDataset(torch.utils.data.Dataset):
  def __init__(self, encodings):
    self.encodings = encodings

  def __len__(self):
    return len(self.encodings['input_ids'])

  def __getitem__(self, idx):
    item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    return item

# code from https://nijianmo.github.io/amazon/index.html#subsets
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    if 'reviewText' not in d.keys():
        continue
    d['reviewText'] = d['reviewText'].replace('\n', ' ').replace('\r', '')
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

def preselectOffensiveTitles():
  df = getDF('/data/daten/datasets/reviews/amazon/Video_Games_5.json.gz')
  df = df.append(getDF('/data/daten/datasets/reviews/amazon/Sports_and_Outdoors_5.json.gz'))

  df = df.loc[~pd.isna(df['summary'])]
  isOffensive = np.array(predict(df['summary']), dtype='int')# + np.array(predict(df['reviewText']), dtype='int')
  df = df.assign(isOffensive=isOffensive)
  off = df.loc[df['isOffensive'] != 0]
  reviewText = off['reviewText']
  off = off.drop(columns=['reviewText'])
  off = off.assign(reviewText=reviewText)
  print('# of potentially offensive reviews:', len(off))

  off.to_csv('offensive-reviews.csv')

def refineOffensiveTitles():
  off = pd.read_csv('offensive-reviews.csv')

  device = torch.device('cuda')
  CLASSIFIER_PATH = '/data/daten/python/master/gyafc-classifier/results/olid/lr-1e-05_batch-32_layer-3/best/'
  tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
  model = RobertaForSequenceClassification.from_pretrained(CLASSIFIER_PATH, return_dict=True)
  model.to(device)
  model.eval()
  texts = off['summary'].values.tolist()
  encodings = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, add_special_tokens=True, max_length=82)
  myDataset = MyDataset(encodings)
  batchSize = 1024
  dataLoader = torch.utils.data.DataLoader(myDataset, batch_size=batchSize, shuffle=False)

  preds = []
  with torch.no_grad():
    for i, batch in enumerate(tqdm(dataLoader)):
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      pred = model(input_ids, attention_mask=attention_mask)
      pred = pred['logits'].argmax(1)
      pred = pred.cpu().tolist()
      preds += pred

  off = off.assign(roberta=preds)
  off = off.loc[off['roberta'] == 1]
  off.to_csv('offensive-reviews-roberta.csv')