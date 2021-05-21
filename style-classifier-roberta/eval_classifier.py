import torch
import os
import sys
import wandb
import pickle
import json
import pandas as pd
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))

def computeMetrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def labelToNumber(currentLabel: str, allLabels: list):
    #label = np.zeros(len(allLabels))
    #label[allLabels.index(currentLabel)] = 1
    return allLabels.index(currentLabel)

def labelNumberToString(labelNumber, allLabels):
    #return allLabels[np.argmax(labelNumber)]
    return allLabels[labelNumber]

def readSplit(csvFilePath, allLabels: list):
    # my split is already preprocessed and does not need to be shuffled
    df = pd.read_csv(csvFilePath)

    texts = []
    labels = []
    for index, row in df.iterrows():
        texts.append(row['text'])
        labels.append(labelToNumber(row['label'], allLabels))
    
    return texts, labels

# dataset class from https://huggingface.co/transformers/custom_datasets.html
class StyleClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--data_set', type=str, required=True)
    parser.add_argument('--org_data_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--label_0', type=str, required=True)
    parser.add_argument('--label_1', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='roberta-base')

    args = parser.parse_args()

    # train model
    def testCheckpoint():
        dataSet = args.data_set
        modelName = args.model_name
        gradAccumulationSteps = 1

        #now = datetime.datetime.now()
        runName = 'best-model-test'
       
        wandb.init(project='final-{}-{}'.format(dataSet, modelName), name=runName, entity='philno')
        wandb.config.update(args)
        outputDir = os.path.join(args.checkpoint_dir, 'temp')
        trainingArgs = TrainingArguments(
            output_dir=outputDir,          # output directory
            per_device_eval_batch_size=args.eval_batch_size,   # batch size for evaluation
            run_name=runName,
            dataloader_num_workers=4,
            gradient_accumulation_steps=gradAccumulationSteps,
            fp16=True
        )

        trainer = Trainer(
            model=model,
            args=trainingArgs,
            compute_metrics=computeMetrics
        )
        with torch.no_grad():
            model.eval()
            evalMetrics = trainer.predict(testDataset).metrics
            wandb.run.summary.update(evalMetrics)
            print(evalMetrics)
        
        with open(os.path.join(args.checkpoint_dir, '..', 'test_results.json'), 'w') as f:
            json.dump(evalMetrics, f)


    allLabels = [
        args.label_0,
        args.label_1
    ]
    numLabels = len(allLabels)
    dataSet = args.data_set

    # model, optimizer etc. setup from https://huggingface.co/transformers/training.html
    modelName = args.model_name
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_dir, num_labels=numLabels, return_dict=True)
    tokenizer = AutoTokenizer.from_pretrained(modelName)
    
    def createDataset(orgDataDir, targetDataDir):
        # prepare dataset
        ORG_DATA_DIR = orgDataDir
    
        testTexts, testLabels = readSplit(os.path.join(ORG_DATA_DIR, 'test.csv'), allLabels)

        def tokenizeTexts(texts):
            return tokenizer(texts, return_tensors='pt', add_special_tokens=True, truncation=True, padding='max_length', max_length=256)
        
        testEncodings = tokenizeTexts(testTexts)

        testDataset = StyleClassificationDataset(testEncodings, testLabels)

        print('Test dataset examples:')
        print(testTexts[0] + ' == ' + labelNumberToString(testLabels[0], allLabels))
        print(testTexts[-1] + ' == ' + labelNumberToString(testLabels[-1], allLabels))

        Path(targetDataDir).mkdir(parents=True, exist_ok=True)

        with open(os.path.join(targetDataDir, 'test.pickle'), 'wb') as f2:
            pickle.dump(testDataset, f2)

        print('dataset written')
        return testDataset

    targetDataDir = os.path.join(get_script_path(), 'data', dataSet)
    if (not os.path.exists(targetDataDir)):
        createDataset(args.org_data_dir, targetDataDir)

    print('before loading dataset from pickle')
    # testing in separate script!
    with open(os.path.join(targetDataDir, 'test.pickle'), 'rb') as f:
        testDataset = pickle.load(f)

    print('Model:', modelName)
    print('Dataset:', dataSet)
    print('Num of test samples:', len(testDataset))

    print('Num of cuda devices:', torch.cuda.device_count())
    device = torch.device('cuda:0')
    model.to(device)
    testCheckpoint()