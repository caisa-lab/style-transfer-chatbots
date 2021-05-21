import torch
import os
import sys
import wandb
import pickle
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
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--max_epochs', type=int, required=True)
    parser.add_argument('--freeze_last_layers', type=int, default=0)
    parser.add_argument('--data_set', type=str, required=True)
    parser.add_argument('--org_data_dir', type=str, required=True)
    parser.add_argument('--label_0', type=str, required=True)
    parser.add_argument('--label_1', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='roberta-base')

    args = parser.parse_args()

    # train model
    def useTrainer():
        learningRate = args.learning_rate
        batchSize = args.train_batch_size
        dataSet = args.data_set
        modelName = args.model_name
        numOfEpochs = args.max_epochs
        gradAccumulationSteps = 1
        numOfSteps = int((len(trainDataset) * numOfEpochs) / (batchSize * gradAccumulationSteps))
        warmupSteps = int(0.06 * numOfSteps)
        loggingSteps = min(500, int(numOfSteps / (2 * numOfEpochs)))
        evalSteps = loggingSteps

        print('total steps:', numOfSteps)
        print('warmup steps:', warmupSteps)
        print('logging steps:', loggingSteps)
        print('eval steps:', evalSteps)
        
        #now = datetime.datetime.now()
        runName = 'lr-{}_batch-{}'.format(learningRate, batchSize)
        if (freezeLastLayersNum > 0):
            runName += '_layer-{}'.format(freezeLastLayersNum)
        wandb.init(project='final-{}-{}'.format(dataSet, modelName), name=runName, entity='philno')
        wandb.config.update(args)
        outputDir = os.path.join('./results', dataSet, modelName, runName)
        trainingArgs = TrainingArguments(
            output_dir=outputDir,          # output directory
            num_train_epochs=numOfEpochs,              # total number of training epochs
            per_device_train_batch_size=batchSize,  # batch size per device during training
            per_device_eval_batch_size=args.eval_batch_size,   # batch size for evaluation
            warmup_steps=warmupSteps,                # number of warmup steps for learning rate scheduler
            weight_decay=0.1,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=loggingSteps,
            evaluation_strategy='steps',
            eval_steps=evalSteps,
            learning_rate=learningRate,
            run_name=runName,
            dataloader_num_workers=4,
            gradient_accumulation_steps=gradAccumulationSteps,
            load_best_model_at_end=True,
            metric_for_best_model='eval_f1',
            greater_is_better=True,
            fp16=True,
            save_total_limit=3
        )

        trainer = Trainer(
            model=model,
            args=trainingArgs,
            train_dataset=trainDataset,
            eval_dataset=valDataset,
            compute_metrics=computeMetrics
        )

        trainer.train()
        model.eval()
        evalMetrics = trainer.evaluate()
        trainer.save_model(os.path.join(outputDir, 'best'))
        wandb.summary.update(evalMetrics)
        trainer.save_metrics('eval', evalMetrics)


    lr = args.learning_rate

    allLabels = [
        args.label_0,
        args.label_1
    ]
    numLabels = len(allLabels)
    dataSet = args.data_set

    # model, optimizer etc. setup from https://huggingface.co/transformers/training.html
    modelName = args.model_name
    model = AutoModelForSequenceClassification.from_pretrained(modelName, num_labels=numLabels, return_dict=True)
    tokenizer = AutoTokenizer.from_pretrained(modelName)
    
    def createDataset(orgDataDir, targetDataDir):
        # prepare dataset
        ORG_DATA_DIR = orgDataDir
        trainTexts, trainLabels = readSplit(os.path.join(ORG_DATA_DIR, 'train.csv'), allLabels)
        valTexts, valLabels = readSplit(os.path.join(ORG_DATA_DIR, 'val.csv'), allLabels)
        testTexts, testLabels = readSplit(os.path.join(ORG_DATA_DIR, 'test.csv'), allLabels)

        def tokenizeTexts(texts):
            return tokenizer(texts, return_tensors='pt', add_special_tokens=True, truncation=True, padding='max_length', max_length=256)
        #trainEncodings = tokenizer(trainTexts)
        trainEncodings = tokenizeTexts(trainTexts)
        #valEncodings = tokenizer(valTexts)
        valEncodings = tokenizeTexts(valTexts)
        testEncodings = tokenizeTexts(testTexts)

        trainDataset = StyleClassificationDataset(trainEncodings, trainLabels)
        valDataset = StyleClassificationDataset(valEncodings, valLabels)
        testDataset = StyleClassificationDataset(testEncodings, testLabels)

        print('Train dataset examples:')
        print(trainTexts[0] + ' == ' + labelNumberToString(trainLabels[0], allLabels))
        print(trainTexts[-1] + ' == ' +  labelNumberToString(trainLabels[-1], allLabels))
        print('Val dataset examples:')
        print(valTexts[0] + ' == ' + labelNumberToString(valLabels[0], allLabels))
        print(valTexts[-1] + ' == ' + labelNumberToString(valLabels[-1], allLabels))
        print('Test dataset examples:')
        print(testTexts[0] + ' == ' + labelNumberToString(testLabels[0], allLabels))
        print(testTexts[-1] + ' == ' + labelNumberToString(testLabels[-1], allLabels))

        Path(targetDataDir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(targetDataDir, 'train.pickle'), 'wb') as f2:
            pickle.dump(trainDataset, f2)
        with open(os.path.join(targetDataDir, 'val.pickle'), 'wb') as f2:
            pickle.dump(valDataset, f2)
        with open(os.path.join(targetDataDir, 'test.pickle'), 'wb') as f2:
            pickle.dump(testDataset, f2)

        print('dataset written')
        return trainDataset, valDataset, testDataset

    targetDataDir = os.path.join(get_script_path(), 'data', dataSet)
    if (not os.path.exists(targetDataDir)):
        createDataset(args.org_data_dir, targetDataDir)

    print('before loading dataset from pickle')
    with open(os.path.join(targetDataDir, 'train.pickle'), 'rb') as f:
        trainDataset = pickle.load(f)
    with open(os.path.join(targetDataDir, 'val.pickle'), 'rb') as f:
        valDataset = pickle.load(f)
    # testing in separate script!
    #with open(os.path.join(targetDataDir, 'test.pickle'), 'rb') as f:
    #    testDataset = pickle.load(f)

    print('Model:', modelName)
    print('Dataset:', dataSet)
    print('Num of train samples:', len(trainDataset))
    print('Num of val samples:', len(valDataset))

    # freeze all but last few layers
    freezeLastLayersNum = args.freeze_last_layers
    if (freezeLastLayersNum > 0):
        # freeze base model weights if set
        baseModel = model.base_model

        for param in baseModel.embeddings.parameters():
            param.requires_grad = False
    
        for layer in baseModel.encoder.layer[:-freezeLastLayersNum]:
            for param in layer.parameters():
                param.requires_grad = False

    print('Num of cuda devices:', torch.cuda.device_count())
    device = torch.device('cuda:0')
    model.to(device)
    model.train()
    useTrainer()