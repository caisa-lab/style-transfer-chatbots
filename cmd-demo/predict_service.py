import argparse
import json
import os
import torch
import pandas as pd
import numpy as np
from style_paraphrase.inference_utils import GPT2Generator
from profanity_filter import ProfanityFilter
from nltk import sent_tokenize
from tqdm import tqdm

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
pf = ProfanityFilter()

with open(os.path.join(dname, 'config.json'), 'r') as f:
    configuration = json.loads(f.read())
    OUTPUT_DIR = configuration['output_dir']

parser = argparse.ArgumentParser()

#parser.add_argument('--seed', type=int, default=34,
#                    help='Random seed to use for selecting inputs.')
parser.add_argument('--model', type=str, required=True,
                    help='Paraphrasing model to use.')
parser.add_argument('--input_file', type=str, required=True,
                    help='Input CSV file with "text" column that will be paraphrased.')
parser.add_argument('--text_column', type=str, default='text',
                    help='Column name of the input CSV file that contains the text that will be paraphrased.')
parser.add_argument('--output_file', type=str, default=os.path.join(OUTPUT_DIR, 'paraphrases.csv'),
                    help='Path of output CSV file.')
parser.add_argument('--top_p_paraphrase', type=float, default=0.3,
                    help='Top p (nucleus) sampling value to use for the intermediate paraphrase.')
parser.add_argument('--top_p_style', type=float, default=0.6,
                    help='Top p (nucleus) sampling value to use for the stylistic paraphrase.')
parser.add_argument('--num_of_candidates', type=int, default=5,
                    help='Number of candidates to generate for each paraphrase input.')
parser.add_argument('--filter_candidates', type=bool, default=False,
                    help='If set to true, will only save the candidate with the best perplexity.')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch size to use for predictions')
parser.add_argument('--use_intermediate_paraphraser', type=bool, default=True,
                    help='Controls if a diverse paraphrasing model will be used before feeding the sentences to the stylistic paraphraser.')


args = parser.parse_args()

with torch.cuda.device(0):
    print('Loading target style model:', args.model)
    model = GPT2Generator(os.path.join(OUTPUT_DIR, 'models', args.model))
    model.gpt2_model.eval()

    if (args.use_intermediate_paraphraser):
        print('Loading paraphraser...')
        paraphraser = GPT2Generator(os.path.join(OUTPUT_DIR, 'models', 'paraphraser_gpt2_large'), upper_length='same_5')
        paraphraser.gpt2_model.eval()

style_mapping = {
    args.model: {'model': model, 'device': 0, 'data_file': args.model}
}

model_style_list = list(style_mapping.keys())

#random.seed(args.seed)

def main():
    input_samples = []
    sourceIndices = []
    sentenceIndices = []
    csvPath = args.output_file
    textCol = args.text_column

    print('Input file:', args.input_file)
    print('Using text from the column named:', args.text_column)
    inDf = pd.read_csv(args.input_file)
    numOfSamples = args.num_of_candidates
    sentenceIndex = 0
    print('Tokenizing {} samples...'.format(len(inDf[textCol])))
    for idx, text in tqdm(inDf[textCol].items()):
        sentences = [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 5]
        for sentence in sentences:
            for _ in range(numOfSamples):
                input_samples.append(sentence)
                sourceIndices.append(idx)
                sentenceIndices.append(sentenceIndex)
            sentenceIndex += 1

    print('Generating intermediate paraphrases of {} sentences...'.format(len(input_samples)))
    batchSize = args.batch_size
    with torch.no_grad():
        output_paraphrase = []
        paraphrase_perplexities = []
        if (args.use_intermediate_paraphraser):
            for i in tqdm(range(0, len(input_samples), batchSize)):
                currentBatch = input_samples[i:i + batchSize]
                with torch.cuda.device(0):
                    currentParaphrases, currentScores = paraphraser.generate_batch(currentBatch, top_p=args.top_p_paraphrase, get_scores=True)
                    currentPerplexities = 2 ** (-np.array(currentScores))
                
                output_paraphrase += currentParaphrases
                paraphrase_perplexities += currentPerplexities.tolist()

            # create df of intermediate / diverse paraphrases and save it
            df = pd.DataFrame().assign(source_idx=sourceIndices, sentence_idx=sentenceIndices, original_sentence=input_samples, paraphrase=output_paraphrase, paraphrase_perplexity=paraphrase_perplexities)
            df = df.sort_values(by=['sentence_idx', 'paraphrase_perplexity'])
            intermediatePath = csvPath.replace('.csv', '_intermediate.csv')
            if (os.path.isfile(intermediatePath)):
                df.to_csv(intermediatePath, mode='a', header=False)
            else:
                df.to_csv(intermediatePath)
            print('Saved intermediate / diverse paraphrases in', intermediatePath)
        
            # only use the most probable intermediate paraphrase
            if (args.top_p_paraphrase > 0):
                # exclude index columns
                replaceColumns = [c for c in df.columns if not 'idx' in c]
                for inputSentence in df['original_sentence'].unique():
                    myMask = df['original_sentence'] == inputSentence
                    currentDf = df.loc[myMask].sort_values(by='paraphrase_perplexity')
                    bestCandidate = currentDf.head(1)
                    df.loc[myMask, replaceColumns] = bestCandidate[replaceColumns].values.tolist()

            nextColumn = 'paraphrase'
        else:
            # just use the input samples
            df = pd.DataFrame().assign(source_idx=sourceIndices, sentence_idx=sentenceIndices, original_sentence=input_samples, paraphrase=output_paraphrase, paraphrase_perplexity=paraphrase_perplexities)
            nextColumn = 'original_sentence'

        print('Stylistically paraphrasing {} sentences...'.format(len(df)))
        transferred_output = []
        transferred_perplexities = []
        for i in tqdm(range(0, len(df), batchSize)):
            currentBatch = df[nextColumn][i:i + batchSize].tolist()
            with torch.cuda.device(style_mapping[args.model]['device']):
                model = style_mapping[args.model]['model']
                currentTransfers, currentScores = model.generate_batch(currentBatch, top_p=args.top_p_style, get_scores=True)
                currentPerplexities = 2 ** (-np.array(currentScores))

            transferred_output += currentTransfers
            transferred_perplexities += currentPerplexities.tolist()

    df = df.assign(style_transfer=transferred_output, style_perplexity=transferred_perplexities)
    df = df.assign(target_style=args.model, top_p_style=args.top_p_style, top_p_paraphrase=args.top_p_paraphrase)
    df = df.sort_values(by=['sentence_idx', 'style_perplexity'])

    if (args.filter_candidates):
        tempDf = []
        for sentenceIdx in df['sentence_idx'].unique():
            candidates = df.loc[df['sentence_idx'] == sentenceIdx]
            candidates = candidates.sort_values(by='style_perplexity')
            bestCandidate = candidates.iloc[0].to_dict()
            tempDf.append(bestCandidate)
        df = pd.DataFrame(tempDf)

    if (os.path.isfile(csvPath)):
        df.to_csv(csvPath, mode='a', header=False)
    else:
        df.to_csv(csvPath)

    print('Done. Saved stylistic paraphrases in', csvPath)

if __name__ == '__main__':
    main()

