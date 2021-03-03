import argparse
import json
import pickle
import os
import random
import subprocess
import torch
import time
import tqdm
import string
import pandas as pd
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from style_paraphrase.inference_utils import GPT2Generator
from profanity_filter import ProfanityFilter

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
pf = ProfanityFilter()

with open("config.json", "r") as f:
    configuration = json.loads(f.read())
    OUTPUT_DIR = configuration["output_dir"]

parser = argparse.ArgumentParser()

#parser.add_argument('--seed', type=int, default=34,
#                    help='Random seed to use for selecting inputs.')
parser.add_argument('--model', type=str, default='formal',
                    help='Paraphrasing model to use.')
parser.add_argument('--input_file', type=str, default='',
                    help='Input CSV file with "text" column that will be paraphrased')
parser.add_argument('--output_file', type=str, default='',
                    help='Input CSV file with "text" column that will be paraphrased')
parser.add_argument('--top_p_paraphrase', type=float, default=0.0,
                    help='Top p (nucleus) sampling value to use for the intermediate paraphrase.')
parser.add_argument('--top_p_style', type=float, default=0.6,
                    help='Top p (nucleus) sampling value to use for the stylistic paraphrase.')
parser.add_argument('--num_of_candidates', type=int, default=5,
                    help='Number of candidates to generate for each paraphrase input.')

args = parser.parse_args()

with torch.cuda.device(0):
    print("Loading paraphraser....")
    paraphraser = GPT2Generator(OUTPUT_DIR + "/models/paraphraser_gpt2_large", upper_length="same_5")
    paraphraser.gpt2_model.eval()
    print("Loading target style model:", args.model)
    model = GPT2Generator(os.path.join(OUTPUT_DIR, 'models', args.model))
    model.gpt2_model.eval()

style_mapping = {
    args.model: {"model": model, "device": 0, "data_file": args.model}
}

model_style_list = list(style_mapping.keys())

#random.seed(args.seed)

def generation_service():
    outputJson = []
    input_samples = []

    batchSize = 16
    with torch.no_grad():
        for i in range(0, len(input_samples), batchSize):
            output_paraphrase = []
            transferred_output = []
            currentBatch = input_samples[i:i + batchSize] 
            with torch.cuda.device(0):
                #output_paraphrase = paraphraser.generate_batch(currentBatch, top_p=data["settings"]["top_p_paraphrase"])[0]
                output_paraphrase, paraphrase_scores = paraphraser.generate_batch(currentBatch, top_p=args.top_p_paraphrase, get_scores=True)
                paraphrase_perplexities = 2 ** (-np.array(paraphrase_scores))

                with torch.cuda.device(style_mapping[args.model]["device"]):
                    model = style_mapping[args.model]["model"]
                    transferred_output, transferred_scores = model.generate_batch(output_paraphrase, top_p=args.top_p_style, get_scores=True)
                    transferred_perplexities = 2 ** (-np.array(transferred_scores))

            for inputText, paraphrase, paraphrasePerplexity, transferred, transferredPerplexity in zip(
                    currentBatch, output_paraphrase, paraphrase_perplexities, transferred_output, transferred_perplexities
                ):
                outputJson.append({
                    "input_text": inputText,
                    "paraphrase": paraphrase,
                    "paraphrase_perplexity": paraphrasePerplexity,
                    "style_transfer": transferred,
                    "style_perplexity": transferredPerplexity,
                    "target_style": args.model
                })
    
    df = pd.DataFrame(outputJson)
    df = df.assign(top_p_style=args.top_p_style, top_p_paraphrase=args.top_p_paraphrase)

    csvPath = args.output_file
    if (os.path.isfile(csvPath)):
        df.to_csv(csvPath, mode='a', header=False)
    else:
        df.to_csv(csvPath)


if __name__ == "__main__":
    path = OUTPUT_DIR + "/generated_outputs"
    print(path)
    generation_service()

