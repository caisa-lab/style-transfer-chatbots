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
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from style_paraphrase.inference_utils import GPT2Generator
from profanity_filter import ProfanityFilter

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
pf = ProfanityFilter()

parser = argparse.ArgumentParser()

#parser.add_argument('--seed', type=int, default=34,
#                    help='Random seed to use for selecting inputs.')
args = parser.parse_args()

with open("config.json", "r") as f:
    configuration = json.loads(f.read())
    OUTPUT_DIR = configuration["output_dir"]


with torch.cuda.device(0):
    print("Loading paraphraser....")
    paraphraser = GPT2Generator(OUTPUT_DIR + "/models/paraphraser_gpt2_large", upper_length="same_5")
    print("Loading target style model...")
    #formality = GPT2Generator(OUTPUT_DIR + "/models/formality")
    politeness = GPT2Generator(OUTPUT_DIR + "/models/politeness")
    #impoliteness = GPT2Generator(OUTPUT_DIR + "/models/impoliteness")


style_mapping = {
    #"Formality": {"model": formality, "device": 0, "data_file": "formality"}
    "Politeness": {"model": politeness, "device": 0, "data_file": "politeness"}
    #"Impoliteness": {"model": impoliteness, "device": 0, "data_file": "politeness"}
}

data_style_mapping = {
    #"aae": {"data_file": "aae"},
    #"bible": {"data_file": "bible"},
    #"english_tweets": {"data_file": "english_tweets"},
    #"lyrics": {"data_file": "lyrics"},
    #"switchboard": {"data_file": "switchboard"},
    #"arg-summaries": {"data_file": "arg-summaries"},
    #"prod-summaries": {"data_file": "prod-summaries"}
    "off-reviews": {"data_file": "off-reviews"},
    "receptiveness": {"data_file": "receptiveness"}
}

data_style_list = list(data_style_mapping.keys())
model_style_list = list(style_mapping.keys())

if os.path.exists(OUTPUT_DIR + "/all_styles_clean.pkl"):
    with open(OUTPUT_DIR + "/all_styles_clean.pkl", "rb") as f:
        all_sents_clean = pickle.load(f)
else:
    print("Loading datasets for random cases...")
    all_sents = {}
    for style, data in data_style_mapping.items():
        with open("../samples/data_samples/{}.txt".format(data["data_file"]), "r") as f:
            all_sents[style] = f.read().strip().split("\n")

    all_sents_clean = {}
    for style, sents in all_sents.items():
        #all_sents_clean[style] = [x.strip().strip("\"").strip("\'").strip() for x in tqdm.tqdm(sents) if pf.is_clean(x) and len(x.split()) < 25]
        all_sents_clean[style] = [x.strip().strip("\"").strip("\'").strip() for x in tqdm.tqdm(sents) if len(x.split()) < 25]

    with open(OUTPUT_DIR + "/all_styles_clean.pkl", "wb") as f:
        pickle.dump(all_sents_clean, f)

#random.seed(args.seed)


# class EventHandler(FileSystemEventHandler):
#     def on_any_event(self, event):
def generation_service():
    next_key = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    data = {
        'random': True,
        'target_style': None
    }
    data['settings'] = {
        'top_p_style': 0.6,
        'top_p_paraphrase': 0.0
    }

    numOfSamples = 20
    outputJson = []
    for i in range(numOfSamples):
        if data["random"]:
            if data["target_style"] is None:
                data["target_style"] = random.choice(model_style_list)

            other_styles = [style for style in data_style_list if style != data["target_style"]] + ['prod-summaries']
            input_style = random.choice(data_style_list)

            data["input_text"] = random.choice(all_sents_clean[input_style])

        input_samples = [data["input_text"] for _ in range(5)]

        with torch.cuda.device(0):
            output_paraphrase = paraphraser.generate_batch(input_samples, top_p=data["settings"]["top_p_paraphrase"])[0]

        if data["target_style"] is None:
            transferred_output = ["", "", "", "", ""]
        else:
            with torch.cuda.device(style_mapping[data["target_style"]]["device"]):
                model = style_mapping[data["target_style"]]["model"]
                transferred_output = model.generate_batch(output_paraphrase, top_p=data["settings"]["top_p_style"])[0]

        for paraphrase, transferred in zip(output_paraphrase, transferred_output):
            outputJson.append({
                "input_text": data["input_text"],
                "paraphrase": paraphrase,
                "style_transfer": transferred,
                "target_style": data["target_style"]
            })

    outPath = OUTPUT_DIR + '/generated_outputs/' + data["target_style"] + '/' 
    os.makedirs(outPath, exist_ok=True)
    with open(outPath + next_key + '.json', 'w') as f:
        f.write(json.dumps(outputJson))
    
    df = pd.DataFrame(outputJson)
    df.to_csv(outPath + next_key + '.csv')
    


if __name__ == "__main__":
    path = OUTPUT_DIR + "/generated_outputs"
    print(path)
    #for _ in range(10):
    generation_service()
    # event_handler = EventHandler()
    # observer = Observer()
    # observer.schedule(event_handler, path, recursive=True)
    # observer.start()
    # try:
    #     while True:
    #         time.sleep(1)
    # except KeyboardInterrupt:
    #     observer.stop()
    # observer.join()
