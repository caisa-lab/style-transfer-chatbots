## Receptiveness demo

Set up your environment and download the `paraphraser_gpt2_large` as described in the original `README.md`. Additionally, run `pip install numpy pandas` to install extra dependencies.

Next, download the receptiveness paraphrasing models from [MEGA](https://mega.nz/file/wkpz2CCC#3JzZqKVMnF8OSo7pDlUEfqNJUYeu_IDvCqnPajck20I) and extract them to `man-generation/models`.

Place the models so you end up with a directory structure like this:

```
.
   |-cmd-demo
   |-datasets
   |-fairseq
   |-man-generation
   |---models
   |-----not_receptive
   |-----paraphraser_gpt2_large
   |-----receptive
   |-samples
   |-style_paraphrase
   |-web-demo
```

You can use the `cmd-demo/predict_service.py` script to paraphrase data of your choice with the model of your choice. The main difference to the much more simple scripts from `README_terminal_demo.md` are:

* it can work with text that is not split up into sentences
* it can paraphrase from original sentence to target style using the two-step approach in one script call
* it can select from multiple candidates based on model perplexity (lower == better)

There are some sample calls in the `cmd-demo` directory. All scripts assume that you have a `.csv` file with your input data. You can change most settings in the variables at the top of the script. 

To get a full list of possible arguments, run `python3 cmd-demo/predict_service.py --help`.

You can ignore the `Some weights of the model checkpoint...` warning that is shown when loading the intermediate paraphraser. This is due to a transformers version mismatch, but shouldn't impact the generated sentences.

### Experimenting with the "strength" of the style transfer

In contrast to some other paraphrasing approaches like variational autoencoders, there is no direct control over the "strength" of the style transfer. However, you can still influence the text generation in a number of ways.

You can vary the degree of "novelty" by changing the `top_p` value used for sampling for the intermediate and inverse paraphrasers. For some background on the different sampling strategies, please refer to [this blog article](https://huggingface.co/blog/how-to-generate). 

Even though the inverse paraphrasers are trained with the intermediate paraphraser (2-step approach), it might make sense in some cases to not use the intermediate paraphraser. For example, you could try chaining different styles or paraphrase to a certain style multiple times.
