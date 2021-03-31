## Receptiveness demo

Set up your environment and download the `paraphraser_gpt2_large` as described in the `README_terminal_demo.md`.

Additionally, download the receptiveness paraphrasing models from [MEGA](https://mega.nz/file/wkpz2CCC#3JzZqKVMnF8OSo7pDlUEfqNJUYeu_IDvCqnPajck20I) and extract them to `man-generation/models`.

You can use the `cmd-demo/predict_service.py` script to paraphrase data of your choice with the model of your choice. The main difference to the much more simple scripts from `README_terminal_demo.md` are:

* it can work with text that is not split up into sentences
* it can paraphrase from original sentence to target style in one script call
* it can select from multiple candidates based on model perplexity

There are some sample calls in the `cmd-demo` directory. All scripts assume that you have a `.csv` file with your input data. You can change most settings in the variables at the top of the script.

### Experimenting with the "strength" of the style transfer

In contrast to some other paraphrasing approaches like variational autoencoders, there is no direct control over the "strength" of the style transfer. However, you can still influence the text generation in a number of ways.

You can vary the degree of "novelty" by changing the `top_p` value used for sampling for the intermediate and inverse paraphrasers. For some background on the different sampling strategies, please refer to [this blog article](https://huggingface.co/blog/how-to-generate). 

Even though the inverse paraphrasers are trained with the intermediate paraphraser (2-step approach), it might make sense in some cases to not use the intermediate paraphraser. For example, you could try chaining different styles or paraphrase to a certain style multiple times.
