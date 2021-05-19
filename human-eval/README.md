# Human evaluation

This directory contains the code used for human evaluation of an appropriate chatbot style.
We conduct a crowdsourcing study on Amazon Mechanical Turk (MTurk).

## Project structure

* `mturk` contains a wrapper around the boto3 client for MTurk.
* `paraphrases` contains scripts to generate and select paraphrases of conversations.
* `templates` contains various template files for HITs and qualification tests.

There are lots of smaller scripts for specific use cases.
The most important scripts are `create-hits.py` to create and deploy HITs and `retrieve_results.py` to collect the resulting annotations.
These scripts mainly use data from two folders which will be automatically created: 

* `hit-data` contains HIT objects and corresponding API responses from MTurk.
* `hit-output` contains the resulting assignments, worker lists and other direct results of annotations.

## Setting up access to MTurk

Follow AWS best practices for setting up credentials here:
http://boto3.readthedocs.io/en/latest/guide/configuration.html
