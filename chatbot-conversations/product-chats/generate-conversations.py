from collections import defaultdict
import pandas as pd
import numpy as np
import os
import multiprocessing

import en_core_web_lg
import pickle

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer 

import random
import re

if __name__ == '__main__':
    nlp_spacy = en_core_web_lg.load()
    stop_words = set(stopwords.words('english'))
    wordnet_lemmatizer = WordNetLemmatizer()

    # functions from vahid
    def review_aspect_checker(aspect, review, wrong_aspects, correct_forms):
        review_words = re.sub('[^a-zA-Z]+', ' ', str(review)).split()
        if aspect != None:
            aspect_words = re.sub('[^a-zA-Z]+', ' ', str(aspect)).split()
            if len(aspect_words) == 1:
                form = wordnet.synsets(str(aspect_words[0]))
                if form:
                    verb = form[0].pos()
                    if review_words:
                        if verb == "v":
                            reason = "VERB"
                            result = False
                        elif len(review_words) < 3:
                            reason = "LEN_REVIEW"
                            result = False
                        elif str(aspect_words[0]).lower() in wrong_aspects:
                            reason = "WRONG_ASPECT"
                            result = False
                        elif len(str(aspect_words[0])) < 3:
                            reason = "LEN_ASPECT"
                            result = False
                        else:
                            reason = "OK"
                            result = True
                    else:
                        reason = "NO_REVIEW"
                        result = False
                else:
                    if str(aspect_words[0]).lower() not in correct_forms:
                        reason = "NO_FORM"
                        result = False
                    else:
                        reason = "OK"
                        result = True          
            else:
                if review_words:
                    if len(review_words) < 3:
                        reason = "LEN_REVIEW"
                        result = False
                    elif len(aspect_words) == 0:
                        reason = "NO_ASPECT"
                        result = False
                    else:
                        reason = "OK"
                        result = True
                else:
                    reason = "NO_REVIEW"
                    result = False
        else:
            if len(review_words) < 2:
                reason = "LEN_REVIEW"
                result = False
            else:
                reason = "ASPECT_NONE"
                result = False
        return(result, reason)

    nouns = {x.name().split('.', 1)[0] for x in wordnet.all_synsets('n')}
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())

    # @Vahid I should find a better solution >> E.g. First use the Punctuation model (https://github.com/chrisspen/punctuator2) for reviews and then ...
    def retrieve_sentences_from_review(review, word):
        final_sent = []
        condition = re.compile(r'%s' %word, flags=re.IGNORECASE)
        review = review.replace("\n\n", " ")
        review = review.replace("\n", " ")
        sentences = sent_tokenize(review)
        for sent in sentences:
            check = condition.search(sent)
            if check:
                final_sent.append(sent)
        all_sent = " ".join(final_sent)
        returnLength = len(all_sent)
        if (returnLength <= 3 or returnLength >= 200):
            return ''
        # else
        return(all_sent)

    def cleaning_process(text):
        cleaned_text = text.replace("\n\n", " ")
        cleaned_text = cleaned_text.replace("\n", " ")
        cleaned_text = cleaned_text.replace("##", " ")
        cleaned_text = cleaned_text.replace("(", " ")
        cleaned_text = cleaned_text.replace(")", " ")
        cleaned_text = cleaned_text.replace("*", " ")
        cleaned_text = cleaned_text.replace("+", " ")
        cleaned_text = re.sub(' +', ' ', cleaned_text)
        cleaned_text = cleaned_text.strip()
        return(cleaned_text)

    def aspects_similarity_check(aspect_1, aspect_2, english_vocab=english_vocab):
        aspects_list = [aspect_1, aspect_2]
        aspects_list.sort(key=len, reverse=False)
        aspect_1 = aspects_list[0]
        aspect_2 = aspects_list[1]
        
        aspect_2_new = aspect_2.replace(str(aspect_1), "")
        words_list_aspect_2_new = word_tokenize(aspect_2_new)
        for i in words_list_aspect_2_new:
            if i not in english_vocab or len(i) == 1:
                return(False)
                break
            else:
                return(True)
            
    def aspects_check(aspect_1, aspect_2, english_vocab=english_vocab):
        aspects_list = [aspect_1, aspect_2]
        aspects_list.sort(key=len, reverse=False)
        aspect_1 = aspects_list[0]
        aspect_2 = aspects_list[1]
        
        aspect_2_new = aspect_2.replace(str(aspect_1), "")
        words_list_aspect_2_new = word_tokenize(aspect_2_new)
        for i in words_list_aspect_2_new:
            if i not in english_vocab or len(i) == 1:
                return(False)
                break
            else:
                return(True)
            
    def Qpos1A_Apos1A(item, wrong_aspects, correct_forms, Q1A_list, dict_AspectSentiment):
        blocks = {}
        counter = 0
        item_review_list = dict_AspectSentiment.get(item)
        if item_review_list:
            for review_dict in item_review_list:
                for item_reviewer_aspect_key, review_sentiment in (review_dict.items()):
                    key = item_reviewer_aspect_key[0]
                    aspect = item_reviewer_aspect_key[1]
                    review = review_sentiment['review']
                    polarity = review_sentiment['polarity']
                    result, reason = review_aspect_checker(aspect, review, wrong_aspects, correct_forms)
                    if result and aspect not in wrong_aspects:
                        if str(polarity).lower() == 'positive':
                            aspect = cleaning_process(str(aspect))
                            aspect = re.sub('[^a-zA-Z]+', ' ', str(aspect))
                            
                            Qpos1A = random.choice(Q1A_list).format(aspect)
                            
                            review = cleaning_process(str(review))
                            
                            #@Vahid I should find a solution; how to retrieve the related sentence for the aspect?
                            Apos1A = retrieve_sentences_from_review(review, aspect)
                            #Apos1A = review

                            if (len(Apos1A) <= 3):
                                continue
                            
                            counter += 1
                            blocks["Qpos1A_Apos1A_" + str(counter)] = {}
                            blocks["Qpos1A_Apos1A_" + str(counter)]['Qpos1A'] = {}
                            blocks["Qpos1A_Apos1A_" + str(counter)]['Qpos1A']['Question'] = Qpos1A
                            blocks["Qpos1A_Apos1A_" + str(counter)]['Qpos1A']['Labels'] = {}
                            blocks["Qpos1A_Apos1A_" + str(counter)]['Qpos1A']['Labels']['Key'] = key
                            blocks["Qpos1A_Apos1A_" + str(counter)]['Qpos1A']['Labels']['Aspect'] = aspect
                            blocks["Qpos1A_Apos1A_" + str(counter)]['Qpos1A']['Labels']['Polarity'] = str(polarity).lower()
                            
                            blocks["Qpos1A_Apos1A_" + str(counter)]['Apos1A'] = {}
                            blocks["Qpos1A_Apos1A_" + str(counter)]['Apos1A']['Answer'] = Apos1A
                            blocks["Qpos1A_Apos1A_" + str(counter)]['Apos1A']['Labels'] = {}
                            blocks["Qpos1A_Apos1A_" + str(counter)]['Apos1A']['Labels']['Key'] = key
                            blocks["Qpos1A_Apos1A_" + str(counter)]['Apos1A']['Labels']['Aspect'] = aspect
                            blocks["Qpos1A_Apos1A_" + str(counter)]['Apos1A']['Labels']['Polarity'] = str(polarity).lower()
        return(blocks)

    def Oneg1A_Opos1A(item, wrong_aspects, correct_forms, Oneg1A_list, Opos1A_list, dict_AspectSentiment, nlp=nlp_spacy):
        blocks = {}
        counter = 0
        similarity = 0
        aspect_review_polarity_key_list = []
        item_review_list = dict_AspectSentiment.get(item)
        if item_review_list:
            for review_dict in item_review_list:
                for item_reviewer_aspect_key, review_sentiment in (review_dict.items()):
                    key = item_reviewer_aspect_key[0]
                    aspect = item_reviewer_aspect_key[1]
                    review = review_sentiment['review']
                    polarity = review_sentiment['polarity']
                    result, reason = review_aspect_checker(aspect, review, wrong_aspects, correct_forms)
                    if result and aspect not in wrong_aspects and aspect != None:
                        if str(polarity).lower() == 'positive':
                            aspect_review_polarity_key_list.append((aspect, review, polarity, key))

            for review_dict in item_review_list:
                for item_reviewer_aspect_key, review_sentiment in (review_dict.items()):
                    key = item_reviewer_aspect_key[0]
                    aspect = item_reviewer_aspect_key[1]
                    review = review_sentiment['review']
                    polarity = review_sentiment['polarity']
                    result, reason = review_aspect_checker(aspect, review, wrong_aspects, correct_forms)
                    if result and aspect != None and aspect not in wrong_aspects:
                        if str(polarity).lower() == 'negative':
                            aspect = cleaning_process(str(aspect))
                            aspect = re.sub('[^a-zA-Z]+', ' ', str(aspect))
                            nlp_aspect = nlp(str(aspect))
                            
                            review = cleaning_process(str(review))
                            
                            sentence_aspect = retrieve_sentences_from_review(review, aspect)

                            if (len(sentence_aspect) <= 3):
                                continue
                            #sentence_aspect = review
                            
                            Oneg1A = random.choice(Oneg1A_list).format(aspect) + sentence_aspect
                            
                            for aspect_, review_, polarity_, key_ in aspect_review_polarity_key_list:
                                aspect_ = cleaning_process(str(aspect_))
                                aspect_ = re.sub('[^a-zA-Z]+', ' ', str(aspect_))
                                nlp_aspect_ = nlp(str(aspect_))
                                
                                if nlp_aspect_.vector_norm and nlp_aspect.vector_norm:
                                    similarity = nlp_aspect_.similarity(nlp_aspect)
                                    
                                if str(polarity_).lower() == 'positive' and np.logical_or(aspect.lower() in aspect_.lower(), aspect_.lower() in aspect.lower()) and np.logical_or(similarity > 0.75, aspects_check(aspect.lower(), aspect_.lower(), english_vocab)):
                                #if str(polarity_).lower() == 'positive' and np.logical_or(aspect.lower() in aspect_.lower(), aspect_.lower() in aspect.lower()):    
                                    result_, reason_ = review_aspect_checker(aspect_, review_, wrong_aspects, correct_forms)
                                    if result_:
                                        review_ = cleaning_process(str(review_))
                                        
                                        sentence_aspect_ = retrieve_sentences_from_review(review_, aspect_)
                                        #sentence_aspect_ = review_
                                        if (len(sentence_aspect_) <= 3):
                                            continue
                                        
                                        Opos1A = random.choice(Opos1A_list) + sentence_aspect_
                                        
                                        counter += 1
                                        blocks["Oneg1A_Opos1A_" + str(counter)] = {}
                                        blocks["Oneg1A_Opos1A_" + str(counter)]['Oneg1A'] = {}
                                        blocks["Oneg1A_Opos1A_" + str(counter)]['Oneg1A']['Opinion'] = Oneg1A
                                        blocks["Oneg1A_Opos1A_" + str(counter)]['Oneg1A']['Labels'] = {}
                                        blocks["Oneg1A_Opos1A_" + str(counter)]['Oneg1A']['Labels']['Key'] = key
                                        blocks["Oneg1A_Opos1A_" + str(counter)]['Oneg1A']['Labels']['Aspect'] = aspect
                                        blocks["Oneg1A_Opos1A_" + str(counter)]['Oneg1A']['Labels']['Polarity'] = str(polarity).lower()

                                        blocks["Oneg1A_Opos1A_" + str(counter)]['Opos1A'] = {}
                                        blocks["Oneg1A_Opos1A_" + str(counter)]['Opos1A']['Opinion'] = Opos1A
                                        blocks["Oneg1A_Opos1A_" + str(counter)]['Opos1A']['Labels'] = {}
                                        blocks["Oneg1A_Opos1A_" + str(counter)]['Opos1A']['Labels']['Key'] = key_
                                        blocks["Oneg1A_Opos1A_" + str(counter)]['Opos1A']['Labels']['Aspect'] = aspect_
                                        blocks["Oneg1A_Opos1A_" + str(counter)]['Opos1A']['Labels']['Polarity'] = str(polarity_).lower()
        return(blocks)

    def Oneg1A_Opos1B(item, retrieved_items, wrong_aspects, correct_forms, Oneg1A_list, Opos1B_list, dict_AspectSentiment, DF, retrieved=True, also_view=False, nlp=nlp_spacy):
        blocks = {}
        counter = 0
        similarity = 0
        item_1 = item
        other_items_list = None
        if retrieved:
            other_items_list = [i for i in retrieved_items if i != item_1]
        elif also_view:
            if DF.query("asin == @item_1").also_view.values.size > 0:
                other_items_list = DF.query("asin == @item_1").also_view.values[0]
            else:
                other_items_list = None
        if other_items_list:
            aspect_review_polarity_key_lists = []
            for item_2 in other_items_list:
                item_2_review_list = dict_AspectSentiment.get(item_2)
                if item_2_review_list:
                    aspect_review_polarity_key_list = []
                    for item_2_review_dict in item_2_review_list:
                        for item_2_reviewer_aspect_key, item_2_review_sentiment in (item_2_review_dict.items()):
                            item_2_key = item_2_reviewer_aspect_key[0]
                            item_2_aspect = item_2_reviewer_aspect_key[1]
                            item_2_review = item_2_review_sentiment['review']
                            item_2_polarity = item_2_review_sentiment['polarity']
                            if item_2_aspect not in wrong_aspects and item_2_aspect != None:
                                if str(item_2_polarity).lower() == 'positive':
                                    aspect_review_polarity_key_list.append((item_2, item_2_aspect, item_2_review, item_2_polarity, item_2_key))

                    aspect_review_polarity_key_lists.append(aspect_review_polarity_key_list)


            item_1_review_list = dict_AspectSentiment.get(item_1)
            if item_1_review_list:
                for item_1_review_dict in item_1_review_list:
                    for item_1_reviewer_aspect_key, item_1_review_sentiment in (item_1_review_dict.items()):
                        item_1_key = item_1_reviewer_aspect_key[0]
                        item_1_aspect = item_1_reviewer_aspect_key[1]
                        item_1_review = item_1_review_sentiment['review']
                        item_1_polarity = item_1_review_sentiment['polarity']
                        item_1_result, item_1_reason = review_aspect_checker(item_1_aspect, item_1_review, wrong_aspects, correct_forms)
                        if item_1_result and item_1_aspect != None and item_1_aspect not in wrong_aspects:
                            if str(item_1_polarity).lower() == 'negative':
                                item_1_aspect = cleaning_process(item_1_aspect)
                                item_1_aspect = re.sub('[^a-zA-Z]+', ' ', str(item_1_aspect))
                                nlp_item_1_aspect = nlp(str(item_1_aspect))
                                
                                item_1_review = cleaning_process(str(item_1_review))

                                item_1_sentence_aspect = retrieve_sentences_from_review(item_1_review, item_1_aspect)
                                #item_1_sentence_aspect = item_1_review

                                if (len(item_1_sentence_aspect) <= 3):
                                    continue
                                
                                Oneg1A = random.choice(Oneg1A_list).format(item_1_aspect) + item_1_sentence_aspect

                                for item_aspect_review_polarity_key in aspect_review_polarity_key_lists:
                                    for item_, aspect_, review_, polarity_, key_ in item_aspect_review_polarity_key:
                                        aspect_ = cleaning_process(aspect_)
                                        nlp_aspect_ = nlp(str(aspect_))

                                        if nlp_aspect_.vector_norm and nlp_item_1_aspect.vector_norm:
                                            similarity = nlp_aspect_.similarity(nlp_item_1_aspect)

                                        if str(polarity_).lower() == 'positive' and np.logical_or(item_1_aspect.lower() in aspect_.lower(), aspect_.lower() in item_1_aspect.lower()) and np.logical_or(similarity > 0.75, aspects_check(item_1_aspect.lower(), aspect_.lower(), english_vocab)):
                                        #if str(polarity_).lower() == 'positive' and np.logical_or(item_1_aspect.lower() in aspect_.lower(), aspect_.lower() in item_1_aspect.lower()):    
                                            result_, reason_ = review_aspect_checker(aspect_, review_, wrong_aspects, correct_forms)
                                            if result_:
                                                review_ = cleaning_process(str(review_))

                                                sentence_aspect_ = retrieve_sentences_from_review(review_, aspect_)
                                                if (len(sentence_aspect_) <= 3):
                                                    continue
                                                #sentence_aspect_ = review_
                                                
                                                counter += 1
                                                Opos1B = random.choice(Opos1B_list).format(item_1_aspect, item_) + sentence_aspect_

                                                blocks["Oneg1A_Opos1B_" + str(counter)] = {}
                                                blocks["Oneg1A_Opos1B_" + str(counter)]['Oneg1A'] = {}
                                                blocks["Oneg1A_Opos1B_" + str(counter)]['Oneg1A']['Opinion'] = Oneg1A
                                                blocks["Oneg1A_Opos1B_" + str(counter)]['Oneg1A']['Labels'] = {}
                                                blocks["Oneg1A_Opos1B_" + str(counter)]['Oneg1A']['Labels']['Key'] = item_1_key
                                                blocks["Oneg1A_Opos1B_" + str(counter)]['Oneg1A']['Labels']['Aspect'] = item_1_aspect
                                                blocks["Oneg1A_Opos1B_" + str(counter)]['Oneg1A']['Labels']['Polarity'] = str(item_1_polarity).lower()

                                                blocks["Oneg1A_Opos1B_" + str(counter)]['Opos1B'] = {}
                                                blocks["Oneg1A_Opos1B_" + str(counter)]['Opos1B']['Opinion'] = Opos1B
                                                blocks["Oneg1A_Opos1B_" + str(counter)]['Opos1B']['Labels'] = {}
                                                blocks["Oneg1A_Opos1B_" + str(counter)]['Opos1B']['Labels']['Key'] = key_
                                                blocks["Oneg1A_Opos1B_" + str(counter)]['Opos1B']['Labels']['Aspect'] = aspect_
                                                blocks["Oneg1A_Opos1B_" + str(counter)]['Opos1B']['Labels']['Polarity'] = str(polarity_).lower()
                                        
        return(blocks)

    def Oneg1A_Opos2A(item, wrong_aspects, correct_forms, Oneg1A_list, Opos1B_list, dict_AspectSentiment, restricted_version=True, nlp=nlp_spacy):
        blocks = {}
        counter = 0
        similarity = 0
        aspect_review_polarity_key_list = []
        item_review_list = dict_AspectSentiment.get(item)
        if item_review_list:
            for review_dict in item_review_list:
                for item_reviewer_aspect_key, review_sentiment in (review_dict.items()):
                    key = item_reviewer_aspect_key[0]
                    aspect = item_reviewer_aspect_key[1]
                    review = review_sentiment['review']
                    polarity = review_sentiment['polarity']
                    if aspect not in wrong_aspects and aspect != None:
                        if str(polarity).lower() == 'positive':
                            aspect_review_polarity_key_list.append((aspect, review, polarity, key))

            for review_dict in item_review_list:
                for item_reviewer_aspect_key, review_sentiment in (review_dict.items()):
                    key = item_reviewer_aspect_key[0]
                    aspect = item_reviewer_aspect_key[1]
                    review = review_sentiment['review']
                    polarity = review_sentiment['polarity']
                    result, reason = review_aspect_checker(aspect, review, wrong_aspects, correct_forms)
                    if result and aspect != None and aspect not in wrong_aspects:
                        if str(polarity).lower() == 'negative':
                            aspect = cleaning_process(aspect)
                            aspect = re.sub('[^a-zA-Z]+', ' ', str(aspect))
                            
                            review = cleaning_process(review)
                            
                            sentence_aspect = retrieve_sentences_from_review(review, aspect)

                            if (len(sentence_aspect) <= 3):
                                continue
                            #sentence_aspect = review
                            
                            Oneg1A = random.choice(Oneg1A_list).format(aspect) + sentence_aspect
                            
                            check = False
                            # We agree with the user only when there is a positive review for the aspect, mentioned by user
                            if restricted_version == True:
                                nlp_aspect = nlp(str(aspect))
                                for aspect_, review_, polarity_, key_ in aspect_review_polarity_key_list:
                                    if check == False:
                                        aspect_ = cleaning_process(str(aspect_))
                                        aspect_ = re.sub('[^a-zA-Z]+', ' ', str(aspect_))
                                        nlp_aspect_ = nlp(str(aspect_))

                                        if nlp_aspect_.vector_norm and nlp_aspect.vector_norm:
                                            similarity = nlp_aspect_.similarity(nlp_aspect)

                                        if str(polarity_).lower() == 'positive' and \
                                        np.logical_or(aspect.lower() in aspect_.lower(), aspect_.lower() in aspect.lower()) and \
                                        np.logical_or(similarity > 0.75, aspects_check(aspect.lower(), aspect_.lower(), english_vocab)):
                                            # POSITIVE_SENT = retrieve_sentences_from_review(review_, aspect_)
                                            check = True
                                
                                if check == True:
                                    for aspect_, review_, polarity_, key_ in aspect_review_polarity_key_list:
                                        aspect_ = cleaning_process(aspect_)
                                        if str(polarity_).lower() == 'positive':
                                            result_, reason_ = review_aspect_checker(aspect_, review_, wrong_aspects, correct_forms)
                                            if result_:
                                                review_ = cleaning_process(review_)

                                                sentence_aspect_ = retrieve_sentences_from_review(review_, aspect_)
                                                #sentence_aspect_ = review_

                                                if (len(sentence_aspect_) <= 3):
                                                    continue

                                                Opos2A = random.choice(Opos2A_list).format(aspect, aspect_) + sentence_aspect_

                                                counter += 1
                                                blocks["Oneg1A_Opos2A_" + str(counter)] = {}
                                                blocks["Oneg1A_Opos2A_" + str(counter)]['Oneg1A'] = {}
                                                blocks["Oneg1A_Opos2A_" + str(counter)]['Oneg1A']['Opinion'] = Oneg1A
                                                blocks["Oneg1A_Opos2A_" + str(counter)]['Oneg1A']['Labels'] = {}
                                                blocks["Oneg1A_Opos2A_" + str(counter)]['Oneg1A']['Labels']['Key'] = key
                                                blocks["Oneg1A_Opos2A_" + str(counter)]['Oneg1A']['Labels']['Aspect'] = aspect
                                                blocks["Oneg1A_Opos2A_" + str(counter)]['Oneg1A']['Labels']['Polarity'] = str(polarity).lower()

                                                blocks["Oneg1A_Opos2A_" + str(counter)]['Opos2A'] = {}
                                                blocks["Oneg1A_Opos2A_" + str(counter)]['Opos2A']['Opinion'] = Opos2A
                                                blocks["Oneg1A_Opos2A_" + str(counter)]['Opos2A']['Labels'] = {}
                                                blocks["Oneg1A_Opos2A_" + str(counter)]['Opos2A']['Labels']['Key'] = key_
                                                blocks["Oneg1A_Opos2A_" + str(counter)]['Opos2A']['Labels']['Aspect'] = aspect_
                                                blocks["Oneg1A_Opos2A_" + str(counter)]['Opos2A']['Labels']['Polarity'] = str(polarity_).lower()
                            
                            else:
                                for aspect_, review_, polarity_, key_ in aspect_review_polarity_key_list:
                                    aspect_ = cleaning_process(aspect_)
                                    if str(polarity_).lower() == 'positive':
                                        result_, reason_ = review_aspect_checker(aspect_, review_, wrong_aspects, correct_forms)
                                        if result_:
                                            review_ = cleaning_process(review_)

                                            sentence_aspect_ = retrieve_sentences_from_review(review_, aspect_)

                                            if (len(sentence_aspect_) <= 3):
                                                continue
                                            #sentence_aspect_ = review_

                                            Opos2A = random.choice(Opos2A_list).format(aspect, aspect_) + sentence_aspect_
                                        
                                            counter += 1
                                            blocks["Oneg1A_Opos2A_" + str(counter)] = {}
                                            blocks["Oneg1A_Opos2A_" + str(counter)]['Oneg1A'] = {}
                                            blocks["Oneg1A_Opos2A_" + str(counter)]['Oneg1A']['Opinion'] = Oneg1A
                                            blocks["Oneg1A_Opos2A_" + str(counter)]['Oneg1A']['Labels'] = {}
                                            blocks["Oneg1A_Opos2A_" + str(counter)]['Oneg1A']['Labels']['Key'] = key
                                            blocks["Oneg1A_Opos2A_" + str(counter)]['Oneg1A']['Labels']['Aspect'] = aspect
                                            blocks["Oneg1A_Opos2A_" + str(counter)]['Oneg1A']['Labels']['Polarity'] = str(polarity).lower()

                                            blocks["Oneg1A_Opos2A_" + str(counter)]['Opos2A'] = {}
                                            blocks["Oneg1A_Opos2A_" + str(counter)]['Opos2A']['Opinion'] = Opos2A
                                            blocks["Oneg1A_Opos2A_" + str(counter)]['Opos2A']['Labels'] = {}
                                            blocks["Oneg1A_Opos2A_" + str(counter)]['Opos2A']['Labels']['Key'] = key_
                                            blocks["Oneg1A_Opos2A_" + str(counter)]['Opos2A']['Labels']['Aspect'] = aspect_
                                            blocks["Oneg1A_Opos2A_" + str(counter)]['Opos2A']['Labels']['Polarity'] = str(polarity_).lower()
                                        
        return(blocks)

    # @Vahid: it will find the pairs for every item, However, we need something like what I have in function: Oneg1A_Opos1B. i.e. Retrieved_items and also_view_items
    def Opos1B_Opos1B2(item, wrong_aspects, correct_forms, Opos1B1_list, Opos1B2_list,
                    dict_AspectSentiment, only_aggrement=True, aggrement_and_more=True, nlp=nlp_spacy):
        blocks = {}
        counter = 0
        similarity = 0
        aspect_review_polarity_key_list = []
        item_review_list = dict_AspectSentiment.get(item)
        if item_review_list and aggrement_and_more:
            for review_dict in item_review_list:
                for item_reviewer_aspect_key, review_sentiment in (review_dict.items()):
                    key = item_reviewer_aspect_key[0]
                    aspect = item_reviewer_aspect_key[1]
                    review = review_sentiment['review']
                    polarity = review_sentiment['polarity']
                    result, reason = review_aspect_checker(aspect, review, wrong_aspects, correct_forms)
                    if result and aspect not in wrong_aspects and aspect != None:
                        if str(polarity).lower() == 'positive':
                            aspect_review_polarity_key_list.append((aspect, review, polarity, key))
                            
        if item_review_list:
            for review_dict in item_review_list:
                for item_reviewer_aspect_key, review_sentiment in (review_dict.items()):
                    key = item_reviewer_aspect_key[0]
                    aspect = item_reviewer_aspect_key[1]
                    review = review_sentiment['review']
                    polarity = review_sentiment['polarity']
                    result, reason = review_aspect_checker(aspect, review, wrong_aspects, correct_forms)
                    if result and aspect != None and aspect not in wrong_aspects:
                        if str(polarity).lower() == 'positive':
                            aspect = cleaning_process(str(aspect))
                            aspect = re.sub('[^a-zA-Z]+', ' ', str(aspect))
                            
                            review = cleaning_process(str(review))
                            
                            sentence_aspect = retrieve_sentences_from_review(review, aspect)
                            #sentence_aspect = review
                            if (len(sentence_aspect) <= 3):
                                continue
                            
                            Opos1B = random.choice(Opos1B1_list).format(item, aspect) + sentence_aspect
                            counter += 1
                            
                            if only_aggrement:
                                
                                Opos1B2 = random.choice(Opos1B2_list)
                                
                                blocks["Opos1B_Opos1B2_" + str(counter)] = {}
                                blocks["Opos1B_Opos1B2_" + str(counter)]['Opos1B'] = {}
                                blocks["Opos1B_Opos1B2_" + str(counter)]['Opos1B']['Opinion'] = Opos1B
                                blocks["Opos1B_Opos1B2_" + str(counter)]['Opos1B']['Labels'] = {}
                                blocks["Opos1B_Opos1B2_" + str(counter)]['Opos1B']['Labels']['Key'] = key
                                blocks["Opos1B_Opos1B2_" + str(counter)]['Opos1B']['Labels']['Aspect'] = aspect
                                blocks["Opos1B_Opos1B2_" + str(counter)]['Opos1B']['Labels']['Polarity'] = str(polarity).lower()

                                blocks["Opos1B_Opos1B2_" + str(counter)]['Opos1B2'] = {}
                                blocks["Opos1B_Opos1B2_" + str(counter)]['Opos1B2']['Opinion'] = Opos1B2
                                blocks["Opos1B_Opos1B2_" + str(counter)]['Opos1B2']['Labels'] = {}
                                blocks["Opos1B_Opos1B2_" + str(counter)]['Opos1B2']['Labels']['Key'] = key
                                blocks["Opos1B_Opos1B2_" + str(counter)]['Opos1B2']['Labels']['Aspect'] = aspect
                                blocks["Opos1B_Opos1B2_" + str(counter)]['Opos1B2']['Labels']['Polarity'] = str(polarity).lower()
                                
                            elif aggrement_and_more:
                                nlp_aspect = nlp(str(aspect))
                                
                                for aspect_, review_, polarity_, key_ in aspect_review_polarity_key_list:
                                    aspect_ = cleaning_process(str(aspect_))
                                    aspect_ = re.sub('[^a-zA-Z]+', ' ', str(aspect_))
                                    nlp_aspect_ = nlp(str(aspect_))

                                    if nlp_aspect_.vector_norm and nlp_aspect.vector_norm:
                                        similarity = nlp_aspect_.similarity(nlp_aspect)

                                    if str(polarity_).lower() == 'positive' and np.logical_or(aspect.lower() in aspect_.lower(), aspect_.lower() in aspect.lower()) \
                                    and np.logical_or(similarity > 0.75, aspects_check(aspect.lower(), aspect_.lower(), english_vocab)) \
                                    and review != cleaning_process(str(review_)):
                                    #if str(polarity_).lower() == 'positive' and np.logical_or(aspect.lower() in aspect_.lower(), aspect_.lower() in aspect.lower()):    
                                        result_, reason_ = review_aspect_checker(aspect_, review_, wrong_aspects, correct_forms)
                                        if result_:
                                            review_ = cleaning_process(str(review_))

                                            sentence_aspect_ = retrieve_sentences_from_review(review_, aspect_)
                                            #sentence_aspect_ = review_

                                            if (len(sentence_aspect_) <= 3):
                                                continue
                                            
                                            Opos1B2 = random.choice(Opos1B2_list) + " because " + sentence_aspect_

                                            counter += 1
                                            blocks["Opos1B_Opos1B2_" + str(counter)] = {}
                                            blocks["Opos1B_Opos1B2_" + str(counter)]['Opos1B'] = {}
                                            blocks["Opos1B_Opos1B2_" + str(counter)]['Opos1B']['Opinion'] = Opos1B
                                            blocks["Opos1B_Opos1B2_" + str(counter)]['Opos1B']['Labels'] = {}
                                            blocks["Opos1B_Opos1B2_" + str(counter)]['Opos1B']['Labels']['Key'] = key
                                            blocks["Opos1B_Opos1B2_" + str(counter)]['Opos1B']['Labels']['Aspect'] = aspect
                                            blocks["Opos1B_Opos1B2_" + str(counter)]['Opos1B']['Labels']['Polarity'] = str(polarity).lower()

                                            blocks["Opos1B_Opos1B2_" + str(counter)]['Opos1B2'] = {}
                                            blocks["Opos1B_Opos1B2_" + str(counter)]['Opos1B2']['Opinion'] = Opos1B2
                                            blocks["Opos1B_Opos1B2_" + str(counter)]['Opos1B2']['Labels'] = {}
                                            blocks["Opos1B_Opos1B2_" + str(counter)]['Opos1B2']['Labels']['Key'] = key_
                                            blocks["Opos1B_Opos1B2_" + str(counter)]['Opos1B2']['Labels']['Aspect'] = aspect_
                                            blocks["Opos1B_Opos1B2_" + str(counter)]['Opos1B2']['Labels']['Polarity'] = str(polarity_).lower()
        return(blocks)

    # @Vahid: it will find the pairs for every item, However, we need something like what I have in function: Oneg1A_Opos1B. i.e. Retrieved_items and also_view_items
    def Opos1B_Opos2B(item, wrong_aspects, correct_forms, Opos1B1_list, Opos2B_list, dict_AspectSentiment, nlp=nlp_spacy):
        blocks = {}
        counter = 0
        similarity = 0
        aspect_review_polarity_key_list = []
        item_review_list = dict_AspectSentiment.get(item)
        if item_review_list:
            for review_dict in item_review_list:
                for item_reviewer_aspect_key, review_sentiment in (review_dict.items()):
                    key = item_reviewer_aspect_key[0]
                    aspect = item_reviewer_aspect_key[1]
                    review = review_sentiment['review']
                    polarity = review_sentiment['polarity']
                    result, reason = review_aspect_checker(aspect, review, wrong_aspects, correct_forms)
                    if result and aspect not in wrong_aspects and aspect != None:
                        if str(polarity).lower() == 'positive':
                            aspect_review_polarity_key_list.append((aspect, review, polarity, key))
                            
        if item_review_list:
            for review_dict in item_review_list:
                for item_reviewer_aspect_key, review_sentiment in (review_dict.items()):
                    key = item_reviewer_aspect_key[0]
                    aspect = item_reviewer_aspect_key[1]
                    review = review_sentiment['review']
                    polarity = review_sentiment['polarity']
                    result, reason = review_aspect_checker(aspect, review, wrong_aspects, correct_forms)
                    if result and aspect != None and aspect not in wrong_aspects:
                        if str(polarity).lower() == 'positive':
                            aspect = cleaning_process(str(aspect))
                            aspect = re.sub('[^a-zA-Z]+', ' ', str(aspect))
                            
                            review = cleaning_process(str(review))
                            
                            sentence_aspect = retrieve_sentences_from_review(review, aspect)
                            #sentence_aspect = review
                            if (len(sentence_aspect) <= 3):
                                continue
                            
                            counter += 1
                            Opos1B = random.choice(Opos1B1_list).format(item, aspect) + sentence_aspect
        
                            for aspect_, review_, polarity_, key_ in aspect_review_polarity_key_list:
                                aspect_ = cleaning_process(str(aspect_))
                                aspect_ = re.sub('[^a-zA-Z]+', ' ', str(aspect_))

                                if str(polarity_).lower() == 'positive' and review != cleaning_process(str(review_)): 
                                    result_, reason_ = review_aspect_checker(aspect_, review_, wrong_aspects, correct_forms)
                                    if result_:
                                        review_ = cleaning_process(str(review_))

                                        sentence_aspect_ = retrieve_sentences_from_review(review_, aspect_)
                                        #sentence_aspect_ = review_
                                        if (len(sentence_aspect_) <= 3):
                                            continue

                                        Opos2B = random.choice(Opos2B_list).format(aspect_) + sentence_aspect_

                                        counter += 1
                                        blocks["Opos1B_Opos2B_" + str(counter)] = {}
                                        blocks["Opos1B_Opos2B_" + str(counter)]['Opos1B'] = {}
                                        blocks["Opos1B_Opos2B_" + str(counter)]['Opos1B']['Opinion'] = Opos1B
                                        blocks["Opos1B_Opos2B_" + str(counter)]['Opos1B']['Labels'] = {}
                                        blocks["Opos1B_Opos2B_" + str(counter)]['Opos1B']['Labels']['Key'] = key
                                        blocks["Opos1B_Opos2B_" + str(counter)]['Opos1B']['Labels']['Aspect'] = aspect
                                        blocks["Opos1B_Opos2B_" + str(counter)]['Opos1B']['Labels']['Polarity'] = str(polarity).lower()

                                        blocks["Opos1B_Opos2B_" + str(counter)]['Opos2B'] = {}
                                        blocks["Opos1B_Opos2B_" + str(counter)]['Opos2B']['Opinion'] = Opos2B
                                        blocks["Opos1B_Opos2B_" + str(counter)]['Opos2B']['Labels'] = {}
                                        blocks["Opos1B_Opos2B_" + str(counter)]['Opos2B']['Labels']['Key'] = key_
                                        blocks["Opos1B_Opos2B_" + str(counter)]['Opos2B']['Labels']['Aspect'] = aspect_
                                        blocks["Opos1B_Opos2B_" + str(counter)]['Opos2B']['Labels']['Polarity'] = str(polarity_).lower()
        return(blocks)

    # @Vahid: it will find the pairs for every item, However, we need something like what I have in function: Oneg1A_Opos1B. i.e. Retrieved_items and also_view_items
    def Opos1B_Oneg2B(item, wrong_aspects, correct_forms, Opos1B1_list, Oneg2B_list, dict_AspectSentiment, nlp=nlp_spacy):
        blocks = {}
        counter = 0
        similarity = 0
        aspect_review_polarity_key_list = []
        item_review_list = dict_AspectSentiment.get(item)
        if item_review_list:
            for review_dict in item_review_list:
                for item_reviewer_aspect_key, review_sentiment in (review_dict.items()):
                    key = item_reviewer_aspect_key[0]
                    aspect = item_reviewer_aspect_key[1]
                    review = review_sentiment['review']
                    polarity = review_sentiment['polarity']
                    result, reason = review_aspect_checker(aspect, review, wrong_aspects, correct_forms)
                    if result and aspect not in wrong_aspects and aspect != None:
                        if str(polarity).lower() == 'negative':
                            aspect_review_polarity_key_list.append((aspect, review, polarity, key))
                            
        if item_review_list:
            for review_dict in item_review_list:
                for item_reviewer_aspect_key, review_sentiment in (review_dict.items()):
                    key = item_reviewer_aspect_key[0]
                    aspect = item_reviewer_aspect_key[1]
                    review = review_sentiment['review']
                    polarity = review_sentiment['polarity']
                    result, reason = review_aspect_checker(aspect, review, wrong_aspects, correct_forms)
                    if result and aspect != None and aspect not in wrong_aspects:
                        if str(polarity).lower() == 'positive':
                            aspect = cleaning_process(str(aspect))
                            aspect = re.sub('[^a-zA-Z]+', ' ', str(aspect))
                            
                            review = cleaning_process(str(review))
                            
                            sentence_aspect = retrieve_sentences_from_review(review, aspect)
                            #sentence_aspect = review
                            if (len(sentence_aspect) <= 3):
                                continue
                            
                            counter += 1
                            Opos1B = random.choice(Opos1B1_list).format(item, aspect) + sentence_aspect
        
                            for aspect_, review_, polarity_, key_ in aspect_review_polarity_key_list:
                                aspect_ = cleaning_process(str(aspect_))
                                aspect_ = re.sub('[^a-zA-Z]+', ' ', str(aspect_))

                                if str(polarity_).lower() == 'negative' and review != cleaning_process(str(review_)): 
                                    result_, reason_ = review_aspect_checker(aspect_, review_, wrong_aspects, correct_forms)
                                    if result_:
                                        review_ = cleaning_process(str(review_))

                                        sentence_aspect_ = retrieve_sentences_from_review(review_, aspect_)
                                        #sentence_aspect_ = review_
                                        if (len(sentence_aspect_) <= 3):
                                            continue

                                        Oneg2B = random.choice(Oneg2B_list).format(aspect_) + sentence_aspect_

                                        counter += 1
                                        blocks["Opos1B_Oneg2B_" + str(counter)] = {}
                                        blocks["Opos1B_Oneg2B_" + str(counter)]['Opos1B'] = {}
                                        blocks["Opos1B_Oneg2B_" + str(counter)]['Opos1B']['Opinion'] = Opos1B
                                        blocks["Opos1B_Oneg2B_" + str(counter)]['Opos1B']['Labels'] = {}
                                        blocks["Opos1B_Oneg2B_" + str(counter)]['Opos1B']['Labels']['Key'] = key
                                        blocks["Opos1B_Oneg2B_" + str(counter)]['Opos1B']['Labels']['Aspect'] = aspect
                                        blocks["Opos1B_Oneg2B_" + str(counter)]['Opos1B']['Labels']['Polarity'] = str(polarity).lower()

                                        blocks["Opos1B_Oneg2B_" + str(counter)]['Oneg2B'] = {}
                                        blocks["Opos1B_Oneg2B_" + str(counter)]['Oneg2B']['Opinion'] = Oneg2B
                                        blocks["Opos1B_Oneg2B_" + str(counter)]['Oneg2B']['Labels'] = {}
                                        blocks["Opos1B_Oneg2B_" + str(counter)]['Oneg2B']['Labels']['Key'] = key_
                                        blocks["Opos1B_Oneg2B_" + str(counter)]['Oneg2B']['Labels']['Aspect'] = aspect_
                                        blocks["Opos1B_Oneg2B_" + str(counter)]['Oneg2B']['Labels']['Polarity'] = str(polarity_).lower()
        return(blocks)



    # Inputs from Vahid

    dict_AspectSentiment = pd.read_pickle('AspectSentiment_Results_new_filtered.pkl')

    wrong_aspects = ["it", "i", "not", "sis", "works", "snaps", 'iphone', 'nokia', 'thanks', 'product', 'zero key', 'smart phones', 'smart phone']
    correct_forms = ['bluetooth']

    Q1A_list = ["What do you think about its {}?", "May I know your opinion on its {}?",
                "What about its {}?", "Do you have any views on its {}?",
                "Could you tell me your opinion on its {}?",
                "Do you have any opinion about its {}?", "In your honest opinion, how is its {}?",
                "Can you give me your thoughts on its {}?", "I’d like to know your views on its {}.",
                "From your point of view, how is the {}?",
                "I’d be very interested to know your views on its {}.", "What do you think about its {}?"
            ]

    Oneg1A_list = ["I heard that ",
                "I was told by one of my friends that ",
                "As far as I know, ",
                "What I know about it is that ",
                "I saw in a review that "]

    Opos1A_list = ["No, I don't think so, because ",
                "Let me disagree with you, because ",
                "That is partly true, but ",
                "I understand your point, but "]

    Opos1B_list = ["If {} is important for you, we can offer this item: {} ", 
                "If {} is a crucial feature for you, we have this item: {} "]

    Opos2A_list = ["I understand what you’re saying about its {}. However, there are other good points about this phone: ",
                "Sorry, but I don’t agree with you about its {}. Also, I need to mention that ",
                "I can understand your point about its {}. I still think it is a good choice because "
                ]

    Opos1B1_list = ["I heard about this phone {} that ",
                "I was told by one of my friends about this phone {} that ",
                "I was wondering what you think about this phone {}. It might be a good alternative because "]

    Opos1B2_list = ["Yes, it's true! This phone is also a good choice",
                    "Yes, That's so true. This phone is also a good choice",
                    "Yes, That's for sure. This phone is also a good choice",
                    "Yes, I think so too. This phone is also a good choice",
                    "Yes, That is what I think too. This phone is also a good choice",
                    "Yes! I agree with you. This phone is also a good choice",
                    "Yes, I agree with you about it. This phone is also a good choice",
                    "Yes, That's exactly what I know about it. This phone is also a good choice"]

    Opos2B_list = ["Yes, it's true! This phone is also a good choice and even I can tell you something interesting about this phone and its {} that ",
                    "Yes, That's so true. This phone is also a good choice and I would mention something about the {} of this phone that ",
                    "Yes, That's for sure. This phone is also a good choice and even I can tell you something interesting about this phone and its {} that ",
                    "Yes, I think so too. This phone is also a good choice and I would mention something about the {} of this phone that ",
                    "Yes, That is what I think too. This phone is also a good choice and even I can tell you something interesting about this phone and its {} that ",
                    "Yes! I agree with you. This phone is also a good choice and I would mention something about the {} of this phone that ",
                    "Yes, I agree with you about it. This phone is also a good choice and even I can tell you something interesting about this phone and its {} that ",
                    "Yes, That's exactly what I know about it. This phone is also a good choice and I would mention something about the {} of this phone that "]

    Oneg2B_list = ["Yes, it's true! This phone might be a good choice but you should know about its {} that ",
                    "Yes, That's so true. This phone can be also a good choice but I should say about the {} of this phone that ",
                    "Yes, That's for sure. This phone is also a good choice However about the {} of this phone I should say that ",
                    "Yes, I completely agree with you. This phone might be a good choice but you should know about its {} that ",
                    "Yes, I totally agree with you. This phone can be also a good choice but I should say about the {} of this phone that ",
                    "Yes! I agree with you. This phone is also a good choice However about the {} of this phone I should say that ",
                    "Yes, I agree with you about it. This phone can be also a good choice but I should say about the {} of this phone that ",
                    "Yes, That's exactly what I know about it. However I should say something about the {} of this phone that "]


    def getBlocks(item):
        # only get the blocks about this article for now (no cross referencing)

        blocks = {
            'Qpos1A_Apos1A': Qpos1A_Apos1A(item, wrong_aspects, correct_forms, Q1A_list, dict_AspectSentiment),
            'Oneg1A_Opos1A': Oneg1A_Opos1A(item, wrong_aspects, correct_forms, Oneg1A_list, Opos1A_list, dict_AspectSentiment, nlp_spacy),
            'Oneg1A_Opos2A_restricted': Oneg1A_Opos2A(item, wrong_aspects, correct_forms, Oneg1A_list, Opos1B_list, dict_AspectSentiment, restricted_version=True, nlp=nlp_spacy),
            'Oneg1A_Opos2A_unrestricted': Oneg1A_Opos2A(item, wrong_aspects, correct_forms, Oneg1A_list, Opos1B_list, dict_AspectSentiment, restricted_version=False, nlp=nlp_spacy),
            #'Opos1B_Opos1B2_only_agree': Opos1B_Opos1B2(item, wrong_aspects, correct_forms, Opos1B1_list, Opos1B2_list,
            #                                              dict_AspectSentiment, only_aggrement=True, aggrement_and_more=False, nlp=nlp_spacy),
            #'Opos1B_Opos1B2_more': Opos1B_Opos1B2(item, wrong_aspects, correct_forms, Opos1B1_list, Opos1B2_list,
            #                                                  dict_AspectSentiment, only_aggrement=False, aggrement_and_more=True, nlp=nlp_spacy),
            #'Opos1B_Opos2B': Opos1B_Opos2B(item, wrong_aspects, correct_forms, Opos1B1_list, Opos2B_list, dict_AspectSentiment, nlp=nlp_spacy),
            #'Opos1B_Oneg2B': Opos1B_Oneg2B(item, wrong_aspects, correct_forms, Opos1B1_list, Oneg2B_list, dict_AspectSentiment, nlp=nlp_spacy),
        }

        return blocks

    def isBadSentence(sentence):
        sentenceLen = len(sentence)
        return (sentenceLen <= 5) or (sentenceLen >= 300)

    def isBadPart(part):
        # Question, Opinion, Answer
        fields = ['Question', 'Opinion', 'Answer']
        for field in fields:
            if (field in part):
                return isBadSentence(part[field])
        # we should never get here. if we do, the field is unknown.
        return False

    def filterBlocks(blocks):
        # remove blocks that are too long or of low quality
        for blockType, blockDict in list(blocks.items()):
            for blockId, blockEntry in list(blockDict.items()):
                for partName, part in blockEntry.items():
                    if (isBadPart(part)):
                        del blockDict[blockId]
                        break

    def buildConversations(blocks, item):
        # build a conversation from generated blocks; all for a single item

        # 'Qpos1A_Apos1A'
        # 'Oneg1A_Opos1A'
        # 'Oneg1A_Opos2A_restricted' 
        #   restricuted == We agree with the user only when there is a positive review for the aspect
        # 'Oneg1A_Opos2A_unrestricted'

        whitespaceRegex = re.compile(r'\s+')
        metaDf = pd.read_csv('cell_phones_df_filtered.csv', index_col=0)
        phoneFeatures = pd.read_pickle('phoneFeatures.pkl')
        
        qaBlocks = blocks['Qpos1A_Apos1A']
        qaBlockAspectMap = defaultdict(list)

        for blockEntry in qaBlocks.values():
            aspect = ''
            for block in blockEntry.values():
                aspect = block['Labels']['Aspect'].lower()
                break
            # aspect should be the same for question and answer.
            if (aspect and len(aspect) > 0):
                qaBlockAspectMap[aspect].append(blockEntry)

        qaAspects = set(qaBlockAspectMap.keys())  
        usedAspects = set()
        usedReviews = set()

        def isAspectNew(aspect, knownAspects):
            aspect = aspect.lower()
            aspect = whitespaceRegex.sub('', aspect)

            for knownAspect in knownAspects:
                if (aspect in knownAspect) or (knownAspect in aspect):
                    return False
            
            return True

        # generate up to 3 conversations for this product
        dfs = []
        try:
            for j in range(3):
                newDf = []

                # 1. add greeting

                botGreetings = [
                    '{}, how can I {} you?',
                    '{}, what can I {} you with?'
                ]
                hiList = ['Hi', 'Hello', 'Hey']
                hi = random.choice(hiList)
                helpVerb = random.choice(['help', 'assist'])
                botGreeting = random.choice(botGreetings)
                botGreeting = botGreeting.format(hi, helpVerb)

                newDf.append({
                    'text': botGreeting,
                    'sender': 'bot',
                    'item': item,
                    'isConst': False
                })

                hi = random.choice(hiList)
                buyVerb = random.choice(['buy', 'purchase'])
                userGreetings = [
                    '{}, I want to {} a new phone.'.format(hi, buyVerb),
                    '{}, I am looking for a new phone.'.format(hi)
                ]
                newDf.append({
                    'text': random.choice(userGreetings),
                    'sender': 'user',
                    'item': item
                })

                # 2. ask for a preference on price, OS or brand
                preferenceQuestions = [
                    'Do you have any preference on the brand, operating system or price  of the phone?',
                    'What kind of phone are you looking for?',
                    'Should the phone have any particular features or price?'
                ]
                newDf.append({
                    'text': random.choice(preferenceQuestions),
                    'sender': 'bot',
                    'item': item,
                    'isConst': False
                })

                possiblePreferences = []
                meta = metaDf.loc[metaDf['asin'] == item].iloc[0]
                features = phoneFeatures.get(item, {})
                price = meta['price']
                phoneOs = features.get('os', 'No')
                brand = features.get('brand', '')

                if (len(brand) > 0 and brand != 'No'):
                    possiblePreferences.append('The phone should be made by {}.'.format(brand))
                    if (brand.lower() == 'apple'):
                        phoneOs = 'iOS'
                if (price != np.nan and price >= 30):
                    targetPrice = 420
                    if (price <= 100):
                        targetPrice = np.round(price, -1)
                    else:
                        targetPrice = np.round(price, -2)
                    targetPrice = int(targetPrice)
                    possiblePreferences.append('I can spend about {}$.'.format(targetPrice))
                if (phoneOs != 'No' and len(phoneOs) > 0):
                    possiblePreferences.append("The phone's operating system should be {}.".format(phoneOs))
                
                numOfPreferences = min(random.randint(1,2), len(possiblePreferences))
                selectedPreferences = random.sample(possiblePreferences, numOfPreferences)
                preferenceAnswer = ''
                for i, selectedPreference in enumerate(selectedPreferences):
                    if (i > 0):
                        preferenceAnswer += ' Also, '
                    preferenceAnswer += selectedPreference

                newDf.append({
                    'text': preferenceAnswer,
                    'sender': 'user',
                    'item': item,
                })
                
                # 3. present recommendation
                recommendationTexts = [
                    'I can recommend the following phone:',
                    'I think this phone would be a great choice:',
                    'I think you will like this phone:',
                    'This is the phone I would also recommend to my friends. It is a good choice.',
                ] 
                newDf.append({
                    'text': random.choice(recommendationTexts),
                    'sender': 'bot',
                    'item': item,
                    'isConst': False
                })
                newDf.append({
                    'text': meta['title'],
                    'sender': 'bot',
                    'item': item,
                    'isConst': True
                })

                # 4. ask a few questions  
                numOfQuestions = min(random.randint(1,2), len(qaAspects))
                selectedQaAspects = []
                while (len(selectedQaAspects) < numOfQuestions) and (len(qaAspects) > 0):
                    aspect = random.choice(list(qaAspects)).lower()
                    if (isAspectNew(aspect, usedAspects)):
                        selectedQaAspects.append(aspect)
                        usedAspects.add(aspect)
                    
                    # remove similar aspects
                    for otherAspect in qaAspects.copy():
                        if (not isAspectNew(otherAspect, usedAspects)):
                            qaAspects.remove(otherAspect)
                
                for aspect in selectedQaAspects:
                    qaEntry = random.choice(qaBlockAspectMap[aspect])
                    currentEntry = qaEntry['Qpos1A']
                    # turnId,text,sender,scenario,targetStyle,isConst
                    newDf.append({
                        'text': currentEntry['Question'],
                        'sender': 'user',
                        'aspect': aspect,
                        'item': item,
                        'reviewKey': currentEntry['Labels']['Key'],
                        'blockType': 'Qpos1A_Apos1A'
                    })
                    currentEntry = qaEntry['Apos1A']
                    newDf.append({
                        'text': currentEntry['Answer'],
                        'sender': 'bot',
                        'aspect': aspect,
                        'item': item,
                        'reviewKey': currentEntry['Labels']['Key'],
                        'blockType': 'Qpos1A_Apos1A',
                        'isConst': False
                    })
                                
                # 5. fill up with exchanging opinions
                # 'Oneg1A_Opos1A'
                # 'Oneg1A_Opos2A_restricted' 
                #   restricuted == We agree with the user only when there is a positive review for the aspect
                # 'Oneg1A_Opos2A_unrestricted'

                opinionSources = ['Oneg1A_Opos2A_restricted', 'Oneg1A_Opos2A_unrestricted']

                def getEntry(source):
                    opinionBlocks = blocks[source]
                    possibleKeys = []
                    for blockKey, blockEntry in opinionBlocks.items():
                        isUsableBlock = True
                        for partName, part in blockEntry.items():
                            reviewKey = part['Labels']['Key']
                            if (reviewKey in usedReviews):
                                isUsableBlock = False
                                break
                        if (isUsableBlock):
                            possibleKeys.append(blockKey)
                    
                    if (len(possibleKeys) == 0):
                        print('Error:', 'There are no possible reviews left.')
                        raise ValueError('There are no possible reviews left.')
                    targetKey = random.choice(possibleKeys)
                    targetBlock = opinionBlocks[targetKey]

                    entries = []
                    partOrder = ['Oneg1A', 'Opos1A']
                    senderOrder = ['user', 'bot']
                    textColumn = 'Opinion'
                    if (source == 'Oneg1A_Opos1A'):
                        partOrder = ['Oneg1A', 'Opos1A']
                    elif ('Oneg1A_Opos2A' in source):
                        partOrder = ['Oneg1A', 'Opos2A']
                    
                    for sender, part in zip(senderOrder, partOrder):
                        currentEntry = targetBlock[part]
                        currentLabels = currentEntry['Labels']
                        reviewKey = currentLabels['Key']
                        entries.append({
                            'text': currentEntry[textColumn],
                            'sender': sender,
                            'aspect': currentLabels['Aspect'].lower(),
                            'item': item,
                            'reviewKey': reviewKey,
                            'blockType': source
                        })
                        if (sender == 'bot'):
                            entries[-1]['isConst'] = False

                        usedReviews.add(reviewKey)
                    
                    return entries
            
                numOfOpinions = 2 - numOfQuestions
                addContradiction = random.choice([True, False])
                if (addContradiction and numOfOpinions > 0):
                    newDf += getEntry('Oneg1A_Opos1A')
                    numOfOpinions -= 1

                errorCount = 0
                while (numOfOpinions > 0 and errorCount < 2):
                    try:
                        newDf += getEntry(random.choice(opinionSources))
                        numOfOpinions -= 1
                    except:
                        errorCount += 1

                if (errorCount > 1):
                    raise ValueError('Not enough reviews.')

                # 6. add closing words
                thanksText = [
                    'Thank you for your help!',
                    'Thank you!'
                ]
                newDf.append({
                    'text': random.choice(thanksText),
                    'sender': 'user',
                    'item': item,
                })

                newDf = pd.DataFrame(newDf)
                newDf = newDf.assign(
                    turnId=list(range(len(newDf))),
                    scenario='webshop',
                    targetStyle='original',
                    brand=brand,
                    phoneOs=phoneOs,
                    price=price
                )
                newDf.loc[newDf['sender'] == 'user', ['isConst']] = True

                dfs.append(newDf)
                print('Done. Added conversation', j)
        except:
            pass
        return dfs

    def getConvForItem(item):
        print('item:', item)
        blocks = getBlocks(item)
        filterBlocks(blocks)
        try:
            return buildConversations(blocks, item)
        except:
            print('why am I here?')
            return []


    items = list(dict_AspectSentiment.keys())
    random.shuffle(items)
    with multiprocessing.Pool(processes=4) as pool:
        results = pool.map(getConvForItem, items)

    print('all processes finished!')
    convIdCtr = 0
    newDfs = []
    for result in results:
        for newDf in result:
            newDf = newDf.assign(conversationId=convIdCtr)
            newDfs.append(newDf)
            convIdCtr += 1
    
    resultsDf = pd.concat(newDfs, ignore_index=True)
    resultsDf.to_csv('product-conversations.csv')