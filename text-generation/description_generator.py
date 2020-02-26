import numpy as np
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from random import sample
import string
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import os
import time
import gs_probdist as gspd
import semrel as sr
import gensim
import cardgen as cg

def train(context, target):
    hidden = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0

    for t in range(len(trigrams)):
        output, hidden = decoder(context[t], hidden)
        loss += criterion(output, target[t])

    loss.backward()
    decoder_optimizer.step()

    return loss.data.item() / len(trigrams)

def model_selection(x):
    if x ==1:
        path = os.getcwd()+'/test5_trained_inference.pt'
        hidden_s = 150
        n_layers = 1
        lr = 0.015
    if x==2:
        path = os.getcwd()+'/test4_trained_inference.pt'
        hidden_s = 75
        n_layers = 2
        lr = 0.015
    decoder = GRU(voc_length, hidden_s, voc_length, n_layers)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    decoder = torch.load(path)
    decoder.eval()
    return decoder

def evaluate(prime_str='this process', predict_len=100, temperature=0.8):
    hidden = decoder.init_hidden()

    for p in range(predict_len):

        prime_input = torch.tensor([word_to_freq[w] for w in prime_str.split()], dtype=torch.long)
        cont = prime_input[-2:] #last two words as input
        output, hidden = decoder(cont, hidden)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted word to string and use as next input
        predicted_word = list(word_to_freq.keys())[list(word_to_freq.values()).index(top_i)]
        prime_str += " " + predicted_word
#         inp = torch.tensor(word_to_ix[predicted_word], dtype=torch.long)

    return prime_str

def gen_input_words(mw, model):
    #mw = main word
    #model = embeddings used to generate the cards

    #generating the corresponding taboo card
    card_words = cg.card_generator(mw, cg.get_gold_probdist(), model)
    #set of words that we hope will appear in the description
    input_words = card_words[mw] + [mw]

    # extending the input_words set using semantic relations. Bigger set --> better chances of generating an approved word!
    # we will use the make_semrel_dict function to get synonyms, hyponyms and hypernyms of the MW.
    # we considered adding also semrel words from the tw, but the loose connection to the MW very fast
    # we will leave out antonyms as they might make they are "riskier" to use in a description.

    adds = []
    temp = sr.make_semrel_dict(mw)
    for k in temp.keys():
        if k != 'semrel_antonym':
            new = list(temp[k])
            adds += new
    adds = np.unique(adds)
    adds = [x.lower() for x in adds]
    input_words = np.unique(input_words + adds)

    # filtering out the input words that are not in our vocab. Shouldn't be a thing when using larger corpus
    input_words = [word for word in input_words if word in voc]
    return input_words

def description_generator(mw, model, n_seeds = 3, n_iterations = 10, debugging = False, printing = False):
    #mw = main word
    #model = embeddings used to generate the cards
    #n_seeds = if we are using 2 or 3 seeds during the sentence generation step
    #n_iterations = how many iterations we will do in the generation step
    #debugging = True if we want to print some statistics about the process. False if we only want the last 5 generated sentences.
    #printing = True will print something, based on debugging. If false, it will only return the final sentence

    #generating the input_words we are aiming to include in our description
    input_words = gen_input_words(mw, model)
    #on average a descriptive sentence had 27 words/symbols.
    # we will equally divide them between our seeds


    # iterate until nice sentence comes up
    # we will add safety measure to not break everything
    i = 0
    index_in_sentence = -1


    #if we are using 3 seeds
    #the 3 most frequent ones in our corpus were "x is", 'x means' and "x can be found"
    if n_seeds == 3:
        #create the first sentence
        sentence_parts = np.array([evaluate(mw+' means', 7, temperature = 1), evaluate(mw+' is', 7, temperature = 1), evaluate(mw+' can be found', 5, temperature = 1)])
        sentence =  " ".join(sentence_parts)
        eval_sentence = sentence.split()

        # to keep track of scores
        scores = np.zeros(n_iterations)
        #first score vector and score
        #and accounting for the 3 times the TW appears already in the seeds
        score_vector = np.array([eval_sentence.count(word) for word in input_words])
        score_vector[input_words.index(mw)] -= 3
        score = np.sum(score_vector)

        # the covered vector will take care that we don't replace a segment that we already "like"
        covered = np.array([0,0,0])
        changes = np.zeros(len(score_vector))

        #known positions of input words in our sentence
        positions = np.zeros(len(eval_sentence))

        #we know the positions of the seeds
        positions[0] = 1
        positions[9] = 1
        positions[18] = 1

        while i < n_iterations:
            #aware that with this flow we are doing one iteration after reaching the desired score, but it's no big deal because score is designed to only go up.

            #checking if score improved
            new_score_vector = np.array([eval_sentence.count(word) for word in input_words])
            new_score_vector[input_words.index(mw)] -= 3
            changes = new_score_vector - score_vector

            if True in (changes>0): #there was a change. Assuming there is max 1 change per iteration from now on
                index = np.where(changes == 1)[0][0] #looking for the position in which an input_word was added
                word_that_was_added = input_words[index] #if we stop assuming that, here we have to keep track of location and magnitude of changes

                #finding in which segment that new added word is in order to leave the segment untouched

                #this detects the index of the word that just came up in case that word was already in our sentence
                indices_in_sentence = np.where(np.array(eval_sentence) == word_that_was_added)[0]
                if len(indices_in_sentence) >1: #word appears at least twice
                    for d in indices_in_sentence:
                        if positions[d] != 1:
                            index_in_sentence = d
                            positions[d] = 1
                else:
                    index_in_sentence = indices_in_sentence[0]
                    positions[index_in_sentence] = 1
                #keeping the segment in which the improvement took place
                if index_in_sentence in range(9) & covered[0]!=1:
                    sentence_parts[1] = evaluate(mw+' is', 7, temperature = 1)
                    sentence_parts[2] = evaluate(mw+' can be found', 5, temperature = 1)
                    sentence = ' '.join(sentence_parts)
                    covered[0] = 1
                elif index_in_sentence in range(9, 18) & covered[1] !=1:
                    sentence_parts[0] = evaluate(mw+' means', 7, temperature = 1)
                    sentence_parts[2] = evaluate(mw+' can be found', 5, temperature = 1)
                    sentence = ' '.join(sentence_parts)
                    covered[1] = 1
                elif index_in_sentence in range(18, 27) & covered[2] != 1:
                    sentence_parts[1] = evaluate(mw+' is', 7, temperature = 1)
                    sentence_parts[0] = evaluate(mw+' means', 7, temperature = 1)
                    sentence = ' '.join(sentence_parts)
                    covered[2] = 1
                eval_sentence = sentence.split()
                changes = np.zeros(len(score_vector))
                index_in_sentence = 0
                score_vector = new_score_vector
                score = np.sum(score_vector)

            #if there was no change
            else: #based on what is already covered
                if covered[0] ==0:
                    sentence_parts[0] = evaluate(mw+' means', 7, temperature = 1) +' '
                #if the first part is already covered we can add it as input to generate the second
                if covered[1] ==0:
                    if covered[0]==1:
                        temp =  evaluate(sentence_parts[0]+' '+ mw+' is', 7, temperature = 1) +' '
                        #taking off the first part from it
                        temp = temp.split()
                        sentence_parts[1] = " ".join(temp[9:])
                    else:
                        sentence_parts[1] = evaluate(mw+' is', 7, temperature = 1) +' '
                # same logic for the third part.
                if covered[2] == 0:
                    if covered[1] == 0:
                        sentence_parts[2] = evaluate(mw+' can be found', 5, temperature = 1)
                    else:
                        temp =  evaluate(sentence_parts[1]+' '+ mw+' can be found', 5, temperature = 1) +' '
                        #taking off the second part from it
                        temp = temp.split()
                        sentence_parts[2] = " ".join(temp[9:])
                sentence = ' '.join(sentence_parts)
                eval_sentence = sentence.split()
                score_vector = new_score_vector
                score = np.sum(score_vector)
            if printing == True:
                if debugging ==True:
                    print("Sentence number: " + str(i+1))
                    print(sentence)
                    if True in (changes>0):
                        print(changes)
                    print(covered)
                    print(positions)
                else:
                    if i in range(n_iterations-5, n_iterations):
                        print(sentence)
            scores[i] = score
            i +=1

    #if we are using 2 seeds
    #the 2 most frequent ones in our corpus were "x is" and 'x means'
    if n_seeds == 2:
        #create the first sentence
        sentence_parts = np.array([evaluate(mw+' means', 11, temperature = 1), evaluate(mw+' is', 12, temperature = 1)])
        sentence =  " ".join(sentence_parts)
        eval_sentence = sentence.split()

        # to keep track of scores
        scores = np.zeros(n_iterations)
        #first score vector and score
        #and accounting for the 3 times the TW appears already in the seeds
        score_vector = np.array([eval_sentence.count(word) for word in input_words])
        score_vector[input_words.index(mw)] -= 3
        score = np.sum(score_vector)

        # the covered vector will take care that we don't replace a segment that we already "like"
        covered = np.array([0,0])
        changes = np.zeros(len(score_vector))

        #known positions of input words in our sentence
        positions = np.zeros(len(eval_sentence))

        #we know the positions of the seeds
        positions[0] = 1
        positions[14] = 1

        while i < n_iterations:
            #aware that with this flow we are doing one iteration after reaching the desired score, but it's no big deal because score is designed to only go up.

            #checking if score improved
            new_score_vector = np.array([eval_sentence.count(word) for word in input_words])
            new_score_vector[input_words.index(mw)] -= 3
            changes = new_score_vector - score_vector

            if True in (changes>0): #there was a change. Assuming there is max 1 change per iteration from now on
                index = np.where(changes == 1)[0][0] #looking for the position in which an input_word was added
                word_that_was_added = input_words[index] #if we stop assuming that, here we have to keep track of location and magnitude of changes

                #finding in which segment that new added word is in order to leave the segment untouched

                #this detects the index of the word that just came up in case that word was already in our sentence
                indices_in_sentence = np.where(np.array(eval_sentence) == word_that_was_added)[0]
                if len(indices_in_sentence) >1: #word appears at least twice
                    for d in indices_in_sentence:
                        if positions[d] != 1:
                            index_in_sentence = d
                            positions[d] = 1
                else:
                    index_in_sentence = indices_in_sentence[0]
                    positions[index_in_sentence] = 1
                #keeping the segment in which the improvement took place
                if index_in_sentence in range(14):
                    sentence_parts[1] = evaluate(mw+' is', 12, temperature = 1)
                    sentence = ' '.join(sentence_parts)
                    covered[0] = 1
                elif index_in_sentence in range(14, 27):
                    sentence_parts[0] = evaluate(mw+' means', 11, temperature = 1)
                    sentence = ' '.join(sentence_parts)
                    covered[1] = 1
                eval_sentence = sentence.split()
                changes = np.zeros(len(score_vector))
                index_in_sentence = 0
                score_vector = new_score_vector
                score = np.sum(score_vector)

            #if there was no change
            else: #based on what is already covered
                if covered[0] ==0:
                    sentence_parts[0] = evaluate(mw+' means', 11, temperature = 1) +' '
                #if the first part is already covered we can add it as input to generate the second
                if covered[1] ==0:
                    if covered[0]==1:
                        temp =  evaluate(sentence_parts[0]+' '+ mw+' is', 12, temperature = 1) +' '
                        #taking off the first part from it
                        temp = temp.split()
                        sentence_parts[1] = " ".join(temp[12:])
                    else:
                        sentence_parts[1] = evaluate(mw+' is', 7, temperature = 1) +' '
                sentence = ' '.join(sentence_parts)
                eval_sentence = sentence.split()
                score_vector = new_score_vector
                score = np.sum(score_vector)

            if printing == True:
                if debugging ==True:
                    print("Sentence number: " + str(i+1))
                    print(sentence)
                    if True in (changes>0):
                        print(changes)
                    print(covered)
                    print(positions)
                else:
                    if i in range(n_iterations-5, n_iterations):
                        print(sentence)
            scores[i] = score
            i +=1
    return sentence

def sentence_cleaner(sentence, mw, model):
    #replacing MW with "the main word" and TWs appearing in the sentence with one of their synonyms
    sentence = sentence.replace(mw, 'The main word')

    #replacing any TWs appearing in our sentence with some allowed synonym
    taboo_words = cg.card_generator(mw, cg.get_gold_probdist(), model)[mw]

    spl = np.array(sentence.split())
    for tw in taboo_words:
        if tw in spl:
           #getting synonyms of detected tw
            syns = sr.get_synonyms(tw)
            if len(syns) > 0:
                syns = list(syns)
                choice = np.random.choice(syns)
                sentence = sentence.replace(tw, choice)
    return sentence

def final_output(mw, model, n_seeds = 3, n_iterations = 10, debugging = False, printing = False):
    sentence = description_generator(mw, model, n_seeds, n_iterations, debugging, printing)
    output = sentence_cleaner(sentence, mw, model)
    return output
