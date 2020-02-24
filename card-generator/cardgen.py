import gs_probdist as gspd
import semrel as sr
import pandas as pd
import numpy as np
import random as rd


def select_five_categories(prob_dist_dict):
    """
    Given a probability distribution of semantic relation labels, randomly returns a list of five of them, weighted
    by probability.

    Arg:
        prob_dist_dict: a dictionary with semantic relation labels as keys and their probability as values
        (output of freq_dist_to_prob_dist() )
    Returns:
        A list containing five semantic relation labels (intended as the starting point for each card).
    """
    # For clarity, save keys as labels and values as probabilities.
    labels = list( prob_dist_dict.keys() )
    probs = list( prob_dist_dict.values() )

    # Use numpy's .choice() to return a label based on the given weight.
    return list( np.random.choice(labels, 5, p=probs) )


def get_good_label_distrib(semrel_dict, semrel_counts):
    """
    Finds a distribution of five labels that is compatible with the cardinality of the available semrel sets.
    (Where any label is over-represented, it is replaced with 'collocation').

    Args:
        semrel_dict: A dictionary containing the semantic relations for the given MW.
        semrel_counts: A dictionary containing the number of each semantic relation the randomly-generated distribution wants.
    Returns:
        A dictionary containing the labels to be used as keys, and values of how many of each there should be.
    """

    # Do cardinality check on srdict first, to see if there are enough elements to fulfill the distribution in five_semrels.
    srdict_counts = {key:len(value) for key, value in semrel_dict.items()}

    # This variable will hold the number of collocations to add to the distribution in place of unfulfillable other labels.
    num_coll_to_add = 0

    # Go through all non-'collocation' labels in the dictionary that contains the requested number of labels.
    for label, count in semrel_counts.items():

        if label != 'collocation':

            # Get the difference in cardinality between the available set and the requested set.
            diffc = srdict_counts[label] - semrel_counts[label]

            # If negative, i.e. if there are more requested than available, record the difference (this is how many instances
            # of 'collocation' to add) and change the number of requested words to the number available.
            if diffc < 0:
                num_coll_to_add += abs(diffc)
                semrel_counts[label] = srdict_counts[label]

    # Adjust the values in 'collocation' in the dictionary.
    if 'collocation' in set(semrel_counts.keys()):
        semrel_counts['collocation'] += num_coll_to_add
    else:
        semrel_counts['collocation'] = num_coll_to_add

    return semrel_counts


def card_generator(mw, prob_dist_dict, gensim_model):
    """
    Generates a Taboo card with one main word and five Taboo words.

    Args:
        mw: A string, the main word to generate the Taboo words for.
        prob_dist_dict: a dictionary with semantic relation labels as keys and their probability as values
           (output of freq_dist_to_prob_dist() )
        gensim_model: The pre-trained word embeddings.
    Returns:
        A dictionary with main word as key and a list of five taboo words as values.
    """

    # First things first: make sure that the word is actually in the word2vec vocab.
    # word_vectors = gensim_model.wv
    if mw not in gensim_model.wv.vocab:
        return False

    # Generate five categories with the weighted probabilities based on their frequency in the gold standard data.
    five_semrels_list = select_five_categories(prob_dist_dict)
    five_semrels = pd.Series(five_semrels_list)

    # Count the number of instances of each semrel category in that list.
    semrels_counts = dict( five_semrels.value_counts() )

    # Generate the semantic relations dictionary.
    srdict = sr.make_semrel_dict(mw)

    # Rejig five_semrels_list, if need be, to one whose labels are compatible with the cardinality of the sets available
    # in srdict.
    good_five_labels = get_good_label_distrib(srdict, semrels_counts)

    # Now we just populate a list with the required number of each kind of word!
    # First, initialise list to contain the five final Taboo words (yay!)
    tws = []

    # Go through good_five_labels and, for the labels that aren't 'collocation', access their list in the dictionary and
    # randomly select however many out of it.
    for label, count in good_five_labels.items():
        if label != 'collocation':
            tws.extend( rd.sample( tuple( srdict[label] ), count ) )

    # Now, take the number of collocations needed and return the most similar words according to gensim, removing the
    # words that are forbidden (i.e. the main word and also the other words that are already in tws)
    forbidden_words = set(tws + [mw])
    num_coll = good_five_labels['collocation']
    collocates = sr.get_collocations(mw, forbidden_words, gensim_model, num_collocates =  num_coll)

    # If there are more collocates than needed, randomly select num_coll of them and add to tws. Else just add list to tws.
    if len(collocates) > num_coll:
        tws.extend( rd.sample( tuple(collocates), num_coll ) )
    else:
        tws.extend(collocates)

    return {mw: tws}


def pretty_print(card):
    """
    Pretty-prints an ASCII Taboo card to the screen.

    Arg:
        card: A dictionary with the main word as the key and a list of five Taboo words as the value (or False, if main word wasn't in word2vec vocab)
    Returns:
        Nothing. Prints a card.
    """

    # If word not in word2vec vocab, then card's value is just False. Check if that's the case.
    if not card:
        print('Sorry, no card can be generated for this word! Please try another one.')
        return None

    # If the card does have some value, we continue on...

    # Assign some useful values to variables to use in printing below.
    mw = list(card.keys())[0]
    tws = list( *card.values() )
    words = tws + list(card.keys())

    # Get length of longest word to appear on the card and use this as orientation for printing.
    longest = max(len(w) for w in words)
    width = longest + 8  # between borders
    hline = ' -----' + '-'*longest + '-----'

    # Print header containing MW.
    print(hline)
    print(' |    ' + mw + ' '*(width - len(mw) - 4) + '|')
    print(hline)

    # Print body containing the five TWs.
    for tw in tws:
        print(' |    ' + tw + ' '*(width - len(tw) - 4) + '|')
    print(hline)


def get_gold_probdist():
    """
    Creates a probability distribution based on the frequency distribution of the semantic categories in the
    manually annotated csv 'gold-std-categorised.csv' saved in the current directory.

    Returns:
        A dictionary whose keys are the five semantic relations and whose values are their probabilities.
    """

    # Read in the dataset as a pandas dataframe.
    card_data_annot = gspd.read_in_categorised()

    # Based on the frequencies of each category in the data, create probability distribution and return.
    probdist_dict = gspd.freq_dist_to_prob_dist(card_data_annot)
    return probdist_dict


def draw_card(mw, gensim_model):
    """
    Generates a Taboo card with one main word and five Taboo words and pretty-prints it.

    Args:
        mw: A string, the main word to generate the Taboo words for.
        gensim_model: The pre-trained word embeddings.
    Returns:
        Nothing. Prints a card.
    """

    # Get the probability distribution of labels for each TW slot, based on the frequency of each semantic category
    # in the gold-standard Taboo cards.
    prob_dist_dict = get_gold_probdist()

    # Use this probability, the main word, and the model to generate a card, and then pretty-print it.
    c = card_generator(mw, prob_dist_dict, gensim_model)
    pretty_print(c)
