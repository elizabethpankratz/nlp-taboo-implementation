from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.metrics import edit_distance

def word_to_synsets(word):
    """
    Converts the given word to a synset object.

    Arg:
        word: a string like 'cat'
        pos: the desired part of speech (choices: wn.NOUN, wn.VERB, wn.ADJ, wn.ADV)
    Returns:
        A string containing the first synset ID, formatted according to WordNet's conventions, e.g. 'cat.n.01',
        corresponding to that word.
    """
    # Convert word string to the synset with the corresponding part of speech.
    return wn.synsets(word)


def synset_to_word(synset):
    """
    Converts the given synset to the actual word it represents.

    Arg:
        synset: a WordNet Synset object
    Returns:
        A string containing the word corresponding to that synset.
    """
    # Convert synset to lemma, since this is what name() is defined over.
    return synset.lemmas()[0].name()


def get_antonyms(synset):
    """
    Returns all antonyms for the given synset.

    Arg:
        synset: a WordNet Synset object
    Returns:
        A list of antonymic words as strings, if there are any, or else the empty list.
    """
    # Convert synset to lemma, since this is what the antonym relation is defined over, and get antonym(s).
    ant_lemmas = synset.lemmas()[0].antonyms()

    # Convert each antonym in this list to a string and return list (empty if no antonyms).
    return [ant_lemma.name() for ant_lemma in ant_lemmas]


def get_hypernyms(synset):
    """
    Returns all immediate hypernyms for the given synset.

    Arg:
        synset: a WordNet Synset object
    Returns:
        A list of hypernymic words as strings.
    """
    # Convert hypernyms of the synset to strings and return list.
    return [synset_to_word(hyper) for hyper in synset.hypernyms()]


def get_hyponyms(synset):
    """
    Returns all immediate hyponyms for the given synset. (There are often many.)

    Arg:
        synset: a WordNet Synset object
    Returns:
        A list of hyponymic words as strings.
    """
    # Convert hypernyms of the synset to strings and return list.
    return [synset_to_word(hypo) for hypo in synset.hyponyms()]


def get_synonyms(word):
    """
    Returns a set of synonyms, according to WordNet, for the given input word (using all of its senses, if
    there are multiple).

    Arg:
        word: a string representing the word whose synonyms we want.
    Returns:
        A set containing all of the other words in the same WordNet synset as the given word.
    """
    # Initialise set that will collect the synonyms.
    synonym_set = set()

    # Convert the word to a list of synsets.
    synset_list = word_to_synsets(word)

    # Get all the lemmas corresponding to the given word's synset.
    synonym_lems = [x.lemmas() for x in synset_list]

    # Go through them, get the names from the lemma (lowercasing everything for consistency), and add
    # to synonym_set.
    for lemma_list in synonym_lems:
        syn = lemma_list[0].name().lower()
        synonym_set.update( [syn] )

    # Remove from the synonym set the input word and any words that also contain the input word and return.
    to_rm = set()
    for synonym in synonym_set:
        if synonym == word or word in synonym:
            to_rm.update({synonym})

    return synonym_set.difference(to_rm)


def make_semrel_dict(word):
    """
    Creates a dictionary that contains all words standing in the given semantic relation to the main word.

    Arg:
        word: a string like 'cat' (the main word)
    Returns:
        A dictionary with the semantic relations as keys and a set of words that have that relation to all senses
        of the input word, according to WordNet, as values.
    """

    # Initialise dictionary (and we can get synonyms right away).
    semrel_dict = {
        'semrel_synonym': get_synonyms(word),
        'semrel_antonym': set(),
        'semrel_hypernym': set(),
        'semrel_hyponym': set()
    }

    # Convert the input word to all of its synsets.
    ss = word_to_synsets(word)

    # Go through each synset, determining its antonyms, hypernyms, and hyponyms, and adding each to the set in the
    # appropriate entry of the dictionary, as long as the main word does not appear as part of any of those strings.
    for s in ss:
        semrel_dict['semrel_antonym'].update( [w for w in get_antonyms(s) if word not in w] )
        semrel_dict['semrel_hypernym'].update( [w for w in get_hypernyms(s) if word not in w] )
        semrel_dict['semrel_hyponym'].update( [w for w in get_hyponyms(s) if word not in w] )

    return semrel_dict


def get_collocations(word, forbidden_wds, gensim_model, num_collocates, num_to_check = 10):
    """
    Returns minimum num_collocates most similar words to the given word based on gensim word embeddings.

    Arg:
        word: A string representing the main word.
        forbidden_wds: A set containing words as strings that may not be included as output.
        gensim_model: The pre-trained word embeddings.
        num_collocates: An integer, the number of collocates to generate.
        num_to_check: (default 10) the number of most similar words to begin with.
    Returns:
        A list of collocated words as strings.
    """

    lemmatizer = WordNetLemmatizer()

    # Use gensim's most_similar() function to get the (initially ten) words whose embeddings are most similar to the
    # input word's. Lemmatise the words to remove plural/other inflections.

    try:  # If the MW is in the word2vec vocabulary
        similar_tups = gensim_model.most_similar(word, topn=num_to_check)
        similar_wds = [lemmatizer.lemmatize( tup[0] ) for tup in similar_tups]

        # Now save only a subset of those similar words, namely the ones that:
        # - do not contain the main word
        # - are not in the passed-in set of forbidden words
        # - have a Levenshtein distance of up to 3 from the MW (i.e. are similar, probably typos)
        # - words with underscores in them, indicating multi-word units (often pretty weird)
        filtered = [wd for wd in similar_wds if (word not in wd.lower() and wd not in forbidden_wds and edit_distance(word, wd) > 4 and '_' not in wd)]
        filtered = set(filtered)

        # Recursive bit: Check if there are at least num_collocates different words in filtered (base case).
        # If not, increase the number of words to check in each recursive iteration by three and run the function again.
        # Will stop once there are minimum num_collocates words in filtered.

        if len(filtered) >= num_collocates:
            return filtered
        else:
            num_to_check += 3
            return get_collocations(word, forbidden_wds, gensim_model, num_collocates, num_to_check)

    except KeyError:  # arises when input word not in vocab
        pass
