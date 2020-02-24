# Taboo Implementation

A Taboo-style card generator, using pre-trained word2vec embeddings and semantic relations from WordNet, and a Taboo player text generator, implemented with an RNN using GRUs.

(card generator gif)

(text generator gif)

Final project for the course **BM1 Advanced Natural Language Processing** at the Universität Potsdam in the winter semester 2019/2020.
Developed by **Anna-Janina Goecke**, **Rodrigo Lopez Portillo Alcocer**, and **Elizabeth Pankratz**.


## What it does

- Given a "main word" (the word that your team members should guess), the card generator generates five "taboo words" (the words you cannot us in your description of the main word).
- Given a main word and five taboo words, the text generator produces syntactically coherent text that ideally would describe the main word, without using any of the taboo words.


## How to use

To run this project, you need Python 3 and the libraries `gensim`, `pandas`, `numpy`, `random`, `nltk`, **(RLPA: Add your libraries here too!)**.
Also, you should download the pre-trained word2vec embeddings, `GoogleNews-vectors-negative300.bin`, from the [link](https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM) provided in class.
(These are not included in the current repository because of their size.)

Begin by cloning this repository.

```
git clone https://github.com/epankratz/nlp-taboo-implementation
```

### Card generator

Move `GoogleNews-vectors-negative300.bin` into the directory `gold-standard/`.

Now, from your command line:

```
# Import the necessary libraries/modules
import cardgen as cg
import gensim

# Load the pre-trained embeddings
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# ...
```

For a commentated illustration of the card generator, see `dir/xyz-llustration.ipynb`.
For more detail about the card generator's development, see `dir/xyz-walkthrough.ipynb`.


### Text generator

- text gen detail
- detail 1
- detail 2


## Sources

Our text generator implementation was inspired by the following tutorials:
- i from j
- n from m
