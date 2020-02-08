#import gs_probdist as gspd
#import semrel as sr
import cardgen as cg
import gensim

MAINWORD = 'victory'

probdist_dict = cg.get_gold_probdist()

# Load pre-trained word embeddings.
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
print('Model loaded :)\n')

c = cg.card_generator(MAINWORD, probdist_dict, model)
cg.pretty_print(c)

# can run this from the command line :)
