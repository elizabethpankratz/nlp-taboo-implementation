import gs_probdist as gspd
import semrel as sr
import cardgen as cg

MAINWORD = 'victory'

card_data_annot = gspd.read_in_categorised()
probdist_dict = gspd.freq_dist_to_prob_dist(card_data_annot)

# Load pre-trained word embeddings.
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
print('Model loaded :)\n')

c = cg.card_generator(MAINWORD, probdist_dict, model)
cg.pretty_print(c)

# can run this from the command line :)
