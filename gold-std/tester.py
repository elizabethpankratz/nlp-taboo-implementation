import gs_probdist as gspd
import semrel as sr
import gensim

# df = gspd.read_in_categorised()
# freq = gspd.freq_dist_to_prob_dist(df)
# print( gspd.select_five_categories(freq) )

print( sr.make_semrel_dict('good') )

# Load pre-trained word embeddings.
# model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
# print( sr.get_collocations(model, 'dog', num_collocates=5) )
