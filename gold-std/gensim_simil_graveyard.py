# contains the excised bits of script from gs.ipynb that we won't actually be using, but that I don't want to throw away

import gensim

# The following line of code requires that the large word embeddings file is in the current directory (not set up on
# GitHub because it is too large to push around nicely, so for replication, have to add this file manually).
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

def add_similarity_to_df(df):
    """
    Computes the similarity between the MW in the first column and the TW in the second of the passed-in dataframe,
    when the TW is flagged in the column collocation.

    Args:
        df: dataframe containing cleaned corpus data

    Returns:
        A dataframe with a new column 'simil' containing word2vec similarity values for all words classified as collocations.
    """

    # Initialise empty list to collect similarity values as we go.
    simil_col=[]

    # Iterate through rows in the dataframe.
    for row_idx in range(len(df)):

        # Check if the value in the collocation column is 1.
        if df.loc[row_idx, ['collocation']][0] == 1:

            # Get the MW and the TW in the current row.
            mw = df.loc[row_idx, ['mw']][0]
            tw = df.loc[row_idx, ['tw']][0]

            # KeyError raised if word not in the word2vec vocabulary, so if that happens, add numpy's null value
            # to the column instead.
            try:
                value = model.similarity(mw, tw)
            except KeyError:
                value = np.nan

        else:
            value = np.nan

        # Add similarity value to the list. At the end, this list will be of the same length
        # as the dataframe and will contain a similarity value (or NaN) for each adjective pair.
        simil_col.append(value)

    # Add this list as a new column to the dataframe and return dataframe.
    df['simil'] = simil_col

    return df


gs_df = add_similarity_to_df(gs_df)
gs_df

# We can look at how the data are distributed.
# First make a dataframe from only the similarity column, for easier plotting.

simil_df = pd.DataFrame(gs_df['simil'])
simil_df.describe()
simil_df.hist();
simil_df.boxplot();
