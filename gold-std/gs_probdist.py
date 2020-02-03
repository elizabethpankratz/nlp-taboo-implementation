import pandas as pd
import semrel as sr

def cats():
    print('cattttts yes yeah')


def read_in(filename):
    """
    Reads in transcribed Taboo card words contained in the given file and returns them in an enumerated list.
    """
    file_lines = []

    with open(filename, "r", encoding='utf-8') as myfile:

        # Go through every line, saving non-empty ones to the list file_lines.
        for line in myfile:
            if line.strip() != '':
                file_lines.append(line.strip())

    return list(enumerate(file_lines))


def format_cards(enum_list):
    """
    Formats the contents of an enumerated list containing lines of the read-in file as a list of dictionaries.

    Arg:
        enum_list: An enumerated list (output of read_in())
    Returns:
        A list of dictionaries (key = MW, values = list of TWs)
    """

    # Initialise dictionary to contain card data.
    card_dict = dict()

    # Assign MWs (every sixth word in the enumerated list) as dictionary keys, and create a list for the dict's
    # value consisting of the five following words (the TWs); the word[1:] removes the dash from the beginning of
    # each TW's string.

    for enum, wd in enum_list:
        if enum % 6 == 0:
            card_dict[wd] = [word[1:] for num, word in enum_list[enum+1:enum+6]]

    return card_dict


def get_card_dicts():
    """
    Reads in transcribed Taboo cards (saved in taboo_cards.txt in current dir) and formats them as a dictionary.

    Returns:
        A dictionary with the main words of the cards as keys and the five taboo words in a list as the values.
    """
    CARDS_FILE = 'taboo_cards.txt'

    enum_lines = read_in(CARDS_FILE)
    cards = format_cards(enum_lines)
    return cards


def cards_to_df(card_dict):
    """
    Converts a dictionary of Taboo cards to a pandas dataframe.

    Arg:
        card_dict: A dictionary containing transcribed Taboo cards (output of get_card_dicts())
    Returns:
        A pandas dataframe with a row per MW/TW combination.
    """
    # Create a pandas dataframe quickly based on a list of dictionaries, with each dictionary corresponding to a row in the
    # dataframe.
    # ( Source: https://stackoverflow.com/questions/10715965/add-one-row-to-pandas-dataframe/17496530#17496530 )

    # Initialise empty list to contain each row.
    rows_list = []

    # Iterate through items in the dictionary.
    for mainwd, tabwds in card_dict.items():
        for tabwd in tabwds:

            # Create a dictionary for each row of the dataframe (key = column name, value = row value for that column)
            row = {
                'mw': mainwd,
                'tw': tabwd
            }

            # Append to rows_list, and use that list as a basis for the new dataframe.
            rows_list.append(row)

    # Convert this list of dictionaries to a dataframe and return.
    data = pd.DataFrame(rows_list)
    return data


def read_in_categorised():
    """
    Reads in manually annotated MW/TW combinations (thanks Anna) and saves as pandas df.
    Assumes annotated file saved as 'gold-std-categorised.csv' in current dir.

    Returns:
        A pandas dataframe with a row per MW/TW combination and a 1 in the category that combination belongs to.
    """
    # Set file location.
    GS_CSV_FILE = 'gold-std-categorised.csv'

    # Read the csv back in and save as pandas dataframe, replacing NaNs with 0, and return.
    gs_df = pd.read_csv(GS_CSV_FILE, encoding='utf-8')
    gs_df.fillna(0, inplace=True)
    return gs_df


def plot_category_freqs(gs_dataframe):
    """
    Count and plot the frequency of each category appearing in the dataset.

    Arg:
        gs_dataframe: A pandas dataframe, output of read_in_categorised()
    Returns:
        Nothing; saves plot as PDF in current dir.
    """
    # Get the sum of each column as a series.
    gs_df_sum = gs_dataframe.loc[:, 'semrel_synonym':'other'].sum()
    gs_df_sum.sort_values(ascending=False, inplace=True)

    # Create a bar plot object out of it.
    # (Plot y axis labels as integers by creating a list of integers based on the y range)
    yint = [x for x in range( int(gs_df_sum.max()) + 1 )]
    gs_df_sum_plot = gs_df_sum.plot.bar(x='category', y='count')

    # For readable x axis tick labels, set their rotation angle to 30 and the horizontal alignment to right.
    gs_df_sum_plot.set_xticklabels(gs_df_sum_plot.get_xticklabels(), rotation=30, horizontalalignment='right')

    # To save the figure, convert to a "figure" object and then export as pdf
    fig = gs_df_sum_plot.get_figure()
    fig.savefig('freq_plot_xxx.pdf', bbox_inches='tight')


def freq_dist_to_prob_dist(gs_dataframe):
    """
    Count the frequency of categories (that we care about) appearing in the dataset and convert to a probability distribution.

    Arg:
        gs_dataframe: A pandas dataframe, output of read_in_categorised()
    Returns:
        A dictionary containing the probability distribution (labels as keys, probabilities as values)
    """
    # Get the sum of each column as a series.
    gs_df_sum = gs_dataframe.loc[:, 'semrel_synonym':'other'].sum()
    gs_df_sum.sort_values(ascending=False, inplace=True)

    # Before converting to a probability distribution, remove the categories we aren't interested in ("cultural_ref" and "other").
    gs_df_sum.drop(labels=['cultural_ref', 'other'], inplace=True)

    # Convert the frequency distribution to a probability distribution by dividing by the sum of all observations
    prob_dist = gs_df_sum / gs_df_sum.sum()

    # Convert this to a dictionary and return.
    return dict(prob_dist)
