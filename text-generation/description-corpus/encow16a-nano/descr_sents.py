# -*- coding: utf-8 -*-

# NOTE: Must be run on the RStudio server of webcorpora.org
# Spring 2020
# Creating a corpus of descriptive sentences to train a neural network on for ANLP final project.

from SeaCOW import Query, ConcordanceWriter, DependencyBuilder


def sent_query_csv(cql_string, csv_filename):

  # Create a Query object and set whatever needs to be set.
  q = Query()
  q.corpus          = 'encow16a-nano'         # Lower-case name of the corpus to use.
  q.string          = cql_string              # A normal CQL string as used in NoSketchEngine.
  q.max_hits        = -1                      # Maximal number of hits to return. Use when testing queries!
  q.attributes      = ['word']                # Attributes (of tokens) to export from corpus.
  q.structures      = []                      # Structure markup to export from corpus.
  q.references      = ['doc.url']
                                              # Which reference attributes (of structures) to export.
  q.container       = 's'                     # Which container strutcure should be exported?

  # This enables an efficient duplicate remover using a scaling Bloom filter.
  q.set_deduplication()

  # Create a Processor object and attach it to the Query object.
  # The ConcordanceWriter processor just writes a nice CSV file
  # containing your concordance, incl. all meta data you need
  # as comments at the top and bottom of the table.
  p                 = ConcordanceWriter() # Create a processor object of apporpriate type.
  p.filename        = csv_filename        # File name for output data. Directories MUST EXIST!
  q.processor       = p                   # Attach the processor to the query.
  q.run()                                 # Run the query.


# Save the filenames and the CQL queries for each descriptive construction.
query_dict = {
  'be_det.csv' : '[lemma="a(n)?"][tag="NN"][lemma="be"][tag="DT"]' ,
  'be_not_det.csv' : '[lemma="a(n)?"][tag="NN"][lemma="be"][lemma="not"][tag="DT"]' ,
  'look_like.csv' : '[lemma="a(n)?"][tag="NN"][lemma="look" & tag!="N*"][lemma="like"]' ,
  'sound_like.csv' : '[lemma="a(n)?"][tag="NN"][lemma="sound" & tag!="N*"][lemma="like"]' ,
  'feel_like.csv' : '[lemma="a(n)?"][tag="NN"][lemma="feel" & tag!="N*"][lemma="like"]' ,
  'smell_like.csv' : '[lemma="a(n)?"][tag="NN"][lemma="smell" & tag!="N*"][lemma="like"]' ,
  'means.csv' : '[lemma="a(n)?"][tag="NN"][lemma="mean" & tag!="N*"]' ,
  'be_part_of.csv' : '[lemma="a(n)?"][tag="NN"][lemma="be"][word="part"][word="of"]' ,
  'be_related_to.csv' : '[lemma="a(n)?"][tag="NN"][lemma="be"][word="related"][word="to"]' ,
  'be_another_word_for.csv' : '[lemma="a(n)?"][tag="NN"][lemma="be"][word="another"][word="word"][word="for"]' ,
  'can_be_found.csv' : '[lemma="a(n)?"][tag="NN"][lemma="can"][lemma="be"][word="found"]' ,
  'be_usually_used.csv' : '[lemma="a(n)?"][tag="NN"][lemma="be"][word="usually"][word="used"]' ,
  'be_something_that_you.csv' : '[lemma="a(n)?"][tag="NN"][lemma="be"][word="something"][word="that"][word="you"]'
}

# Loop through these, run each query, and save the result in a csv file.
for filename, query in query_dict.items():
  sent_query_csv(query, filename)
  print 'Created', filename
