import os
# To install- pip install -U nltk
import nltk
# For first time only
# nltk.download('all')
import nltk.corpus
from nltk.corpus import brown
from nltk.tokenize import word_tokenize

# This is the corpora (i.e. the whole data) provided by nltk. Some have textual data, some have different functions associated to it.
# print(os.listdir(nltk.data.find("corpora")))
# Output- ['unicode_samples', 'mte_teip5.zip', 'indian', 'stopwords', 'brown', 'swadesh', 'mac_morpho', 'conll2002.zip', 'indian.zip', 'abc', 'comparative_sentences', 'brown_tei.zip', 'cmudict.zip', 'conll2000.zip', 'universal_treebanks_v20.zip', 'words', 'pros_cons', 'udhr2', 'nonbreaking_prefixes.zip', 'lin_thesaurus', 'webtext', 'smultron.zip', 'names', 'sentiwordnet', 'dolch.zip', 'wordnet_ic.zip', 'brown.zip', 'alpino.zip', 'panlex_swadesh.zip', 'cmudict', 'sinica_treebank.zip', 'treebank.zip', 'ptb', 'inaugural', 'ppattach.zip', 'dependency_treebank.zip', 'opinion_lexicon.zip', 'cess_esp.zip', 'product_reviews_2', 'genesis.zip', 'reuters.zip', 'conll2007.zip', 'conll2002', 'comparative_sentences.zip', 'switchboard.zip', 'cess_cat.zip', 'udhr.zip', 'subjectivity.zip', 'pl196x.zip', 'ieer', 'problem_reports', 'timit.zip', 'floresta', 'paradigms.zip', 'gazetteers.zip', 'wordnet.zip', 'inaugural.zip', 'sinica_treebank', 'stopwords.zip', 'verbnet.zip', 'gutenberg', 'ieer.zip', 'ycoe.zip', 'shakespeare.zip', 'sentence_polarity', 'framenet_v17.zip', 'kimmo.zip', 'chat80.zip', 'kimmo', 'qc.zip', 'nonbreaking_prefixes', 'senseval', 'verbnet', 'udhr2.zip', 'senseval.zip', 'chat80', 'framenet_v15.zip', 'unicode_samples.zip', 'biocreative_ppi', 'framenet_v17', 'words.zip', 'pil', 'alpino', 'omw', 'cess_cat', 'shakespeare', 'city_database', 'product_reviews_2.zip', 'abc.zip', 'europarl_raw', 'sentiwordnet.zip', 'rte.zip', 'movie_reviews', 'toolbox.zip', 'product_reviews_1.zip', 'omw.zip', 'jeita.zip', 'wordnet_ic', 'names.zip', 'conll2000', 'dependency_treebank', 'floresta.zip', 'nombank.1.0.zip', 'wordnet', 'cess_esp', 'ptb.zip', 'mac_morpho.zip', 'knbc.zip', 'opinion_lexicon', 'toolbox', 'comtrans.zip', 'swadesh.zip', 'propbank.zip', 'mte_teip5', 'gutenberg.zip', 'product_reviews_1', 'twitter_samples.zip', 'treebank', 'state_union.zip', 'machado.zip', 'rte', 'nps_chat', 'crubadan', 'semcor.zip', 'biocreative_ppi.zip', 'ppattach', 'europarl_raw.zip', 'switchboard', 'brown_tei', 'verbnet3.zip', 'verbnet3', 'crubadan.zip', 'pil.zip', 'ycoe', 'webtext.zip', 'sentence_polarity.zip', 'timit', 'pl196x', 'nps_chat.zip', 'state_union', 'city_database.zip', 'subjectivity', 'framenet_v15', 'masc_tagged.zip', 'paradigms', 'genesis', 'gazetteers', 'twitter_samples', 'qc', 'lin_thesaurus.zip', 'udhr', 'movie_reviews.zip', 'dolch', 'problem_reports.zip', 'smultron', 'pros_cons.zip']

# print(brown.words())
# Output- ['The', 'Fulton', 'County', 'Grand', 'Jury', 'said', ...]

# A list of the gutenberg files-
# print(nltk.corpus.gutenberg.fileids())
# Output- ['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt', 'blake-poems.txt', 'bryant-stories.txt', 'burgess-busterbrown.txt', 'carroll-alice.txt', 'chesterton-ball.txt', 'chesterton-brown.txt', 'chesterton-thursday.txt', 'edgeworth-parents.txt', 'melville-moby_dick.txt', 'milton-paradise.txt', 'shakespeare-caesar.txt', 'shakespeare-hamlet.txt', 'shakespeare-macbeth.txt', 'whitman-leaves.txt']

# Inside the hamlet file, it starts with 'The Tragedie of Hamlet by'
# hamlet=nltk.corpus.gutenberg.words('shakespeare-hamlet.txt')
# print(hamlet)
# Output- ['[', 'The', 'Tragedie', 'of', 'Hamlet', 'by', ...]

# If we want to see the first 500 words of the textual file-
# for word in hamlet[:500]:
#     print(word, sep=' ', end=' ')

# For naturual language processing, we can use our own word. Below we're defining a string called AI.
AI = """Tonight Lisa Wilkinson meets the Qantas pilots turned bus drivers who are sticking together and helping each other through the incredibly tough times brought on by Covid."""

# print(type(AI))
# Output- str

# Below will divide the whole AI paragraph into tokens. It has taken a comma also into consideration.
AI_tokens = word_tokenize(AI)
print(AI_tokens)