import os
# To install- pip install -U nltk
import nltk
# For first time only
# nltk.download('all')
import nltk.corpus





# This is the corpora (i.e. the whole data) provided by nltk. Some have textual data, some have different functions associated to it.
# print(os.listdir(nltk.data.find("corpora")))
# Output- ['unicode_samples', 'mte_teip5.zip', 'indian', 'stopwords', 'brown', 'swadesh', 'mac_morpho', 'conll2002.zip', 'indian.zip', 'abc', 'comparative_sentences', 'brown_tei.zip', 'cmudict.zip', 'conll2000.zip', 'universal_treebanks_v20.zip', 'words', 'pros_cons', 'udhr2', 'nonbreaking_prefixes.zip', 'lin_thesaurus', 'webtext', 'smultron.zip', 'names', 'sentiwordnet', 'dolch.zip', 'wordnet_ic.zip', 'brown.zip', 'alpino.zip', 'panlex_swadesh.zip', 'cmudict', 'sinica_treebank.zip', 'treebank.zip', 'ptb', 'inaugural', 'ppattach.zip', 'dependency_treebank.zip', 'opinion_lexicon.zip', 'cess_esp.zip', 'product_reviews_2', 'genesis.zip', 'reuters.zip', 'conll2007.zip', 'conll2002', 'comparative_sentences.zip', 'switchboard.zip', 'cess_cat.zip', 'udhr.zip', 'subjectivity.zip', 'pl196x.zip', 'ieer', 'problem_reports', 'timit.zip', 'floresta', 'paradigms.zip', 'gazetteers.zip', 'wordnet.zip', 'inaugural.zip', 'sinica_treebank', 'stopwords.zip', 'verbnet.zip', 'gutenberg', 'ieer.zip', 'ycoe.zip', 'shakespeare.zip', 'sentence_polarity', 'framenet_v17.zip', 'kimmo.zip', 'chat80.zip', 'kimmo', 'qc.zip', 'nonbreaking_prefixes', 'senseval', 'verbnet', 'udhr2.zip', 'senseval.zip', 'chat80', 'framenet_v15.zip', 'unicode_samples.zip', 'biocreative_ppi', 'framenet_v17', 'words.zip', 'pil', 'alpino', 'omw', 'cess_cat', 'shakespeare', 'city_database', 'product_reviews_2.zip', 'abc.zip', 'europarl_raw', 'sentiwordnet.zip', 'rte.zip', 'movie_reviews', 'toolbox.zip', 'product_reviews_1.zip', 'omw.zip', 'jeita.zip', 'wordnet_ic', 'names.zip', 'conll2000', 'dependency_treebank', 'floresta.zip', 'nombank.1.0.zip', 'wordnet', 'cess_esp', 'ptb.zip', 'mac_morpho.zip', 'knbc.zip', 'opinion_lexicon', 'toolbox', 'comtrans.zip', 'swadesh.zip', 'propbank.zip', 'mte_teip5', 'gutenberg.zip', 'product_reviews_1', 'twitter_samples.zip', 'treebank', 'state_union.zip', 'machado.zip', 'rte', 'nps_chat', 'crubadan', 'semcor.zip', 'biocreative_ppi.zip', 'ppattach', 'europarl_raw.zip', 'switchboard', 'brown_tei', 'verbnet3.zip', 'verbnet3', 'crubadan.zip', 'pil.zip', 'ycoe', 'webtext.zip', 'sentence_polarity.zip', 'timit', 'pl196x', 'nps_chat.zip', 'state_union', 'city_database.zip', 'subjectivity', 'framenet_v15', 'masc_tagged.zip', 'paradigms', 'genesis', 'gazetteers', 'twitter_samples', 'qc', 'lin_thesaurus.zip', 'udhr', 'movie_reviews.zip', 'dolch', 'problem_reports.zip', 'smultron', 'pros_cons.zip']

from nltk.corpus import brown
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

from nltk.tokenize import word_tokenize
# For naturual language processing, we can use our own word. Below we're defining a string called AI.
AI = """Tonight Lisa Wilkinson meets the Qantas pilots turned bus drivers who are sticking together and helping each other through the incredibly tough times brought on by Covid."""

# print(type(AI))
# Output- str

# Below will divide the whole AI paragraph into tokens. It has taken a comma also into consideration.
AI_tokens = word_tokenize(AI)
# print(AI_tokens)

# The number of tokens we have. We'll use the length function.
# print(len(AI_tokens))
# Output- 28

from nltk.probability import FreqDist
# We'll use the frequency testing. We're using the frequency distinct function which is already present in the nltk.-
# fdist = FreqDist()
# for word in AI_tokens:
    # we're going to conver the word to low key to reduce to probability of considering the uppercase and lowercase words as different. And we'll assign it a number. This is basically a word count program.
    # fdist[word.lower()]+=1
# print(fdist.most_common())
# Output- 
# [('the', 2), ('tonight', 1), ('lisa', 1), ('wilkinson', 1), ('meets', 1), ('qantas', 1), ('pilots', 1), ('turned', 1), ('bus', 1), ('drivers', 1), ('who', 1), ('are', 1), ('sticking', 1), ('together', 1), ('and', 1), ('helping', 1), ('each', 1), ('other', 1), ('through', 1), ('incredibly', 1), ('tough', 1), ('times', 1), ('brought', 1), ('on', 1), ('by', 1), ('covid', 1), ('.', 1)]

# To know the frequency of any particular word here, we're going to use the function fdist-
# print(fdist['tonight'])

# To look at the number of distinct words here-
# print(len(fdist))
# Output- 27
# We have 28 tokens. Out of that we have 27 distinct tokens.

# To select the top 10 tokens with the highest frequency-
# fdist_top10 = fdist.most_common(10)
# print(fdist_top10)
# Output- [('the', 2), ('tonight', 1), ('lisa', 1), ('wilkinson', 1), ('meets', 1), ('qantas', 1), ('pilots', 1), ('turned', 1), ('bus', 1), ('drivers', 1)]

# To use blankline tokenizer over the same string to tokenize a paragraph in respect to a blank string.
from nltk.tokenize import blankline_tokenize

# AI_slug="""Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome.

# The sky is pinkish-blue. You shouldn't eat cardboard.

# Hello Mr. Adam, how are you? I hope everything is going well. Today is a good day, see you dude."""

# AI_blank=blankline_tokenize(AI_slug)
# print(len(AI_blank))
# Output- 3
# It tells us the number of paragraphs separated by a new line.

# To look at the second paragraph, which is paragraph number 1-
# print(AI_blank[1])
# Output- The sky is pinkish-blue. You shouldn't eat cardboard.

# Tokenization - Trigram, Ngram and Bigram
from nltk.util import bigrams, trigrams, ngrams
string = "The best and most beautiful things in the world cannot be seen or even touched, they must be felt with the heart"
# First we create the token of words
quotes_tokens = nltk.word_tokenize(string)
# print(quotes_tokens)
# Output- ['The', 'best', 'and', 'most', 'beautiful', 'things', 'in', 'the', 'world', 'can', 'not', 'be', 'seen', 'or', 'even', 'touched', ',', 'they', 'must', 'be', 'felt', 'with', 'the', 'heart']

# quotes_bigrams = list(nltk.bigrams(quotes_tokens))
# print(quotes_bigrams)
# Output- [('The', 'best'), ('best', 'and'), ('and', 'most'), ('most', 'beautiful'), ('beautiful', 'things'), ('things', 'in'), ('in', 'the'), ('the', 'world'), ('world', 'can'), ('can', 'not'), ('not', 'be'), ('be', 'seen'), ('seen', 'or'), ('or', 'even'), ('even', 'touched'), ('touched', ','), (',', 'they'), ('they', 'must'), ('must', 'be'), ('be', 'felt'), ('felt', 'with'), ('with', 'the'), ('the', 'heart')]

# quotes_trigrams = list(nltk.trigrams(quotes_tokens))
# print(quotes_trigrams)
# Output- [('The', 'best', 'and'), ('best', 'and', 'most'), ('and', 'most', 'beautiful'), ('most', 'beautiful', 'things'), ('beautiful', 'things', 'in'), ('things', 'in', 'the'), ('in', 'the', 'world'), ('the', 'world', 'can'), ('world', 'can', 'not'), ('can', 'not', 'be'), ('not', 'be', 'seen'), ('be', 'seen', 'or'), ('seen', 'or', 'even'), ('or', 'even', 'touched'), ('even', 'touched', ','), ('touched', ',', 'they'), (',', 'they', 'must'), ('they', 'must', 'be'), ('must', 'be', 'felt'), ('be', 'felt', 'with'), ('felt', 'with', 'the'), ('with', 'the', 'heart')]

# The output has given us an ngram of length 5.
# quotes_ngrams = list(nltk.ngrams(quotes_tokens, 5))
# print(quotes_ngrams)
# Output- [('The', 'best', 'and', 'most', 'beautiful'), ('best', 'and', 'most', 'beautiful', 'things'), ('and', 'most', 'beautiful', 'things', 'in'), ('most', 'beautiful', 'things', 'in', 'the'), ('beautiful', 'things', 'in', 'the', 'world'), ('things', 'in', 'the', 'world', 'can'), ('in', 'the', 'world', 'can', 'not'), ('the', 'world', 'can', 'not', 'be'), ('world', 'can', 'not', 'be', 'seen'), ('can', 'not', 'be', 'seen', 'or'), ('not', 'be', 'seen', 'or', 'even'), ('be', 'seen', 'or', 'even', 'touched'), ('seen', 'or', 'even', 'touched', ','), ('or', 'even', 'touched', ',', 'they'), ('even', 'touched', ',', 'they', 'must'), ('touched', ',', 'they', 'must', 'be'), (',', 'they', 'must', 'be', 'felt'), ('they', 'must', 'be', 'felt', 'with'), ('must', 'be', 'felt', 'with', 'the'), ('be', 'felt', 'with', 'the', 'heart')]

# Stemming - Normalize words into it's base form or root form
# Affectation, affects, affections, affected - originated from affect
# Once we got all the tokens, we need to make some change to the tokens. For that we have stemming.
# Stemming works by cutting the end or beginning of the word which are some common prefix or suffixes. It can be successful in some occassions.

from nltk.stem import PorterStemmer
pst = PorterStemmer()
print(pst.stem("having"))