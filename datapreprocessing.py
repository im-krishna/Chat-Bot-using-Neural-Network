import nltk
nltk.download('punkt')

# building pipeline 
# .tokenisation
# .lowering
# .stemming
# .removing punctutation
# .embedding(bag of words)
# .neural networks(ffnn)
# .prediction

# .tokenisation and removing punctutation

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
#removes punctutaion from our tokenised dataset

def tokenize(sentence):
  return tokenizer.tokenize(sentence)

# .lowering and .stemming

from nltk.stem import PorterStemmer
ps = PorterStemmer()

def stemming(word):
  return ps.stem(word.lower())

def stemmingText(tokenizedSen):
  stemmedSen = []
  for word in tokenizedSen:
    stemmedSen.append(stemming(word))
  return stemmedSen


#.embedding
# bag of words embedding of words to be used in neural network
def bagOfWords(tokenizedSentecne,allWords):
    tempVector = []   
    for word in allWords:
      if word in tokenizedSentecne:
        tempVector.append(1)
      else:
        tempVector.append(0)
    return tempVector

    
# print(bagOfWords(['hi','my','is','krishna'],['hi','my','is','krishna','yadav']))