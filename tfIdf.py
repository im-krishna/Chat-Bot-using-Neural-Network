
def tf_idf(all_sentences,all_words):
    bag_Of_Words = []
    for sen in all_sentences:
        tempVector = {} 
        for word in all_words:
            if word in sen:
                tempVector[word]=1
            else:
                tempVector[word]=0
        bag_Of_Words.append(tempVector)
    #we have our bag of words as dictionary now
    
    def computeTF(wordDict, bow):
        tfDict = {}
        bowCount = len(bow)
        for word, count in wordDict.items():
            tfDict[word] = count/float(bowCount)
        return tfDict
    
    tfBow = []
    index=0
    for wordDict in bag_Of_Words:
        tfBow.append(computeTF(wordDict,all_sentences[index]))
        index=index+1
        
    
    def computeIDF(docList):
        import math
        idfDict = {}
        N = len(docList)
        
        idfDict = dict.fromkeys(docList[0].keys(), 0)
        for doc in docList:
            for word, val in doc.items():
                if val > 0:
                    idfDict[word] += 1
        
        for word, val in idfDict.items():
            idfDict[word] = math.log10(N / float(val))
            
        return idfDict 
    
    idfs = computeIDF(bag_Of_Words) 
    
    def computeTFIDF(tfBow, idfs):
        tfidf = {}
        for word, val in tfBow.items():
            tfidf[word] = val*idfs[word]
        return tfidf

    tfidfbow = []
    for x in tfBow:
        tfidfbow.append(computeTFIDF(x,idfs))
    
    return convert(tfidfbow)

#first we need to convert to array of arrays
#then we need to convert each array into numpy array 

def convert(tfidfbow):
    tempL = []
    import numpy as np
    for d in tfidfbow:
        tempL.append(np.array(list(d.values())))
    return tempL
    
# print(tf_idf([['The', 'cat', 'sat', 'on', 'my', 'face'],['The', 'dog', 'sat', 'on', 'my', 'bed']],['The', 'bed', 'cat', 'dog', 'face', 'my', 'on', 'sat']))
    
      
