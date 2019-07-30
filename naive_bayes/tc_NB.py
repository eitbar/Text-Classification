import pandas as pd
import sys
import math, random
import re
import collections
import numpy as np
################################################################################
def loadData(filepath):
	datasets = pd.read_csv(filepath, sep='\t')
	dataTexts = datasets['Phrase']
	dataLabels = datasets['Sentiment']
	return dataTexts, dataLabels

def dataClean(dataText):
	data = []
	for phrase in dataText:
		phrase = re.sub('[^a-zA-Z0-9 ]','',phrase)
		phrase = phrase.lower()
		data.append(' '.join(phrase.strip().split(r'\s+')))
	return data

def createWordset(dataTexts):
	wordset = set()
	print("create wordset ing...")
	for phrase in dataTexts:
		plist = set(phrase.split(' '))
		wordset = wordset.union(plist)
	return list(wordset)

def phrase2vec(wordset, phrase):
	vec = [0]*len(wordset)
	plist = phrase.split(' ')
	for i in plist:
		if i in wordset:
			vec[wordset.index(i)] += 1
	return vec

def trainNB(wordset, trainMatrix, trainLabels, alpha = 1):
	numTexts = len(trainMatrix)
	classes = set(trainLabels)
	numwords = len(wordset)
	print("train begin")
	countClass = dict(collections.Counter(trainLabels))
	countword = dict()
	for label in classes:
		countword[label] = np.zeros(numwords)
	print("label count finish")
	for i in range(numTexts):
		countword[trainLabels[i]] += trainMatrix[i]
	print("word count finish")

	NBclassProbability = dict()
	for label in classes:
		NBclassProbability[label] = float(countClass[label])/len(trainLabels)
	print("class probability finish")
	NBwordProbability = dict()
	for label in classes:
		NBwordProbability[label] = (countword[label] + alpha)/(sum(countword[label]) + alpha*numwords) 
	print("word in label probability finish")
	return NBclassProbability, NBwordProbability

def classifyNB(phrase, NBclassProbability, NBwordProbability, wordset, labelsets):
	prodict = dict()
	pro = 0
	for label in labelsets:
		tmppro = NBclassProbability[label]
		plist = phrase.split(' ')
		for i in plist:
			if i in wordset:
				tmppro *= NBwordProbability[label][wordset.index(i)]
		if (tmppro > pro):
			ans = label
			pro = tmppro
	return ans

def cat_for_train_and_cv(datainput, datalabel, p1, p2):
	total = len(datainput)
	num1 = p1 * total
	num2 = p2 * total
	index = [i for i in range(total)]
	random.shuffle(index)
	traininput=[]
	trainlabel=[]
	cvinput=[]
	cvlabel=[]
	for i in range(total):
		if i < num1:
			traininput.append(datainput[index[i]])
			trainlabel.append(datalabel[index[i]])
		else :
			cvinput.append(datainput[index[i]])
			cvlabel.append(datalabel[index[i]])
	return traininput, trainlabel, cvinput, cvlabel
###########################################################

def main():
	dataTexts, dataLabels = loadData('../data/train.tsv')
	dataTexts = dataClean(dataTexts)
	dataLabels = list(dataLabels)
	dataTexts = dataTexts[:10000]
	dataLabels = dataLabels[:10000]
	dataTexts ,dataLabels ,cvtexts, cvlabels = cat_for_train_and_cv(dataTexts, dataLabels, 0.8, 0.3)
	wordsets = createWordset(dataTexts)
	print("total datasets length: %d" % len(dataTexts))
	print("total labels classes num: %d" % len(set(dataLabels)))
	print("labels classes : " , collections.Counter(dataLabels))
	print("total words : %d" % len(wordsets))

	trainMatrix = []
	for i in range(len(dataTexts)):
		trainMatrix.append(phrase2vec(wordsets, dataTexts[i]))
		if i%100 == 0:
			print(i)
	#trainMatrix = [phrase2vec(wordsets, phrase) for phrase in dataTexts]
	
	del dataTexts
	nb1, nb2 = trainNB(wordsets, trainMatrix, dataLabels, alpha = 1)
	del trainMatrix

	cvprec = []
	for i in range(len(cvtexts)):
		cvprec.append(classifyNB(cvtexts[i], nb1, nb2, wordsets, set(dataLabels)))
		if i%100 == 0:
			print(i)
	#cvprec = [classifyNB(phrase, nb1, nb2, wordsets, set(dataLabels)) for phrase in cvtexts]
	t = 0
	for i in range(len(cvprec)):
		if cvprec[i] == cvlabels[i]:
			t+=1
	print (t/len(cvprec))
	# 0.609 ~ 0.625
if __name__=="__main__":
	main()
