import tensorflow as tf
import numpy as np
import pandas as pd
import re
import random
import collections

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

def maxlength(dataText):
	length = 0	
	for phrase in dataText:
		length = max(length, len(phrase.split(' ')))
	return length

def createWordlist(dataTexts):
	wordset = set()
	print("create wordset ing...")
	for phrase in dataTexts:
		plist = set(phrase.split(' '))
		wordset = wordset.union(plist)
	worddict = {}
	wordset = list(wordset)
	for i in range(len(wordset)):
		worddict[wordset[i]] = i
	worddict['UNKWORD'] = len(wordset)
	numword = len(wordset) + 1
	del wordset
	return worddict, numword

def transforinput(worddict, x_input, length):
	vec_input = []
	unk = worddict['UNKWORD']
	for phrase in x_input:
		plist = phrase.split(' ')
		vlist = []
		for word in plist:
			if (len(vlist) == length):
				break
			ind = worddict.get(word, unk)
			vlist.append(ind)
		while len(vlist) < length:
			vlist.append(unk)
		vec_input.append(vlist)
	return vec_input

def cat_for_train_and_cv(datainput, datalabel, rate):
	total = len(datainput)
	num = rate * total
	index = [i for i in range(total)]
	random.shuffle(index)
	traininput=[]
	trainlabel=[]
	cvinput=[]
	cvlabel=[]
	for i in range(total):
		if i < num:
			traininput.append(datainput[index[i]])
			trainlabel.append(datalabel[index[i]])
		else :
			cvinput.append(datainput[index[i]])
			cvlabel.append(datalabel[index[i]])
	return traininput, trainlabel, cvinput, cvlabel

def getembedding(wordsetsize, embedding_size):
	with tf.name_scope("embedding"):
		embedding_dict = tf.Variable(tf.random.uniform([wordsetsize + 1, embedding_size], -1.0, 1.0), name="dict")
	return embedding_dict

def getinput(embedding_dict, x_input):
	embedding_input = tf.nn.embedding_lookup(embedding_dict, x_input)
	embedding_input_expand = tf.expand_dims(embedding_input, -1)
	return embedding_input_expand

def ConvAndPool(fliter_sizes, fliter_num, embedding_size, conv_input, length):
	pooled_outputs=[]
	for fliter_size in fliter_sizes:
		with tf.name_scope('conv-pool-%s' % str(fliter_size)):
			fliter_shape = [fliter_size, embedding_size, 1, fliter_num]
			W = tf.Variable(tf.random.truncated_normal(fliter_shape, stddev=0.1), name="w")
			b = tf.Variable(tf.constant(0.1, shape=[fliter_num]), name="b")
			conv = tf.nn.conv2d(conv_input, W, strides=[1,1,1,1], padding='VALID', name="conv")
			h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
			pooled = tf.nn.max_pool2d(h, ksize=[1, length-fliter_size+1, 1, 1], strides=[1,1,1,1], padding="VALID", name="pool")
			pooled_outputs.append(pooled)
	total_feature_num = fliter_num * len(fliter_sizes)
	h_pool = tf.concat(pooled_outputs, 3)
	h_pool_flat = tf.reshape(h_pool, [-1, total_feature_num], name="flat")
	return h_pool_flat

def getbatch(data_x, data_y, batch_size, index):
	if (index + batch_size > len(data_x)):
		index = 0
	x_batch = data_x[index: index + batch_size]
	y_batch = data_y[index: index + batch_size]
	index += batch_size
	return x_batch, y_batch, index

def dropout(drop_input, drop_keep_pro):
	return tf.nn.dropout(drop_input, drop_keep_pro, name="dropout")

def fullyconnectNN(x_input, x_size, y_size, l2_reg_lambda):
	W = tf.Variable(tf.truncated_normal([x_size, y_size], stddev=0.1))
	b = tf.Variable(tf.constant(0.1, shape=[y_size]))
	W_l2_loss = tf.contrib.layers.l2_regularizer(l2_reg_lambda)(W)
	tf.add_to_collection('losses', W_l2_loss)
	out = tf.nn.xw_plus_b(x_input, W, b)
	return out

def loss(logits, y_input):
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_input)
	xloss = tf.reduce_mean(cross_entropy, name="xloss")
	tf.add_to_collection('losses', xloss)
	loss = tf.add_n(tf.get_collection('losses'))
	return loss

def train(loss, learningrate):
	global_step = tf.Variable(0, name="global_step", trainable=False)
	optimizer = tf.train.AdamOptimizer(learningrate)
	train_op = optimizer.minimize(loss, global_step=global_step, name="train_op")
	return train_op

def evalacc(logits, y_label):
	correct = tf.equal(tf.argmax(logits,1), tf.argmax(y_label, 1))
	eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
	return eval_correct	

def doeval(x_data, label_data, worddict, length, eval_result, sess, x_pl, y_pl, keep_prob_pl):
	lendata = len(x_data)
	numbatch = lendata // 128
	index = 0
	accnum = 0
	totalnum = 0
	for i in range(numbatch):
		x_batch, y_batch, index = getbatch(x_data, label_data, 128, index)
		feed_dict = {x_pl: transforinput(worddict, x_batch, length), y_pl : y_batch, keep_prob_pl : 1.0}
		totalnum += 128
		accnum += sess.run(eval_result, feed_dict = feed_dict)
	print("trun num: %d, total num: %d, acc: %.5lf" % (accnum, totalnum, accnum/totalnum))

def doprec(filepath, worddict, length, x_pl, y_pl, keep_prob_pl, sess, logits):
	datasets = pd.read_csv(filepath, sep='\t')
	PhraseId = datasets['PhraseId']
	datainput = datasets['Phrase']
	datainput = dataClean(datainput)
	PhraseId = list(PhraseId)
	lendata = len(PhraseId)
	print("testtotalnum: %d" % lendata)
	out = []
	numbatch = lendata // 128
	index = 0
	totalnum = 0
	for i in range(numbatch):
		x_batch = datainput[index : index + 128]
		feed_dict = {x_pl:transforinput(worddict, x_batch, length), keep_prob_pl:1.0}
		res = tf.argmax(logits, 1)
		result = sess.run(res, feed_dict = feed_dict)
		for j in range (128):
			out.append([PhraseId[index + j], result[j]])
		print("finish %d" % index)
		index += 128
	if (index != lendata) :
		x_batch = datainput[index : lendata]
		feed_dict = {x_pl:transforinput(worddict, x_batch, length), keep_prob_pl:1.0}
		res = tf.argmax(logits, 1)
		result = sess.run(res, feed_dict = feed_dict)
		for j in range (lendata - index):
			out.append([PhraseId[index + j], result[j]])
		print("finish %d" % index)
		
	return out
	
def savedata(arr, filename):
	arr_df = pd.DataFrame(arr)
	arr_df.to_csv(filename, sep=',', index=False, header=["PhraseId","Sentiment"])

def main():
	dataTexts, dataLabels = loadData('./data/train_data.tsv')
	length = maxlength(dataTexts)
	dataTexts = dataClean(dataTexts)
	dataLabels = list(dataLabels)
	num_classes = len(set(dataLabels))
	#dataTexts, dataLabels, cvtexts, cvlabels = cat_for_train_and_cv(dataTexts, dataLabels, 0.8)
	cvtexts, cvlabels = loadData('./data/cv_data.tsv')
	cvtexts = dataClean(cvtexts)
	cvlabels = list(cvlabels)

	worddict, wordnum = createWordlist(dataTexts)
	print("total datasets length: %d" % len(dataTexts))
	print("total labels classes num: %d" % num_classes)
	print("labels classes : " , collections.Counter(dataLabels))
	print("total words : %d" % wordnum)
	index = 0
	#x_input = transforinput(wordlist, dataTexts, length)
	#cv_input = transforinput(wordlist, cvtexts, length)
	#del dataTexts
	
	x_pl = tf.placeholder(tf.int32, [None, length], name = "x_pl")
	y_pl = tf.placeholder(tf.int32, [None], name = "y_pl")
	keep_prob_pl = tf.placeholder(tf.float32, name="keep_prob")
	y_label = tf.one_hot(y_pl, num_classes)
	
	embedding_dict = getembedding(wordnum, 256)
	embedding_input = getinput(embedding_dict, x_pl)
	pool_out = ConvAndPool([3,4,5,6], 128, 256, embedding_input, length)
	drop_out = dropout(pool_out, keep_prob_pl)
	h1 = fullyconnectNN(drop_out, 4*128, 256, 0.05)
	d1 = dropout(h1, keep_prob_pl)
	logits = fullyconnectNN(d1, 256, num_classes, 0.05)
	loss_value = loss(logits, y_label)
	train_op = train(loss_value, 0.0001)
	eval_result = evalacc(logits, y_label)

	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	index = 0
	for step in range(100000):
		x_batch, y_batch, index = getbatch(dataTexts, dataLabels, 128, index)
		feed_dict = {x_pl : transforinput(worddict, x_batch, length), y_pl : y_batch, keep_prob_pl : 0.5,}
		_, loss_v = sess.run([train_op, loss_value],
								feed_dict=feed_dict)
		if step % 200 == 0:
			print ("step %d: loss = %.5f " % (step, loss_v))				
		if step % 2000 == 0:
			print ("train:")
			doeval(dataTexts, dataLabels, worddict, length, eval_result, sess, x_pl, y_pl, keep_prob_pl)
			print ("test:")
			doeval(cvtexts, cvlabels, worddict, length, eval_result, sess, x_pl, y_pl, keep_prob_pl)
	
	result = doprec('./data/test.tsv', worddict, length, x_pl, y_pl, keep_prob_pl, sess, logits)	
	savedata(result, './data/sub.csv')
if __name__ == "__main__":
	main()

