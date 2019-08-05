import tensorflow as tf
import numpy as np
import pandas as pd
import re
import random
import collections
import gensim.downloader as api

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

def loadworddict(dataTexts):
	print("loading google news 300 ...")
	wv = api.load("word2vec-google-news-300")
	print("load finish")
	worddict = {}
	for phrase in dataTexts:
		plist = set(phrase.split(' '))
		for word in plist:
			try:
				worddict[word] = wv[word]
			except KeyError:
				worddict[word] = np.random.uniform(-1, 1, size=300)
	worddict['UNK'] = wv['UNK']
	del wv
	return worddict

def transforinput(worddict, x_input, length):
	vec_input = []
	for phrase in x_input:
		plist = phrase.split(' ')
		vlist = []
		for word in plist:
			wv = worddict.get(word, np.random.uniform(-1, 1, size=300))
			vlist.append(wv)
			if (len(vlist) == length):
				break
		while len(vlist) < length:
			vlist.append(worddict['UNK'])
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
	x_batch = data_x[index: index + batch_size]
	y_batch = data_y[index: index + batch_size]
	index += batch_size
	index %= len(data_x)
	return x_batch, y_batch, index

def dropout(drop_input, drop_keep_pro):
	return tf.nn.dropout(drop_input, drop_keep_pro, name="dropout")

def fullyconnectNN(x_input, x_size, y_size, l2_reg_lambda):
	W = tf.Variable(tf.truncated_normal([x_size, y_size], stddev=0.1), name="fnn_W")
	b = tf.Variable(tf.constant(0.1, shape=[y_size]), name="b")
	W_l2_loss = tf.contrib.layers.l2_regularizer(l2_reg_lambda)(W)
	tf.add_to_collection('losses', W_l2_loss)
	out = tf.nn.xw_plus_b(x_input, W, b, name="fnn_out")
	return out

def loss(logits, y_input):
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_input)
	cross_loss = tf.reduce_mean(cross_entropy, name="loss")
	tf.add_to_collection('losses', cross_loss)
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

def doeval(x_data, label_data, eval_result, worddict, length, sess, x_pl, y_pl, keep_prob_pl):
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

def doprec(filename, worddict, length, x_pl, y_pl, keep_prob_pl, sess, logits):
    datasets = pd.read_csv(filepath, sep='\t')
    PhraseId = datasets['PhraseId']
    datainput = datasets['Phrase']
    datainput = dataClean(datainput)
	PhraseId = list(PhraseId)
	feed_dict = {x_pl:transforinput(worddict, datainput, length), keep_prob_pl:1.0}
	result = sess.run(logits, feed_dict = feed_dict)
	out = []
	for i in range (len(PhraseId)):
		out.append(PhraseId[i], result[i])
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
	worddict = loadworddict(dataTexts)
	
	cvtexts, cvlabels = loadData('./data/cv_data.tsv')
	cvTexts = dataClean(cvtexts)
	cvLabels = list(cvlabels)
	index = 0
	print("total datasets length: %d" % len(dataTexts))
	print("total labels classes num: %d" % num_classes)
	print("labels classes : " , collections.Counter(dataLabels))
	
	x_pl = tf.placeholder(tf.float32, [None, length, 300], name = "x_pl")
	y_pl = tf.placeholder(tf.int32, [None], name = "y_pl")
	keep_prob_pl = tf.placeholder(tf.float32, name="keep_prob")
	x_input = tf.expand_dims(x_pl, -1)
	y_label = tf.one_hot(y_pl, num_classes)
	
	pool_out = ConvAndPool([3,4,5], 128, 300, x_input, length)
	drop_out = dropout(pool_out, keep_prob_pl)
	logits = fullyconnectNN(drop_out, 3*128, num_classes, 0.01)
	loss_value = loss(logits, y_label)
	train_op = train(loss_value, 0.0001)
	eval_result = evalacc(logits, y_label)

	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	index = 0
	for step in range(200000):
		x_batch, y_batch, index = getbatch(dataTexts, dataLabels, 64, index)
		feed_dict = {x_pl : transforinput(worddict, x_batch, length), y_pl : y_batch, keep_prob_pl : 0.5,}
		_, loss_v = sess.run([train_op, loss_value],
								feed_dict=feed_dict)
		if step % 200 == 0:
			print ("step %d: loss = %.5f " % (step, loss_v))				
		if step % 2000 == 0:
			print("train:")
			doeval(dataTexts, dataLabels, eval_result, worddict, length, sess, x_pl, y_pl, keep_prob_pl)
			print("cv:")
			doeval(cvtexts, cvlabels, eval_result, worddict, length, sess, x_pl, y_pl, keep_prob_pl)
	
	result = doprec("./data/test.tsv")
	saveresult (result, "sub.csv")
if __name__ == "__main__":
	main()

