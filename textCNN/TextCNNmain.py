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
    return list(wordset)

def transforinput(wordlist, x_input, length):
	unk = len(wordlist)
	vec_input = []
	for phrase in x_input:
		plist = phrase.split(' ')
		vlist = []
		for word in plist:
			ind = wordlist.index(word) if word in wordlist else unk
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
		embedding_dict = tf.Variable(tf.random_uniform([wordsetsize + 1, embedding_size], -1.0, 1.0), name="dict")
	return embedding_dict

def getinput(embedding_dict, x_input):
	embedding_input = tf.nn.embedding_lookup(embedding_dict, x_input)
	#拓展一维 变成4维 卷积输入 （训练集，句子，词向量，通道）
	embedding_input_expand = tf.expand_dims(embedding_input, -1)
	return embedding_input_expand

def ConvAndPool(fliter_sizes, fliter_num, embedding_size, conv_input, length):
	pooled_outputs=[]
	for fliter_size in fliter_sizes:
		with tf.name_scope('conv-pool-%s' % str(fliter_size)):
			fliter_shape = [fliter_size, embedding_size, 1, fliter_num]
			W = tf.Variable(tf.truncated_normal(fliter_shape, stddev=0.1), name="w")
			b = tf.Variable(tf.constant(0.1, shape=[fliter_num]), name="b")
			conv = tf.nn.conv2d(conv_input, W, strides=[1,1,1,1], padding='VALID', name="conv")
			h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
			pooled = tf.nn.max_pool(h, ksize=[1, length-fliter_size+1, 1, 1], strides=[1,1,1,1], padding="VALID", name="pool")
			pooled_outputs.append(pooled)
	total_feature_num = fliter_num * len(fliter_sizes)
	h_pool = tf.concat(pooled_outputs, 3)
	h_pool_flat = tf.reshape(h_pool, [-1, total_feature_num], name="flat")
	return h_pool_flat
			
def dropout(drop_input, drop_keep_pro):
	return tf.nn.dropout(drop_input, drop_keep_pro, name="dropout")

def fullyconnectNN(x_input, x_size, y_size, name):
	with tf.name_scope(name):
		W = tf.Variable(tf.truncated_normal([x_size, y_size], stddev=0.1), name="W")
		b = tf.Variable(tf.constant(0.1, shape=[y_size]), name="b")
		out = tf.nn.xw_plus_b(x_input, W, b, name="out")
	return out

def loss(logits, y_input):
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_input)
	loss = tf.reduce_mean(cross_entropy, name="loss")
	return loss

def train(loss, learningrate):
	global_step = tf.Variable(0, name="global_step", trainable=False)
	optimizer = tf.train.AdamOptimizer(learningrate)
	train_op = optimizer.minimize(loss, global_step=global_step, name="train_op")
	return train_op

def evalacc(logits, y_label):
	correct = tf.equal(tf.argmax(logits,1), tf.argmax(y_label,1))
	eval_correct = tf.reduce_mean(tf.cast(correct, tf.int32))
	return eval_correct	

def main():
	dataTexts, dataLabels = loadData('../data/kaggle_sentiment_analysis/train.tsv')
	length = maxlength(dataTexts)
	dataTexts = dataClean(dataTexts)[:10000]
	dataLabels = list(dataLabels)[:10000]
	num_classes = len(set(dataLabels))
	dataTexts, dataLabels, cvtexts, cvlabels = cat_for_train_and_cv(dataTexts, dataLabels, 0.8)
	wordlist = createWordlist(dataTexts)
	print("total datasets length: %d" % len(dataTexts))
	print("total labels classes num: %d" % num_classes)
	print("labels classes : " , collections.Counter(dataLabels))
	print("total words : %d" % len(wordlist))

	x_input = transforinput(wordlist, dataTexts, length)
	
	x_pl = tf.placeholder(tf.int32, [None, length], name = "x_pl")
	y_pl = tf.placeholder(tf.int32, [None], name = "y_pl")
	y_label = tf.one_hot(y_pl, num_classes)
	
	embedding_dict = getembedding(len(wordlist), 128)
	embedding_input = getinput(embedding_dict, x_pl)
	pool_out = ConvAndPool([3,4,5], 128, 128, embedding_input, length)
	drop_out = dropout(pool_out, 0.8)
	logits = fullyconnectNN(drop_out, 3*128, num_classes, "fnn1")
	loss_value = loss(logits, y_label)
	train_op = train(loss_value, 0.01)
	eval_result = evalacc(logits, y_label)

	init = tf.initialize_all_variables()	
	sess = tf.Session()
	sess.run(init)
	for step in range(3000):
		feed_dict = {x_pl : x_input, y_pl : dataLabels,}
		_, loss_v = sess.run([train_op, loss_value],
								feed_dict=feed_dict)
		if step % 100 == 0 :
			print ("step %d: loss = %.5f " % (step, loss_v))				
		if step % 500 == 0:
			feed_dict = {x_pl: transforinput(wordlist, cvtexts, length), y_pl : cvlabels,}
			print(sess.run([eval_result], feed_dict = feed_dict))

if __name__ == "__main__":
	main()

