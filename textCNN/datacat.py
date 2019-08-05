import pandas as pd
from sklearn.model_selection import train_test_split

def loadData(filepath):
	datasets = pd.read_csv(filepath, sep='\t')
	dataTexts = datasets['Phrase']
	dataLabels = datasets['Sentiment']
	return dataTexts, dataLabels
def savedata(arr, filename):
	arr_df = pd.DataFrame(arr)
	arr_df.to_csv(filename, sep='\t', index=False, header=["PhraseId","Phrase","Sentiment"])
def main():
	data_x, data_y = loadData("./data/train.tsv")
	data_x = list(data_x)
	data_y = list(data_y)
	x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2)
	len_train = len(x_train)
	len_test = len(x_test)
	print(len(data_x))
	print("train: %d, test: %d" % (len_train,len_test))
	
	train_csv=[]
	for i in range(len_train):
		train_csv.append([i, x_train[i], y_train[i]])
	savedata(train_csv, "data/train_data.tsv")
	
	test_csv=[]
	for i in range(len_test):
		test_csv.append([i, x_test[i], y_test[i]])
	savedata(test_csv, "data/cv_data.tsv")

if __name__ == "__main__":
	main()
