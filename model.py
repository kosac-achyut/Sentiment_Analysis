from string import punctuation
from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

def load_file(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text

def clean_data(file,vocab):
	tokens = file.split()
	table = str.maketrans('','',punctuation)
	tokens = [w.translate(table) for w in tokens]
	tokens = [w for w in tokens if w in vocab]
	tokens = ' '.join(tokens)
	return tokens


def process_file(direc, vocab, is_train):
	file_list = list()
	for filename in listdir(direc):
		if is_train and filename.startswith('cv9'):
			continue
		if not is_train and not filename.startswith('cv9'):
			continue
		path = direc + '/' + filename
		file = load_file(path)
		tokens = clean_data(file,vocab)
		file_list.append(tokens)
	return file_list


vocab_filename = 'vocab.txt'
vocab = load_file(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

positive_file = process_file('DataSet/pos/pos', vocab,True)
negative_file = process_file('DataSet/neg/neg', vocab,True)
train_file = negative_file + positive_file

tokenizer  = Tokenizer()
tokenizer.fit_on_texts(train_file)
encoded_file = tokenizer.texts_to_sequences(train_file)
max_length = max([len(s.split()) for s in train_file])
Xtrain = pad_sequences(encoded_file, maxlen= max_length,padding='post')
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])


positive_file_test = process_file('DataSet/pos/pos', vocab,False)
negative_file_test = process_file('DataSet/neg/neg', vocab,False)
test_file = negative_file_test + positive_file_test

encoded_file_test = tokenizer.texts_to_sequences(test_file)
Xtest = pad_sequences(encoded_file_test, maxlen=max_length, padding='post')
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])
vocab_size = len(tokenizer.word_index) + 1

#model CNN layer 6

model = Sequential()
model.add(Embedding(vocab_size, 150, input_length=max_length))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(Xtrain, ytrain, epochs=10, verbose=2)
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))

