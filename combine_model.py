from string import punctuation
from os import listdir
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import matplotlib.pyplot as plt

def load_file(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text


def clean_file(doc, vocab):
	tokens = doc.split()
	table = str.maketrans('', '', punctuation)
	tokens = [w.translate(table) for w in tokens]
	tokens = [w for w in tokens if w in vocab]
	tokens = ' '.join(tokens)
	return tokens

def process_file(directory, vocab, is_trian):
	documents = list()
	
	for filename in listdir(directory):
		
		if is_trian and filename.startswith('cv9'):
			continue
		if not is_trian and not filename.startswith('cv9'):
			continue
		path = directory + '/' + filename
		doc = load_file(path)
		tokens = clean_file(doc, vocab)
		documents.append(tokens)
	return documents


def load_embedding(filename):
	
	file = open(filename,'r')
	lines = file.readlines()[1:]
	file.close()

	embedding = dict()
	for line in lines:
		parts = line.split()
		
		embedding[parts[0]] = asarray(parts[1:], dtype='float32')
	return embedding


def get_weight_matrix(embedding, vocab):
	
	vocab_size = len(vocab) + 1
	
	weight_matrix = zeros((vocab_size, 100))

	for word, i in vocab.items():
		weight_matrix[i] = embedding.get(word)
	return weight_matrix


vocab_filename = 'vocab.txt'
vocab = load_file(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)


positive_docs = process_file('DataSet/pos/pos', vocab, True)
negative_docs = process_file('DataSet/neg/neg', vocab, True)
train_docs = negative_docs + positive_docs


tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_docs)


encoded_docs = tokenizer.texts_to_sequences(train_docs)
max_length = max([len(s.split()) for s in train_docs])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])


positive_docs = process_file('DataSet/pos/pos', vocab, False)
negative_docs = process_file('DataSet/neg/neg', vocab, False)
test_docs = negative_docs + positive_docs
encoded_docs = tokenizer.texts_to_sequences(test_docs)
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])

vocab_size = len(tokenizer.word_index) + 1


raw_embedding = load_embedding('embedding_word2vec.txt')
embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index)
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_vectors], input_length=max_length, trainable=False)


model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(Xtrain, ytrain, epochs=10, verbose=2)
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))
plt.plot(Xtest,ytest)
