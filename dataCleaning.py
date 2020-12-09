from nltk.corpus import stopwords
from string import punctuation
from collections import Counter
from os import listdir

def file_open(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text

def clean_data(file):
	#split token by white space
	tokens = file.split()
	#remove punctuation form token
	table = str.maketrans('','',punctuation)
	tokens = [w.translate(table) for w in tokens]
	tokens = [word for word in tokens if word.isalpha()]
	stop_words = set(stopwords.words('english')) # removal of stopword
	tokens = [w for w in tokens if not w in stop_words]
	tokens = [word for word in tokens if len(word) > 1]
	return tokens

#adding vocabilary
def add_vocab(filename,vocab):
	doc = file_open(filename)
	tokens = clean_data(doc)
	vocab.update(tokens)
# this method will go to each file and do the cleaning and vocab
#process	
def process_dataset(direc,vocab,is_train):
	for filename in listdir(direc):
		if is_train and filename.startswith('cv9'):
			continue
		if not is_train and not filename.startswith('cv9'):
			continue
		path = direc + '/'+filename
		print(path)
		add_vocab(path,vocab)

def save_data(lines,filename):
	data = '\n'.join(lines)
	file = open(filename,'w')
	file.write(data)
	file.close()



vocab = Counter()
process_dataset('DataSet/Train/neg',vocab,True)
process_dataset('DataSet/Train/pos',vocab,True)
min_occurance = 2
tokens = [k for k,c in vocab.items() if c >= min_occurance]
print(len(tokens))
save_data(tokens,'vocab.txt');