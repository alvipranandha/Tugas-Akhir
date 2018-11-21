import json
import maxent
import time
before = time.strftime('%X %x')
print (before)
print ("importing dependencies from nltk...")

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.isri import ISRIStemmer
from nltk.stem.snowball import SnowballStemmer
import re

numIterations = 10

stopwords = set(stopwords.words('english'))
porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
lancaster_stemmer = LancasterStemmer()
isri_stemmer = ISRIStemmer()
snowball_stemmer = SnowballStemmer('english')

def readData(fileName) :
    hasil = []
    with open(fileName) as json_data:
        for line in json_data :
            data = json.loads(line.split("\n")[0])
            hasil.append([data['reviewText'], data['label']])
    return hasil

def preprocessing(data) :
    hasil = []
    for line in data :
        line[0] = word_tokenize(line[0].lower()) #Tokenisasi
        #line[0] = nltk.pos_tag(line[0])  # Pos Tag

        tmp = []
        for word in line[0] :
            word = re.sub(r"\d|-|,|=|\$|\(|\)|'|`|:|!|\"|/|\?|\+|\.| |\*", "",word) #Tanda Baca
            #word = wordnet_lemmatizer.lemmatize(word)                          #Lemmatization
            #word = porter_stemmer.stem(word)                                   #StemmingPorter
            #word = lancaster_stemmer.stem(word)                                #StemmingLancaster
            #word = isri_stemmer.stem(word)                                     #StemmingIsri
            word = snowball_stemmer.stem(word)                                 #StemmingSnowball
            if len(word) > 3:                         #Stopword word not in stopwords and
                tmp.append(word)

        line[0] = tmp
        hasil.append(line)
    return hasil

def list_to_dict(words_list):
    return dict([(word, True) for word in words_list])

def accuracy(actual, predicted) :
    count = 0
    for i in range(len(actual)) :
        if actual[i] [:5] == predicted[i] [:5]:
            count+=1
    return count/len(actual)

def F1Score(actual, prediction) :
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for n in range(len(actual)) :
        if actual[n] == 'Positive' and  prediction[n] == 'Positive':
            tp+=1
        elif actual[n] == 'Negative' and  prediction[n] == 'Positive':
            fp+=1
        elif actual[n] == 'Positive' and  prediction[n] == 'Negative':
            fn+=1
        elif actual[n] == 'Negative' and  prediction[n] == 'Negative':
            tn+=1

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    #f1 = (2*tp)/((2*tp)+fp+fn)
    f1 = (2* (precision*recall) / (precision+recall))

    print("Precision :",precision)
    print("Recall    :",recall)
    print("F1-Score  :", f1)
    return precision, recall, f1



#Read Data
data_train = readData("./Data_Training.json")
data_test = readData("./Data_Testing.json")

#Preprocessing
data_train = preprocessing(data_train)
data_test = preprocessing(data_test)

#Training
training_set_formatted = [(list_to_dict(element[0]), element[1]) for element in data_train]
test_set_formatted = [(list_to_dict(element[0]), element[1]) for element in data_test]
algorithm = maxent.MaxentClassifier.ALGORITHMS[0] #['GIS', 'IIS', 'MEGAM', 'TADM']
classifier = maxent.MaxentClassifier.train(training_set_formatted, algorithm='GIS', max_iter=numIterations)

#Print(training_set_formatted)

print("")
print("Top 10 Most Informative Features (Positive)")
print(classifier.show_most_informative_features(10, show="pos"))
print("Top 10 Most Informative Features (Negative)")
print(classifier.show_most_informative_features(10, show="neg"))

#Testing
predicted = []
actual = []
for review in test_set_formatted:
    predicted.append(classifier.classify(review[0]))
    actual.append(review[1])

# print(len(test_set_formatted), len(predicted), len(actual))
print("Actual        :", actual)
print("Predicted     :", predicted)
print("Nilai Akurasi :", accuracy(actual, predicted))
print("F1 Score      :", F1Score(actual, predicted))