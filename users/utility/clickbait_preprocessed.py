from django.conf import settings
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import pandas as pd
train_path = settings.MEDIA_ROOT + "//" + 'train.csv'
test_path = settings.MEDIA_ROOT + "//" + 'test.csv'

df = pd.read_csv(train_path)
dft = pd.read_csv(test_path)
df['title'].fillna('missing', inplace=True)
dft['title'].fillna('missing', inplace=True)

import re
from string import punctuation
def process_text2(title):
    result = title.replace('/','').replace('\n','')
    result = re.sub(r'[0-9]+','number', result)   # we are substituting all kinds of no. with word number
    result = re.sub(r'(\w)(\1{2,})', r'\1', result)  # \w matches one word/non word character
    result = re.sub(r'(?x)\b(?=\w*\d)\w+\s*', '', result)
    result = ''.join(word for word in result if word not in punctuation)  # removes all characters such as "!"#$%&'()*+, -./:;<=>?@[\]^_`{|}~"
    result = re.sub(r' +', ' ', result).lower().strip()
    return result

#removing the stopwords
from nltk.corpus import stopwords
stop = stopwords.words("english")
def cnt_stopwords(title):
    result1 = title.split()
    num1 =  len([word for word in result1 if word in stop])
    return num1

contractions = ['tis', 'aint', 'amnt', 'arent', 'cant', 'couldve', 'couldnt', 'couldntve',
                'didnt', 'doesnt', 'dont', 'gonna', 'gotta', 'hadnt', 'hadntve', 'hasnt',
                'havent', 'hed', 'hednt', 'hedve', 'hell', 'hes', 'hesnt', 'howd', 'howll',
                'hows', 'id', 'idnt', 'idntve', 'idve', 'ill', 'im', 'ive', 'ivent', 'isnt',
                'itd', 'itdnt', 'itdntve', 'itdve', 'itll', 'its', 'itsnt', 'mightnt',
                'mightve', 'mustnt', 'mustntve', 'mustve', 'neednt', 'oclock', 'ol', 'oughtnt',
                'shant', 'shed', 'shednt', 'shedntve', 'shedve', 'shell', 'shes', 'shouldve',
                'shouldnt', 'shouldntve', 'somebodydve', 'somebodydntve', 'somebodys',
                'someoned', 'someonednt', 'someonedntve', 'someonedve', 'someonell', 'someones',
                'somethingd', 'somethingdnt', 'somethingdntve', 'somethingdve', 'somethingll',
                'somethings', 'thatll', 'thats', 'thatd', 'thered', 'therednt', 'theredntve',
                'theredve', 'therere', 'theres', 'theyd', 'theydnt', 'theydntve', 'theydve',
                'theydvent', 'theyll', 'theyontve', 'theyre', 'theyve', 'theyvent', 'wasnt',
                'wed', 'wedve', 'wednt', 'wedntve', 'well', 'wontve', 'were', 'weve', 'werent',
                'whatd', 'whatll', 'whatre', 'whats', 'whatve', 'whens', 'whered', 'wheres',
                'whereve', 'whod', 'whodve', 'wholl', 'whore', 'whos', 'whove', 'whyd', 'whyre',
                'whys', 'wont', 'wontve', 'wouldve', 'wouldnt', 'wouldntve', 'yall', 'yalldve',
                'yalldntve', 'yallll', 'yallont', 'yallllve', 'yallre', 'yallllvent', 'yaint',
                'youd', 'youdve', 'youll', 'youre', 'yourent', 'youve', 'youvent']

def cnt_contract(title):
    result2 = title.split()
    num2 = len([word for word in result2 if word in contractions])
    return num2

question_words = ['who', 'whos', 'whose', 'what', 'whats', 'whatre', 'when', 'whenre', 'whens', 'couldnt',
        'where', 'wheres', 'whered', 'why', 'whys', 'can', 'cant', 'could', 'will', 'would', 'is',
        'isnt', 'should', 'shouldnt', 'you', 'your', 'youre', 'youll', 'youd', 'here', 'heres',
        'how', 'hows', 'howd', 'this', 'are', 'arent', 'which', 'does', 'doesnt']
def question_word(title):
    result3 = title.lower().split()
    if result3[0] in question_words:
        return 1
    else:
        return 0
    #return result3

def pos_tags(title):
    result4 = title.split()
    non_stop = [word for word in result4 if word not in stopwords.words("english")]
    pos = [part[1] for part in nltk.pos_tag(non_stop)]
    pos = " ".join(pos)
    return pos


import nltk
# progress bar
from tqdm import tqdm_notebook,tqdm
# instantiate
tqdm.pandas()
df['processed_headline']     = df['title'].progress_apply(process_text2)
df['question'] = df['title'].progress_apply(question_word)
df['num_words']       = df['title'].progress_apply(lambda x: len(x.split()))
df['part_speech']     = df['title'].progress_apply(pos_tags)
df['num_contract']    = df['title'].progress_apply(cnt_contract)
df['num_stop_words']  = df['title'].progress_apply(cnt_stopwords)
df['stop_word_ratio'] = df['num_stop_words']/df['num_words']
df['contract_ratio']  = df['num_contract']/df['num_words']

# from tqdm import tqdm_notebook,tqdm
#
# # instantiate
# # tqdm.pandas(tqdm_notebook)
# tqdm.pandas()
# dft['processed_headline']     = dft['title'].progress_apply(process_text2)
# dft['question'] = dft['title'].progress_apply(question_word)
# dft['num_words']       = dft['title'].progress_apply(lambda x: len(x.split()))
# dft['part_speech']     = dft['title'].progress_apply(pos_tags)
# dft['num_contract']    = dft['title'].progress_apply(cnt_contract)
# dft['num_stop_words']  = dft['title'].progress_apply(cnt_stopwords)
# dft['stop_word_ratio'] = dft['num_stop_words']/dft['num_words']
# dft['contract_ratio']  = dft['num_contract']/dft['num_words']

def pre_processed_data():
    df.drop(columns = ['num_contract','num_stop_words'])
    return df.head(10)


def start_ml_procedeing():
    df.drop(columns=['num_contract', 'num_stop_words'])
    # from sklearn.feature_extraction.text import TfidfVectorizer
    # tfidf = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word',
    #                         token_pattern=r'\w{1,}', ngram_range=(1, 5), use_idf=1, smooth_idf=1, sublinear_tf=1)
    # X_train_headline = tfidf.fit_transform(df['processed_headline'])  # 1
    # X_test_headline = tfidf.transform(dft['processed_headline'])
    # from sklearn.feature_extraction.text import CountVectorizer
    # from sklearn.preprocessing import StandardScaler
    # cv = CountVectorizer()
    # sc = StandardScaler(with_mean=False)
    # X_train_pos = cv.fit_transform(df['part_speech'])
    # X_train_pos_sc = sc.fit_transform(X_train_pos)  # 2
    # X_test_pos = cv.transform(dft['part_speech'])
    # X_test_pos_sc = sc.transform(X_test_pos)
    #

    df.drop( columns = ['title','processed_headline','part_speech','text']).values
    print(df.columns)
    # print("===?Df:",df.head())
    X = df[['id', 'title', 'text', 'processed_headline', 'question',
       'num_words', 'part_speech', 'num_contract', 'num_stop_words',
       'stop_word_ratio', 'contract_ratio']]
    Y = df[['label']]
    X = df.drop( columns = ['title','label','processed_headline','part_speech','text']).values

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
    sc = StandardScaler()
    X_train_val_sc = sc.fit_transform(x_train)  # 3
    X_test_val_sc = sc.transform(x_test)
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.fit_transform(y_test)
    # Model Building
    from sklearn.linear_model import LogisticRegression
    lg = LogisticRegression()
    lg.fit(X_train_val_sc,y_train)
    y_pred = lg.predict(X_test_val_sc)
    from sklearn.metrics import classification_report
    lg_cr = classification_report(y_test,y_pred,output_dict=True)
    lg_cr = pd.DataFrame(lg_cr).transpose()
    lg_cr = pd.DataFrame(lg_cr)
    lg_cr = lg_cr.to_html

    # Random Forest

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier()
    rf.fit(X_train_val_sc, y_train)
    y_pred = rf.predict(X_test_val_sc)
    from sklearn.metrics import classification_report
    rf_cr = classification_report(y_test, y_pred, output_dict=True)
    rf_cr = pd.DataFrame(rf_cr).transpose()
    rf_cr = pd.DataFrame(rf_cr)
    rf_cr = rf_cr.to_html

    # Stochastic Gradient Descent
    from sklearn.ensemble import GradientBoostingClassifier
    gr = GradientBoostingClassifier()
    gr.fit(X_train_val_sc, y_train)
    y_pred = gr.predict(X_test_val_sc)
    from sklearn.metrics import classification_report
    gr_cr = classification_report(y_test, y_pred, output_dict=True)
    gr_cr = pd.DataFrame(gr_cr).transpose()
    gr_cr = pd.DataFrame(gr_cr)
    gr_cr = gr_cr.to_html

    # Ensemble

    from sklearn.ensemble import AdaBoostClassifier
    en = AdaBoostClassifier()
    en.fit(X_train_val_sc, y_train)
    y_pred = en.predict(X_test_val_sc)
    from sklearn.metrics import classification_report
    en_cr = classification_report(y_test, y_pred, output_dict=True)
    en_cr = pd.DataFrame(en_cr).transpose()
    en_cr = pd.DataFrame(en_cr)
    en_cr = en_cr.to_html

    return lg_cr,rf_cr,gr_cr,en_cr

