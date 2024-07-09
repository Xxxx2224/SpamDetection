import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pickle
datasetdf=pd.read_csv('trspam.csv', on_bad_lines='skip')
df_docs = pd.DataFrame({'email': datasetdf.iloc[:,0].fillna(""),'durum': datasetdf.iloc[:,1].fillna("")})
nRow, nCol = df_docs.shape
print(f'There are {nRow} rows and {nCol} columns')
beşsatır=df_docs.head(5)
print(beşsatır.iloc[:,1])
tfidf=TfidfVectorizer(binary=False, ngram_range=(1,3))

def optimizasyon(dataset):
    dataset.dropna(axis = 0, how ='any')
    stop_words= set(stopwords.words('turkish'))
    for i in dataset.index:
        body=dataset.iloc[i,0]
        body=body.lower()
        body=re.sub(r'http\S+', '',body)
        body = (" ").join([word for word in body.split() if not word in stop_words])
        dataset.iloc[i,0]=body
    return dataset


def optimizasyon1(metin):

    stop_words = set(stopwords.words('turkish'))
    noktalamaIsaretleri = ['•', '!', '"', '#', '”', '“', '$', '%', '&', "'", '-', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '…']
    stop_words.update(noktalamaIsaretleri)

    body = metin
    body = body.lower()
    body = re.sub(r'http\S+', '', body)
    body = re.sub('\[[^]]*\]', '', body)
    body = (" ").join([word for word in body.split() if not word in stop_words])
    body = "".join([char for char in body if not char in noktalamaIsaretleri])
    return body
dataset=optimizasyon(df_docs)
yapay_zeka_modeli=LogisticRegression()
durum1=dataset.iloc[:,1]
email1=dataset.iloc[:,0]
print(email1)
email1_vec=tfidf.fit_transform(email1)
yapay_zeka_modeli.fit(email1_vec,durum1)
pickle.dump ( yapay_zeka_modeli , open ("egitilmis_model",'wb') )
print ( " Lojistik Regresyon modeli eğitildi ve kayıt edildi ! " )
import imaplib
import email
from email.header import decode_header

mail = imaplib.IMAP4_SSL("imap.gmail.com")
username = "gmail yazılacak"
password = "şifre yazılacak"

# Giriş yapma
mail.login(username, password)

mail.select("inbox")

status, messages = mail.search(None, "ALL")
mail_ids = messages[0].split()

body_list = []
for i in mail_ids[-10:]:
    
    status, msg_data = mail.fetch(i, "(RFC822)")
    
    for response_part in msg_data:
        if isinstance(response_part, tuple):
            msg = email.message_from_bytes(response_part[1])
            
            
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))

                    try:
                        body = part.get_payload(decode=True).decode()
                    except:
                        pass

                    if content_type == "text/plain" and "attachment" not in content_disposition:
                        body_list.append(body)
            else:
                content_type = msg.get_content_type()
                body = msg.get_payload(decode=True).decode()
                if content_type == "text/plain":
                    body_list.append(body)


mail.logout()
eposta_imap = []
i=0
print(body_list)
for a in body_list:
    eposta_imap=optimizasyon1(a)
    email_vec=tfidf.transform([a])
    print(email_vec)
    yapay_zeka_modeli=pickle.load ( open ( "egitilmis_model" , 'rb' ) )
    tahmin1=yapay_zeka_modeli.predict(email_vec)
    print(tahmin1) 

metin=input('Kontrol edilecek metni girin:')
metin=optimizasyon1(metin)
metin_vec=tfidf.transform([metin])
print('METİN VECTÖR')
print(metin_vec)
yapay_zeka_modeli=pickle.load ( open ( "egitilmis_model" , 'rb' ) ) 
tahmin=yapay_zeka_modeli.predict(metin_vec)
print(tahmin)






        





    
