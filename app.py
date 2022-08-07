import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
model = pickle.load(open('project5_decision -tree.pkl','rb')) 
model = pickle.load(open('project5_k-nearest.pkl','rb'))  
model = pickle.load(open('project5_random-forest.pkl','rb')) 
model = pickle.load(open('project5_svm.pkl','rb'))  
model = pickle.load(open('project5_naive-base.pkl','rb')) 

def review(text):
  df = pd.read_csv('NLP dataset 1.csv')
  import re
  import nltk
  nltk.download('stopwords')
  from nltk.corpus import stopwords
  from nltk.stem.porter import PorterStemmer
  corpus = []
  for i in range(0, 479):
    review = re.sub('[^a-zA-Z]', ' ', df['text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
  from sklearn.feature_extraction.text import CountVectorizer
  cv = CountVectorizer(max_features = 1500)
  X = cv.fit_transform(corpus).toarray()
  import re
  review = re.sub('[^a-zA-Z]', ' ', text)
  review=review.lower()
  import nltk
  nltk.download('stopwords')
  from nltk.corpus import stopwords
  review = review.split()
  review1 = [word for word in review if not word in set(stopwords.words('english'))]
  from nltk.stem.porter import PorterStemmer
  ps = PorterStemmer()
  review = [ps.stem(word) for word in review1 if not word in set(stopwords.words('english'))]
  review2 = ' '.join(review)
  X = cv.transform(review).toarray()
  input_pred = model.predict(X)
  input_pred = input_pred.astype(int)
  print(input_pred)
  if input_pred[0]==1:
    result= "Review is Positive"
  else:
    result="Review is negative" 

 
    
  return result

def main():
    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;"Summer Internship 2022</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
st.header("Text review system")
text = st.text_area("Write Text")
if st.button("Naive Bayes"):
  result=review(text)
  st.success('Model has predicted {}'.format(result))
if st.button("K-Nearest"):
  result=review(text)
  st.success('Model has predicted {}'.format(result))
if st.button("Random Forest"):
  result=review(text)
  st.success('Model has predicted {}'.format(result))
if st.button("Decision Tree"):
  result=review(text)
  st.success('Model has predicted {}'.format(result))
if st.button("SVM"):
  result=review(text)
  st.success('Model has predicted {}'.format(result))
      
if st.button("About"):
  st.subheader("Developed by Abhishek Kumar Singh")
  st.subheader("Student , Department of Computer Engineering")

if __name__=='__main__':
  main()
