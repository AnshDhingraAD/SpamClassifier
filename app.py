import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
ps=PorterStemmer()
nltk.download('stopwords')
nltk.download('punkt')

import nltk
nltk.data.path.append("nltk_data")

def data_preprocessing(text):
  text=text.lower()
  words=word_tokenize(text)
  cleaned_words=[ps.stem(word) for word in words if word.isalnum() and word not in set(stopwords.words('english'))]
  new_text= " ".join(cleaned_words)
  return new_text

tfidf=pickle.load(open('tfidf.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")
input_=st.text_area("Enter the Message")

if st.button("Predict"):
    if input_.strip()=="":
        st.warning("⚠️ Please enter a message to classify!")
    else:
        # 1 Data preprocessing
        processed_input=data_preprocessing(input_)
        # 2 vectorization
        input_vector=tfidf.transform([processed_input])
        # 3 Prediction]
        prediction=model.predict(input_vector)[0]
        # 4 Display
        if prediction==1:
         st.header("🚨 Spam Detected")
        else:
         st.header("✅ Not Spam")
        prob = model.predict_proba(input_vector)[0][1]
        st.write(f"Spam Probability: {prob * 100:.2f}%")
        with st.expander("See how text was preprocessed"):
            st.write("Processed Text: {}".format(processed_input))
            st.write("Input Vector:")
            st.write(input_vector)
            st.write("Prediction: {}".format(prediction))



