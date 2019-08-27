
#for mail Extraction online
import pandas as pd
import pickle
from flask import Flask, render_template, request
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
app = Flask(__name__)

@app.route('/')
def student():
   return render_template('home.html')

@app.route('/result',methods = ['POST', 'GET'])

def fun1():
   if request.method == 'POST':

      p1= request.form["p1"]
      p2 = request.form["p2"]
      test = request.form["p3"]
      data = pd.read_csv("spam.csv")
      final_result = []

      for i in range(0, 5572):
         review = re.sub('[^a-zA-Z]', ' ', data['EmailText'][i])
         review = review.lower()
         review = review.split()
         ps = PorterStemmer()
         review = [ps.stem(word) for word in review if not word in stopwords.words("english")]
         review = " ".join(review)
         final_result.append(review)

      print(final_result)
      from sklearn.feature_extraction.text import CountVectorizer
      cv = CountVectorizer()
      x = cv.fit_transform(final_result).toarray()

      y = data.iloc[:, 0].values
      y = y.reshape((5572, 1))

      from sklearn.preprocessing import LabelEncoder
      l = LabelEncoder()
      y[:, 0] = l.fit_transform(y[:, 0])
      y = y.astype('int')


      review = re.sub('[^a-zA-Z]', ' ', test)
      review = review.lower()
      review = review.split()
      review = [ps.stem(word) for word in review if not word in stopwords.words("english")]
      review = " ".join(review)
      print([review])
      test = cv.transform([review]).toarray()

      from sklearn.ensemble import BaggingClassifier
      model = BaggingClassifier()
      model.fit(x, y)

      pred = model.predict(test)
      a=pred[0]

      if(a==0):
         d="Not Spam"
      elif(a==1):
         d="Spam"
      else:
          d="null"

      return render_template("result.html",result = d)


if __name__ == '__main__':
   app.run(host="127.0.0.1",port=8080,debug=True)