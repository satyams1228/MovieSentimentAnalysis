from flask import Flask, request,render_template
from flask_cors import CORS
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from flask import jsonify



app = Flask(__name__)
CORS(app)

@app.route('/index',methods=['GET'])
def show():
    return render_template("index.html")
 

@app.route('/predict', methods=['POST'])
def predict():
    # Load the model and vectorizer from the pickle files
    with open("movie_reviews_model.pkl", "rb") as file:
        model = pickle.load(file)   
    with open("movie_reviews_vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)
    
    # Get the review from the POST request
    review = request.form['review']
    
    # Transform the review using the loaded vectorizer
    review_features = vectorizer.transform([review])
    
    # Predict the sentiment using the loaded model
    sentiment = model.predict(review_features)[0]
    
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run()