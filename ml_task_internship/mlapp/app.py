from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction', methods=['POST'])
def pred():
    review = request.form.get('review')
    data_point = str(review)

    model = joblib.load('model/sentiment_model_svc.pkl')

    prediction = model.predict([data_point])[0]
    
    if prediction == 'negative':
        sentiment = 'negative'
        sentiment_emoji = '😞'
    else:
        sentiment = 'positive'
        sentiment_emoji = '😊'
    
    return render_template('output.html', prediction=prediction,review=review,sentiment_emoji=sentiment_emoji)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
