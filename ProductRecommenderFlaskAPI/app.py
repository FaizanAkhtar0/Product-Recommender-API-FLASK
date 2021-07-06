import pandas as pd
from flask import Flask, request, jsonify
from flask_restful import Api
from ProductRecommenderFlaskAPI.models.reviews import ReviewAndRatingPreferenceModel
from flask_cors import CORS, cross_origin


recommender = ReviewAndRatingPreferenceModel()


app = Flask(__name__)

app.config['CORS_HEADERS'] = 'Content-Type'
app.config['JSON_SORT_KEYS'] = False

cors = CORS(app, resources={r"/recommender": {"origins": "http://localhost:5000"}, r"/progress": {"origins": "http://localhost:5000"}})


# api = Api(app)


@app.route('/recommender', methods=['POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def get_data():
    requested_data = request.get_json()
    identifiers = list(requested_data['identifier'])
    reviews = list(requested_data['review'])
    ratings = list(requested_data['rating'])
    ratings = [int(i) for i in ratings]
    data = {'identifier': identifiers, 'review': reviews, 'rating': ratings}
    df = pd.DataFrame(data)
    statistics = recommender.recommendation(df=df)
    return jsonify({'statistics': statistics}), 200

@app.route('/progress', methods=['GET'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def get_progress():
    return jsonify({'progress': recommender.progress_percentage * 100})



if __name__ == "__main__":
    app.run(port=5000, debug=True)


