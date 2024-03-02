
# # Backend for Lung Prediction Model
# ### Importing Libraries

import joblib
import numpy as np
# Make sure that all the following modules are already installed for use.
from flask import Flask
from flask_cors import CORS
from flask_restful import Api, Resource, reqparse

# ### Creating an instance of the flask app and an API 
APP = Flask(__name__)
APP.config['CORS_HEADERS'] = 'Content-Type'
CORS(APP)
API = Api(APP)


# ### Loading the trained model
MODEL = joblib.load('titanic.pkl')


# ### Creating a class which is responsible for the prediction
class Predict(Resource):

    @staticmethod
    def post():
        parser = reqparse.RequestParser()
        parser.add_argument('Age')
        parser.add_argument('RoomService')
        parser.add_argument('FoodCourt')
        parser.add_argument('ShoppingMall')
        parser.add_argument('Spa')
        parser.add_argument('VRDeck')
        
        args = parser.parse_args()  # creates dictionary

        X_new = np.fromiter(args.values(), dtype=float)  # convert input to array

        out = {'Prediction': MODEL.predict([X_new])[1]}

        return out, 200


# ### Adding the predict class as a resource to the API
API.add_resource(Predict, '/predict')

if __name__ == '__main__':
    APP.run(debug=True)


# ### Using the request module by first defining the URL to access and the body to send along with our HTTP request

# import requests

# URL = 'http://127.0.0.1:1080/predict'  # localhost and the defined port + endpoint

# body = {
#     "Age": 25,
#     "RoomService": 0,
#     "FoodCourt": 1673,
#     "ShoppingMall": 0,
#     "Spa": 642,
#     "VRDeck": 612,
#     
# }

# response = requests.post(URL, data=body)
# response.json()


