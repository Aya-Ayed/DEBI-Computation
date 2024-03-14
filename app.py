
# # # Backend for Titanic Prediction Model
# # ### Importing Libraries

# import joblib
# import numpy as np
# # Make sure that all the following modules are already installed for use.
# from flask import Flask
# from flask_cors import CORS
# from flask_restful import Api, Resource, reqparse

# # ### Creating an instance of the flask app and an API 
# APP = Flask(__name__)
# APP.config['CORS_HEADERS'] = 'Content-Type'
# CORS(APP)
# API = Api(APP)


# # ### Loading the trained model
# MODEL = joblib.load('titanic.pkl')


# # ### Creating a class which is responsible for the prediction
# class Predict(Resource):

#     @staticmethod
#     def post():
#         parser = reqparse.RequestParser()
#         parser.add_argument('Age')
#         parser.add_argument('RoomService')
#         parser.add_argument('FoodCourt')
#         parser.add_argument('ShoppingMall')
#         parser.add_argument('Spa')
#         parser.add_argument('VRDeck')
        
#         args = parser.parse_args()  # creates dictionary

#         X_new = np.fromiter(args.values(), dtype=float)  # convert input to array

#         out = {'Prediction': MODEL.predict([X_new])[1]}

#         return out, 200


# # ### Adding the predict class as a resource to the API
# API.add_resource(Predict, '/predict')

# if __name__ == '__main__':
#     APP.run(debug=True)
    
# # ### Using the request module by first defining the URL to access and the body to send along with our HTTP request

# # import requests

# # URL = 'http://127.0.0.1:1080/predict'  # localhost and the defined port + endpoint

# # body = {
# #     "Age": 25,
# #     "RoomService": 0,
# #     "FoodCourt": 1673,
# #     "ShoppingMall": 0,
# #     "Spa": 642,
# #     "VRDeck": 612,
# #     
# # }

# # response = requests.post(URL, data=body)
# # response.json()





# from flask import Flask
# from flask import jsonify
# import joblib

# app = Flask(__name__)

# MODEL = joblib.load('titanic.pkl')

# @app.route("/")
# def hello():
#     """Return a friendly HTTP greeting."""
#     print("I am inside hello world")
#     return "Continuous Delivery Demo"


# @app.route("/echo/<name>")
# def echo(name):
#     print(f"This was placed in the url: new-{name}")
#     val = {"new-name": name}
#     return jsonify(val)


# if __name__ == "__main__":
#     # Setting debug to True enables debug output. This line should be
#     # removed before deploying a production app.
#     app.debug = True
#     app.run(host="0.0.0.0", port=8080)
    
    
    



from flask import Flask,render_template,request
import joblib
import numpy as np

app=Flask(__name__)

model=joblib.load('titanic.pkl')

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET'])
def predict():
    all_data=request.args
    age=int(all_data['age'])
    roomservice=int(all_data['roomservice'])
    foodcourt=int(all_data['foodcourt'])
    shoppingmall=int(all_data['shoppingmall'])
    spa=int(all_data['spa'])
    VRDeck=int(all_data['VRDeck'])
    

    data=[age,roomservice,foodcourt,shoppingmall,spa,VRDeck]
    data = np.array(data).reshape(1, -1)
    pred=model.predict(data)[0]
    
    return render_template('prediction.html',Risk_Flag=pred)


if __name__=='__main__':
    app.debug = True
    app.run(host="0.0.0.0", port=8080)