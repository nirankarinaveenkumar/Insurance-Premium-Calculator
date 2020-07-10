import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x.lower() for x in request.form.values()]

    ##################
    '''
    Taking input values
    '''
    categorical_features = ['male', 'female', 'yes', 'no']
    categorical_number = []
    for j in categorical_features:
        try:
            categorical_number.append(int_features.index(j))
        except:
            pass

    final_list = []
    for index, new in enumerate(int_features):

        if index in categorical_number:
            if ((int_features[index].lower()) == 'male'):
                final_list.append(int(0))
                final_list.append(int(1))
            elif ((int_features[index].lower()) == 'female'):
                final_list.append(int(1))
                final_list.append(int(0))
            elif ((int_features[index].lower()) == 'yes'):
                final_list.append(int(0))
                final_list.append(int(1))
            elif ((int_features[index].lower()) == 'no'):
                final_list.append(int(1))
                final_list.append(int(0))
        else:
            final_list.append(int(new))


    final_list.append(0)
    final_list.append(0)
    final_list.append(0)
    final_list.append(1)
    ##################

    final_features = np.array(final_list).reshape(1,-1)

    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Insurance Premium is $ {}'.format(output))

#@app.route('/predict_api',methods=['POST'])
# def predict_api():
#     '''
#     For direct API calls trought request
#     '''
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])
#
#     output = prediction[0]
#     return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
