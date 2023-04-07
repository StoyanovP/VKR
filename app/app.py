import flask
import numpy as np
from flask import render_template
import pickle
import sklearn
from sklearn.ensemble import AdaBoostRegressor

app = flask.Flask(__name__, template_folder = 'templates')

@app.route('/', methods = ['POST', 'GET'])
@app.route('/index', methods = ['POST', 'GET'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html')

    if flask.request.method == 'POST':
        with open('model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)

        ratio = float(flask.request.form['ratio'])
        dens = float(flask.request.form['density'])
        elastic_module = float(flask.request.form['elastic_module'])
        amount = float(flask.request.form['amount'])
        epoxy_gr = float(flask.request.form['epoxy_gr'])
        flash_point = float(flask.request.form['flash_point'])
        surface_density = float(flask.request.form['surface_density'])
        resin_con = float(flask.request.form['resin_con'])
        patch_angle = float(flask.request.form['patch_angle'])
        patch_step = float(flask.request.form['patch_step'])
        patch_dens = float(flask.request.form['patch_dens'])

        feature_list = [ratio, dens, elastic_module, amount, epoxy_gr, flash_point, 
        surface_density, resin_con, patch_angle, patch_step, patch_dens]
        X = np.array(feature_list).reshape(1,-1)
        y_pred = loaded_model.predict(X)

        return render_template('main.html', result = y_pred)


if __name__ == '__main__':
    app.run()