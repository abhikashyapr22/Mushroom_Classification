import pandas as pd
import pickle
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)

df = pd.read_csv('mushrooms.csv')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    global response
    if request.method == 'POST':
        cap_shape = request.form['cap-shape']
        cap_surface = request.form['cap-surface']
        cap_color = request.form['cap-color']
        bruises = request.form['bruises']
        odor = request.form['odor']
        gill_attachment = request.form['gill-attachment']
        gill_spacing = request.form['gill-spacing']
        gill_size = request.form['gill-size']
        gill_color = request.form['gill-color']
        stalk_shape = request.form['stalk-shape']
        stalk_root = request.form['stalk-root']
        stalk_surface_above_ring = request.form['stalk-surface-above-ring']
        stalk_surface_below_ring = request.form['stalk-surface-below-ring']
        stalk_color_above_ring = request.form['stalk-color-above-ring']
        stalk_color_below_ring = request.form['stalk-color-below-ring']
        veil_type = request.form['veil-type']
        veil_color = request.form['veil-color']
        ring_number = request.form['ring-number']
        ring_type = request.form['ring-type']
        spore_print_color = request.form['spore-print-color']
        population = request.form['population']
        habitat = request.form['habitat']

        specifications = {'cap-shape': [cap_shape], 'cap-surface': [cap_surface], 'cap-color': [cap_color],
                          'bruises': [bruises], 'odor': [odor], 'gill-attachment': [gill_attachment],
                          'gill-spacing': [gill_spacing], 'gill-size': [gill_size], 'gill-color': [gill_color],
                          'stalk-shape': [stalk_shape], 'stalk-root': [stalk_root],
                          'stalk-surface-above-ring': [stalk_surface_above_ring],
                          'stalk-surface-below-ring': [stalk_surface_below_ring],
                          'stalk-color-above-ring': [stalk_color_above_ring],
                          'stalk-color-below-ring': [stalk_color_below_ring], 'veil-type': [veil_type], 'veil-color': [veil_color],
                          'ring-number': [ring_number], 'ring-type': [ring_type],
                          'spore-print-color': [spore_print_color], 'population': [population], 'habitat': [habitat]}

        try:
            with open(r"oe.pickle", "rb") as input_file:
                oe = pickle.load(input_file)
        except:
            flash("Something went wrong! Please try again")
            return redirect(url_for('home'))

        try:
            with open(r"my_model.pickle", "rb") as obj:
                my_model = pickle.load(obj)
        except:
            flash("Something went wrong! Please try again")
            return redirect(url_for('home'))

        sample = pd.DataFrame(specifications)   # or pd.DataFrame(specifications, index=[0])

        try:
            sample.drop('veil-type', axis=1, inplace=True)
            processed_sample = oe.transform(sample)
        except:
            flash("Something went wrong! Please try again")
            return redirect(url_for('home'))

        result = my_model.predict(processed_sample)
        if result[0] == 1:
            response = "Mushrooms with given specifications are poisonous"
        elif result[0] == 0:
            response = "Mushrooms with given specifications are not poisonous"
        else:
            flash("Something went wrong! Please try again")
            return redirect(url_for('home'))

        return render_template('result.html', data=response)
    else:
        return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
