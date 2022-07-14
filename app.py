from flask import Flask, render_template, request
import main

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', data_req='/static/image/data_req.png')
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    datafile = request.files['datafile']
    data_path = './static/'+datafile.filename
    datafile.save(data_path)

    kmeans = main.k_means(path=data_path, outliers_fraction=0.01)
    kmeans.pre_processing()
    data = kmeans.standarizing()
    model, data = kmeans.training(data=data)
    kmeans.predict(K_means=model, data=data)
    kmeans.visualisation()

    gaussian_mix = main.gaussian(path=data_path, outliers_fraaction=0.01)
    gaussian_mix.pre_processing()
    gaussian_mix.predict()
    gaussian_mix.visualisation()

    isoForest = main.isolationForest(path=data_path, outliers_fraction=0.01)
    isoForest.pre_processing()
    data = isoForest.standarizing()
    model, data = isoForest.training(data=data)
    isoForest.predict(iso_model=model, data=data)
    isoForest.visualisation()

    knn = main.unsup_knn(path=data_path, threshold=0.8)
    knn.pre_processing()
    data = knn.standarizing()
    model, data = knn.training(data=data)
    index = knn.predict(knn=model, data=data)
    knn.visualisation(outlier_index=index)

    img_kmean = '/static/image/kmeans_plot.png'
    img_gauss = '/static/image/gaussian.png'
    img_iso = '/static/image/isoForest.png'
    img_kn = '/static/image/unsupKNN.png'

    return render_template('results.html', flag=True)

if __name__  == '__main__':
    app.run(port=3000, debug=True)