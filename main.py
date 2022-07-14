# For data analysis
import pandas as pd
import numpy as np

# For graph plot
import matplotlib.pyplot as plt

# For data modelling
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

class k_means:
    def __init__(self, path='', outliers_fraction=0.01):
        self.filepath = path
        self.df = pd.read_csv(self.filepath)
        self.outliers_fraction = outliers_fraction

    def pre_processing(self):
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

        # Feature Extraction
        self.df['hours'] = self.df['timestamp'].dt.hour

        # To check whether the data was taken in daylight (7:00 - 19-00) or in night
        self.df['daylight'] = ((self.df['hours'] >= 7) & (self.df['hours'] <= 19)).astype(int)
        self.df['DayofWeek'] = self.df['timestamp'].dt.dayofweek
        self.df['WeekDay'] = (self.df['DayofWeek'] < 5).astype(int)
        self.df['time_epoch'] = (self.df['timestamp'].astype(np.int64)/100000000000).astype(np.int64)
    
    
    def standarizing(self):
        data = self.df[['value', 'hours', 'daylight', 'DayofWeek', 'WeekDay']]
        min_max_scaler = preprocessing.StandardScaler()
        np_scaled = min_max_scaler.fit_transform(data)
        data = pd.DataFrame(np_scaled)
        pca = PCA(n_components=2)
        data = pca.fit_transform(data)
        min_max_scaler = preprocessing.StandardScaler()
        np_scaled = min_max_scaler.fit_transform(data)
        data = pd.DataFrame(np_scaled)

        return data

    def getDistanceByPoint(self, data, model):
        distance = pd.DataFrame()
        for i in range(0,len(data)):
            Xa = np.array(data.loc[i])
            Xb = model.cluster_centers_[model.labels_[i]-1]
            distance.at[i, 0] = np.linalg.norm(Xa-Xb)
        return distance

    def training(self, data=[]):
        n_cluster = range(1, 28)
        K_means = [KMeans(n_clusters=i).fit(data) for i in n_cluster]
        scores = [K_means[i].score(data) for i in range(len(K_means))]

        return K_means, data

    def predict(self, K_means=[], data=[]):
        self.df['cluster'] = K_means[14].predict(data)

        distance = self.getDistanceByPoint(data, K_means[14])
        number_of_outliers = int(self.outliers_fraction*len(distance))
        threshold = distance.nlargest(number_of_outliers, 0).min()

        self.df['anomaly_kmeans'] = (distance >= threshold).astype(int)


    def visualisation(self):
        fig, ax = plt.subplots()
        a = self.df.loc[self.df['anomaly_kmeans'] == 1, ['time_epoch', 'value']]
        ax.plot(self.df['time_epoch'], self.df['value'], color='grey', alpha=0.3, label='Recorded Data')
        ax.scatter(a['time_epoch'], a['value'], color='red', label='Anomalies')
        ax.set_xlabel('Time EPOCH')
        ax.set_ylabel('Value')
        ax.legend()
        ax.set_title('Anomaly Detection (K-means Algorithm)')
        fig.savefig('./static/image/kmeans_plot.png', dpi=150, transparent=True)

class gaussian:
    def __init__(self, path = '', outliers_fraaction=0.01):
        self.filepath = path
        self.df = pd.read_csv(self.filepath)
        self.outliers_fraction = outliers_fraaction

    def pre_processing(self):
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

        # Feature Extraction
        self.df['hours'] = self.df['timestamp'].dt.hour

        # To check whether the data was taken in daylight (7:00 - 19-00) or in night
        self.df['daylight'] = ((self.df['hours'] >= 7) & (self.df['hours'] <= 19)).astype(int)
        self.df['DayofWeek'] = self.df['timestamp'].dt.dayofweek
        self.df['WeekDay'] = (self.df['DayofWeek'] < 5).astype(int)
        self.df['time_epoch'] = (self.df['timestamp'].astype(np.int64)/100000000000).astype(np.int64)

    def predict(self):
        self.df['categories'] = self.df['WeekDay'] * 2 + self.df['daylight']
        # Creatting 4 different dataset based on categories
        df_class_0 = self.df.loc[self.df['categories'] == 0, 'value']
        df_class_1 = self.df.loc[self.df['categories'] == 1, 'value']
        df_class_2 = self.df.loc[self.df['categories'] == 2, 'value']
        df_class_3 = self.df.loc[self.df['categories'] == 3, 'value']

        envelope =  EllipticEnvelope(contamination=self.outliers_fraction) 
        X_train = df_class_0.values.reshape(-1,1)
        envelope.fit(X_train)
        df_class_0 = pd.DataFrame(df_class_0)
        df_class_0['deviation'] = envelope.decision_function(X_train)
        df_class_0['anomaly'] = envelope.predict(X_train)

        envelope = EllipticEnvelope(contamination=self.outliers_fraction)
        X_train = df_class_1.values.reshape(-1, 1)
        envelope.fit(X_train)
        df_class_1 = pd.DataFrame(df_class_1)
        df_class_1['deviation'] = envelope.decision_function(X_train)
        df_class_1['anomaly'] = envelope.predict(X_train)

        envelope =  EllipticEnvelope(contamination = self.outliers_fraction) 
        X_train = df_class_2.values.reshape(-1,1)
        envelope.fit(X_train)
        df_class_2 = pd.DataFrame(df_class_2)
        df_class_2['deviation'] = envelope.decision_function(X_train)
        df_class_2['anomaly'] = envelope.predict(X_train)

        envelope =  EllipticEnvelope(contamination = self.outliers_fraction) 
        X_train = df_class_3.values.reshape(-1,1)
        envelope.fit(X_train)
        df_class_3 = pd.DataFrame(df_class_3)
        df_class_3['deviation'] = envelope.decision_function(X_train)
        df_class_3['anomaly'] = envelope.predict(X_train)

        df_class = pd.concat([df_class_0, df_class_1, df_class_2, df_class_3])
        self.df['anomaly_gaussian'] = df_class['anomaly']
        self.df['anomaly_gaussian'] = np.array(self.df['anomaly_gaussian'] == -1).astype(int)

    def visualisation(self):
        fig, ax = plt.subplots()
        a = self.df.loc[self.df['anomaly_gaussian'] == 1, ('time_epoch', 'value')] # Anomaly

        ax.set_xlabel('Time EPOCH')
        ax.set_ylabel('Value')
        ax.plot(self.df['time_epoch'], self.df['value'], color='grey', alpha=0.3, label='Recorded Data')
        ax.scatter(a['time_epoch'], a['value'], color='red', label='Anomalies')
        ax.set_title('Anomaly Detection (Gaussian Mixture')
        ax.legend()
        fig.savefig('./static/image/gaussian.png', dpi=150, transparent=True)

class isolationForest:
    def __init__(self, path='', outliers_fraction=0.01):
        self.filepath = path
        self.df = pd.read_csv(self.filepath)
        self.outliers_fraction = outliers_fraction

    def pre_processing(self):
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

        # Feature Extraction
        self.df['hours'] = self.df['timestamp'].dt.hour

        # To check whether the data was taken in daylight (7:00 - 19-00) or in night
        self.df['daylight'] = ((self.df['hours'] >= 7) & (self.df['hours'] <= 19)).astype(int)
        self.df['DayofWeek'] = self.df['timestamp'].dt.dayofweek
        self.df['WeekDay'] = (self.df['DayofWeek'] < 5).astype(int)
        self.df['time_epoch'] = (self.df['timestamp'].astype(np.int64)/100000000000).astype(np.int64)

    def standarizing(self):
        data = self.df[['value', 'hours', 'daylight', 'DayofWeek', 'WeekDay']]
        min_max_scaler = preprocessing.StandardScaler()
        np_scaled = min_max_scaler.fit_transform(data)
        data = pd.DataFrame(np_scaled)

        return data

    def training(self, data=[]):
        iso_model = IsolationForest(contamination=self.outliers_fraction)
        iso_model.fit(data) 

        return iso_model, data

    def predict(self, iso_model=[], data=[]):
        self.df['anomaly_isolationForest']  = pd.Series(iso_model.predict(data))
        self.df['anomaly_isolationForest'] = (self.df['anomaly_isolationForest'] == -1).astype(int)

    def visualisation(self):
        fig, ax = plt.subplots()

        a = self.df.loc[self.df['anomaly_isolationForest'] == 1, ['time_epoch', 'value']]
        ax.plot(self.df['time_epoch'], self.df['value'], color='grey', alpha=0.3, label='Recorded Data')
        ax.scatter(a['time_epoch'], a['value'], color='red', label='Anomalies')
        ax.set_xlabel('Time EPOCH')
        ax.set_ylabel('Value')
        ax.set_title('Anomaly Detection (Isolation forest)')
        ax.legend()
        fig.savefig('./static/image/isoForest.png', dpi=150, transparent = True)
class unsup_knn:
    def __init__(self, path='', threshold=0.8):
        self.filepath = path
        self.df = pd.read_csv(self.filepath)
        self.threshold = threshold

    def pre_processing(self):
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

        # Feature Extraction
        self.df['hours'] = self.df['timestamp'].dt.hour

        # To check whether the data was taken in daylight (7:00 - 19-00) or in night
        self.df['daylight'] = ((self.df['hours'] >= 7) & (self.df['hours'] <= 19)).astype(int)
        self.df['DayofWeek'] = self.df['timestamp'].dt.dayofweek
        self.df['WeekDay'] = (self.df['DayofWeek'] < 5).astype(int)
        self.df['time_epoch'] = (self.df['timestamp'].astype(np.int64)/100000000000).astype(np.int64)

    def standarizing(self):
        data = self.df[['value', 'hours', 'daylight', 'DayofWeek', 'WeekDay']]
        min_max_scaler = preprocessing.StandardScaler()
        np_scaled = min_max_scaler.fit_transform(data)
        data = pd.DataFrame(np_scaled)

        return data

    def training(self, data=[]):
        knn = NearestNeighbors(n_neighbors=403, algorithm='auto')
        knn.fit(data)

        return knn, data

    def predict(self, knn=[], data=[]):
        distance, indecis = knn.kneighbors(data)
        distance = pd.DataFrame(distance)
        distance_mean = distance.mean(axis=1)
        outlier_index = np.where(distance_mean > self.threshold)
        self.df['anomaly_unsupKNN'] = (distance_mean > self.threshold).astype(int)
        outlier_index = np.where(distance_mean > self.threshold)
        self.df['anomaly_unsupKNN'] = (distance_mean > self.threshold).astype(int)

        return outlier_index

    def visualisation(self, outlier_index=[]):
        outlier_values = self.df.iloc[outlier_index]
        fig, ax = plt.subplots()
        ax.plot(self.df["time_epoch"], self.df["value"], color="grey", alpha=0.3, label='Recorded Data')
        # plot outlier values
        ax.scatter(outlier_values["time_epoch"], outlier_values["value"], color = "red", label='Anomalies')
        ax.set_xlabel('Time EPOCH')
        ax.set_ylabel('Value')
        ax.set_title('Anomaly Detection (Unsupervised KNN)')
        ax.legend()
        fig.savefig('./static/image/unsupKNN.png', dpi=150, transparent=True)
