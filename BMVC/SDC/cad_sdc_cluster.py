from keras.models import load_model
import numpy as np
from numpy.linalg import norm
from utils import get_group_instance
from scipy import io

model_path = './models/cad-sdc-2fc3.h5'
group_net = load_model(model_path)
group_data = io.loadmat('../group_data.mat')['trainX'][:, 0]

for k in range(5):
    datum = get_group_instance(group_data)
    features = datum['features']
    labels = datum['labels']
    phi = group_net.predict(features)
    print(labels)
    N = labels.shape[0]
    pair_dist = np.zeros(shape=(N, N))
    for i in range(N):
        for j in range(N):
            pair_dist[i, j] = norm(phi[i, :] - phi[j, :])
    print(pair_dist)
