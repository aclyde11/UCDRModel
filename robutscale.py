import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

data = np.load("cleaned.npz")
X_train, X_test = data['x_train'].astype(np.int32), data['x_test'].astype(np.int32)
y_train, y_test = data['y_train'], data['y_test']

rb = RobustScaler()
y_train_fit = rb.fit_transform(y_train.reshape((-1,1))).reshape(-1).astype(np.float32)
y_test_fit = rb.transform(y_test.reshape((-1,1))).reshape(-1).astype(np.float32)

np.savez_compressed("cleaned_scaled.npz", x_test=X_test, x_train=X_train, y_test=y_test, y_train=y_train)
