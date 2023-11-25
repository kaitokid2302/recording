import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import joblib

# Đọc dữ liệu
df = pd.read_csv('processed_data.csv')

# Xử lý dữ liệu bị thiếu
df = df.fillna(0)

# Kiểm tra dữ liệu bị thiếu
print(df.isna().any())

# Tách dữ liệu thành features và labels
X = df.drop(labels='Emotion', axis=1)
Y = df['Emotion']

# Chuyển đổi labels thành dạng số
lb = LabelEncoder()
Y = to_categorical(lb.fit_transform(Y))

# In ra các lớp
print(lb.classes_)

# Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.2, shuffle=True)

# Chia dữ liệu train thành tập train và validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=42, test_size=0.1, shuffle=True)

# Khởi tạo và fit scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Lưu scaler vào một tệp
joblib.dump(scaler, 'scaler.pkl')

# Chuyển đổi dữ liệu test và validation
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

# Mở rộng chiều dữ liệu
X_train = np.expand_dims(X_train, axis=2)
X_val = np.expand_dims(X_val, axis=2)
X_test = np.expand_dims(X_test, axis=2)

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
print(X_train.shape, X_test.shape, X_val.shape, y_train.shape,y_test.shape,y_val.shape)
print(X_train.shape,X_test.shape,X_val.shape,y_train.shape,y_test.shape,y_val.shape)
X_train=np.expand_dims(X_train,axis=2)
X_val=np.expand_dims(X_val,axis=2)
X_test=np.expand_dims(X_test,axis=2)

print(X_train[:5])