#import cv2
import numpy as np
from skimage.feature import hog


train = np.loadtxt('train.csv', delimiter=',', skiprows=1)
test = np.loadtxt('test.csv', delimiter=',', skiprows=1)

# сохраняем разметку в отдельную переменную
train_label = train[:, 0]
# приводим размерность к удобному для обаботки виду
train_img = np.resize(train[:, 1:], (train.shape[0], 28, 28))
test_img = np.resize(test, (test.shape[0], 28, 28))

print(train_img.shape)

def extract_hog_features(images):
    features = []
    for img in images:
        # Преобразование изображения в градиенты HOG
        hog_features = hog(img, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
        features.append(hog_features)
    return np.array(features)

# Извлечение признаков HOG
train_features = extract_hog_features(train_img)
test_features = extract_hog_features(test_img)

#RandomForest
from sklearn.ensemble import RandomForestClassifier

# Создание и обучение модели
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(train_features, train_label)

# Прогнозирование на тестовом наборе
lable_pred = model.predict(test_features)



# Для демонстрации оценим модель на части обучающего набора
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train_features, train_label, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
val_predictions = model.predict(X_val)

#Оцениваем качество решение на валидационной выборке
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
print("Accuracy on validation set:", accuracy_score(y_val, val_predictions))
print("Classification report:\n", classification_report(y_val, val_predictions))

#выгружаем данные в файл
with open('PCA_submit_int.csv', 'w') as dst:
    dst.write('ImageId,Label\n')
    for i, p in enumerate(lable_pred, 1):
        dst.write('%s,%s\n' % (i, int(p)))