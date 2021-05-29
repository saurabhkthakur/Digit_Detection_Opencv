from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from hog import HOG
import dataset

(digits, target) = dataset.load_digits('data/train.csv')
data = []

hog = HOG(orientations=18, pixelsPerCell= (10,10),
          cellsPerBlock=(1,1), transform=True)

for image in digits:
    image = dataset.deskew(image,20)
    image = dataset.center_extent(image, (20, 20))

    hist = hog.describe(image)
    data.append(hist)


model = LinearSVC(random_state=42)
model.fit(data, target)

joblib.dump(model, 'data/svm.cpickle')