import ipdb
import numpy
import pylab
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.image as img
import scipy.ndimage.morphology as morph 
import scipy.io as io

pylab.ion()

locationFigure = plt.figure()

# 1
locationData = numpy.loadtxt("./locationData.csv")

print(numpy.shape(locationData))

#2
col1 = locationData[:,0]
col2 = locationData[:,1]
col3 = locationData[:,2]


locationPlot = plt.plot(col1, col2)

ax = plt.subplot(1, 1, 1, projection = "3d")
locationPlot = plt.plot(col1, col2, col3)
plt.close("all")

#3
imageFigure = plt.figure()
image = img.imread("./oulu.jpg")
print("Image shape: " + str(numpy.shape(image)))
print("Whole image mean: " + str(numpy.mean(image)))

print("Red mean: " + str(numpy.mean(image[:,:,0])))
print("Green mean: " + str(numpy.mean(image[:,:,1])))
print("Blue mean: " + str(numpy.mean(image[:,:,2])))

morphedImage = morph.white_tophat(image, 10)

plt.subplot(211)
imagePlot = plt.imshow(image)
plt.subplot(212)
imagePlot = plt.imshow(morphedImage)

plt.close(imageFigure)

#4
fig = plt.figure()
mat = io.loadmat("./twoClassData.mat")
X = mat["X"]
y = mat["y"].ravel()

plt.plot(X[y == 0, 0], X[y == 0, 1], 'ro')
plt.plot(X[y == 1, 0], X[y == 1, 1], 'bo')

plt.close(fig)

#5
def normalize_data(X):
    arrayShape = numpy.shape(X)
    rows = arrayShape[0]
    cols = arrayShape[1]
    X_norm = numpy.empty(arrayShape)

    for col in range(cols):
        column = X[:, col]
        xMean = numpy.mean(column)
        xStd = numpy.std(column)

        # ipdb.set_trace()

        for row in range(rows):
            X_norm[row, col] = (column[row] - xMean)/xStd

    return X_norm

X_norm = normalize_data(X)
printMean = numpy.mean(X_norm, axis = 0) # Should be 0
printStd = numpy.std(X_norm, axis = 0) # Should be 1

print("The mean (should be 0): " + str(printMean))
print("The std (should be 1): " + str(printStd))











