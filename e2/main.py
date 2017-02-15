import numpy as np
import matplotlib as plt

#%% 3

# a
csvFile = open('./locationData.csv', 'r')
ownRead = np.zeros((600, 3))

i = 0
for line in csvFile:
    splitLine = line.split()
    ownRead[i][0] = splitLine[0]
    ownRead[i][1] = splitLine[1]
    ownRead[i][2] = splitLine[2]
    i += 1

csvFile.close()

# b
npRead = np.loadtxt("./locationData.csv")

print(np.all(ownRead == npRead))

#%% 4

# a
def gaussian(x, mu, sigma):

    firstPart = 1/(np.sqrt(2*np.pi*np.power(sigma, 2)))
    secondPart = np.exp(-(1/(2*np.power(sigma, 2)) * np.power((x-mu), 2)))

    p = firstPart*secondPart
    return p

# b
def log_gaussian(x, mu, sigma):

    firstPart = np.log(1/(np.sqrt(2*np.pi*np.power(sigma, 2))))
    secondPart = -(1/(2*np.power(sigma, 2)))*np.power((x - mu), 2)

    p = firstPart + secondPart

    return p

# c
mu = 0
sigma = 1
x = np.linspace(-5,5)

gausPlot = gaussian(x, mu, sigma)
logGausPlot = log_gaussian(x, mu, sigma)

plt.figure()
plt.plot(x, gausPlot, 'r')
plt.figure()
plt.plot(x, logGausPlot, 'b')
#%% 5


# a
f0 = 0.017
w = np.sqrt(0.25) * np.random.randn(100)
n = np.arange(100)

x = np.sin(2*np.pi*f0*n) + w

plt.plot(n, x)


# b
scores = []
frequencies = []
for f in np.linspace(0, 0.5, 1000):
    e = np.zeros(np.shape(x))
    n = np.arange(100)
    z = -2*np.pi*1j*f*n
    e = np.exp(z)
    score = np.abs(np.dot(x, e))
    scores.append(score)
    frequencies.append(f)

fHat = frequencies[np.argmax(scores)]

print(fHat)
