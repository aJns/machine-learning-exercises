import numpy as np
import matplotlib.pyplot as plt

#%% 2
plt.close("all")
fig, (sp1, sp2, sp3, sp4) = plt.subplots(4)

# a
f0 = 0.1
begin = np.zeros(500)
end = np.zeros(300)
n = np.arange(100)
sinusoid = np.cos(2 * np.pi * 0.1 * n)
signal = np.concatenate([begin, sinusoid, end])

sp1.plot(np.arange(np.prod(signal.shape)), signal)

# b

noisy_signal = signal + np.sqrt(0.5) * np.random.randn(signal.size)


sp2.plot(np.arange(noisy_signal.size), noisy_signal)

# c

# h = np.exp(-2 * np.pi * 1j * f0 * n)
y = np.convolve(sinusoid, noisy_signal, 'same')
sp3.plot(np.arange(y.size), y)

h = np.exp(-2 * np.pi * 1j * f0 * n)
y = np.abs(np.convolve(h, noisy_signal, 'same'))
sp4.plot(np.arange(y.size), y)

#%% 3
plt.close("all")
fig, (sp1, sp2, sp3, sp4) = plt.subplots(4)

f0 = 0.03
begin = np.zeros(500)
end = np.zeros(300)
n = np.arange(100)
sinusoid = np.cos(2 * np.pi * f0 * n)
signal = np.concatenate([begin, sinusoid, end])

sp1.plot(np.arange(np.prod(signal.shape)), signal)

# b

noisy_signal = signal + np.sqrt(0.5) * np.random.randn(signal.size)


sp2.plot(np.arange(noisy_signal.size), noisy_signal)

# c

# h = np.exp(-2 * np.pi * 1j * f0 * n)
#y = np.convolve(sinusoid, noisy_signal, 'same')
#sp3.plot(np.arange(y.size), y)

h = np.exp(-2 * np.pi * 1j * f0 * n)
y = np.abs(np.convolve(h, noisy_signal, 'same'))
sp3.plot(np.arange(y.size), y)

#%% 4
plt.close("all")
fig = plt.figure()

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sklearn


digits = load_digits()
plt.gray()
plt.imshow(digits.images[0])
plt.show()

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.20)

#%% 5

clf = KNeighborsClassifier()
clf.fit(x_train, y_train)

predictions = clf.predict(x_test)
accuracy = sklearn.metrics.accuracy_score(y_test, predictions)
print('Accuracy is: ' + repr(accuracy))
