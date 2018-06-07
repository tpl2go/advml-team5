import keras
import keras.backend as K
from keras import initializers
from keras.engine import InputSpec
from keras.layers import Input, Dense, Lambda, Wrapper
from keras.models import Model
import numpy as np

from concretedropout import *
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

num_classes = 10
# preprocess data
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


nb_epochs = 500
nb_features = 512
# 500 epochs -> 2*1e5 iterations --> 400 iterations per epoch
# 60000 points into 400 iterations means batch size is 150

def fit_model_mnist(x_train, y_train, nb_epoch, batch_size, hidden_dim, l):
    if K.backend() == 'tensorflow':
        K.clear_session()
    N = x_train.shape[0] # number of input points
    dim = x_train.shape[1]
    wd = l**2. / N 
    dd = 2. / N
    inp = Input(shape=(dim,))
    x = inp
    x = ConcreteDropout(Dense(hidden_dim, activation='relu'), weight_regularizer=wd, dropout_regularizer=dd, name='relu_1')(x)
    x = ConcreteDropout(Dense(hidden_dim, activation='relu'), weight_regularizer=wd, dropout_regularizer=dd, name='relu_2')(x)
    x = ConcreteDropout(Dense(hidden_dim, activation='relu'), weight_regularizer=wd, dropout_regularizer=dd, name='relu_3')(x)
    out = ConcreteDropout(Dense(num_classes, activation='softmax'), weight_regularizer=wd, dropout_regularizer=dd, name='softmax_output')(x)
    model = Model(inp, out)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    assert len(model.layers[1].trainable_weights) == 3  # kernel, bias, and dropout prob
    assert len(model.losses) == 4  # a loss for each Concrete Dropout layer
    hist = model.fit(x_train, y_train, epochs=nb_epoch, batch_size=batch_size, verbose=0)
    loss = hist.history['loss'][-1]
    return model, loss, hist  # return ELBO up to const.


results = []
data_fraction = [0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1]
nb_epochs = 500  # 500
nb_reps = 3
K_test = 20 # 20
l = 1e-4 # NOTE: why is this specified to be the l parameter?

# get results for multiple N
for f in data_fraction:
    # repeat exp multiple times
    rep_results = []
    for i in range(nb_reps):
        # generate the data points, take the first N for train and the remaining 1000 for validation
        N = x_train.shape[0]
        x = x_train[:int(f*N), :]
        y = y_train[:int(f*N), :]
        print(x.shape)
        print("frac: {} iter: {}".format(f, i))
        # get model and loss trained for nb_epochs
        model, loss, hist = fit_model_mnist(x, y, nb_epoch=nb_epochs,
                                            batch_size=150, hidden_dim=512,
                                            l=l)
        accuracies = np.array([model.evaluate(x_test, y_test) for _ in range(K_test)])
        # MC sample due to sampling from concrete dist, Ktest x Ntest x 10
        MC_samples = np.array([model.predict(x_test) for _ in range(K_test)])
        # epistemic uncertainty = variance of the Ktest MC samples, avg over N
        # data points
        epistemic_uncertainty = np.mean(np.var(MC_samples, 0), axis=0)
        ps = np.array([K.eval(layer.p) for layer in model.layers if hasattr(layer, 'p')])
        # repeat 3 times
        rep_results += [(accuracies, ps, epistemic_uncertainty)]
    results += [rep_results]



import pickle
with open('mnist_results.pkl', 'wb') as f:
    pickle.dump(results, f)

