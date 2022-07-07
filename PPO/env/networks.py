import tensorflow.keras as keras
from tensorflow.keras.layers import Dense


class ActorNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=512, fc2_dims=256, cont=False):
        super(ActorNetwork, self).__init__()

        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        if cont:
            self.fc3 = Dense(4, activation='tanh')
        else:
            self.fc3_1 = Dense(n_actions, activation='softmax')
            self.fc3_2 = Dense(n_actions, activation='softmax')

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x1 = self.fc3_1(x)
        x2 = self.fc3_2(x)

        return x1, x2


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=256):
        super(CriticNetwork, self).__init__()
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        q = self.q(x)

        return q
