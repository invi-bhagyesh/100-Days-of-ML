import numpy as np

class NN:
    def __init__(self, in_size, hid_size, out_size, alpha=0.01):
        self.alpha = alpha
        self.w1 = np.random.randn(in_size, hid_size)*0.01
        self.w2 = np.random.randn(hid_size, out_size)*0.01
        self.b1 = np.zeros((1, hid_size))
        self.b2 = np.zeros((1, out_size))
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_diff(self, x):
        return x*(1-x)
    
    def forward(self, x):
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backprop(self, x, y, y_hat):
        error = y - y_hat
        diff_out = error * self.sigmoid_diff(y_hat)
        diff_in = np.dot(diff_out, self.w2.T) * self.sigmoid_diff(self.a1)
        self.w2 += np.dot(self.a1.T, diff_out)* self.alpha
        self.b2 += np.sum(diff_out, axis=0, keepdims=True)*self.alpha
        self.w1 += np.dot(x.T, diff_in)* self.alpha
        self.b1 += np.sum(diff_in, axis=0, keepdims=True)*self.alpha
    
    def train(self, x, y, epochs=1000):
        for epoch in range(epochs):
            y_hat = self.forward(x)
            self.backprop(x, y, y_hat)
            if epoch%100==0:
                loss = np.mean(np.square(y-y_hat))
                print(f"Epoch: {epoch}, Loss: {loss:.4f}")
    
    def predict(self, x):
        return self.forward(x)
