import numpy as np

class LogisticRegressionGD:
    """Gradient descent-based logistic regression classifier.
    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
    initialization.
    
    Attributes
    -----------
    w_ : 1d-array
      Weights after training.
    b_ : Scalar
      Bias unit after fitting.
    losses_ : list
      Mean squared error loss function values in each epoch.
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the
          number of examples and n_features is the
          number of features.
        y : array-like, shape = [n_examples]
          Target values.
        
        Returns
        -------
        self : Instance of LogisticRegressionGD
        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))) / X.shape[0])
            self.losses_.append(loss)
        return self
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_
    
    def activation(self, z):
        """Compute logistic sigmoid activation"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
  

class SoftmaxRegressionGD:
    """Gradient descent-based multinomial logistic regression (softmax classifier).
    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
    initialization.
    """

    def __init__(self, eta=0.01, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def _softmax(self, z):
        """ softmax function
        Parameters
        ----------
        z: shape [n_samples, n_classes]
        """

        # substract the maximum, avoid overflow
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)

        # sum across each line
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _cross_entropy_loss(self, y_true, y_prob):
        # cross entroy loss function
        # see DLFC chapter5 p161 (5.79)

        # avoid log(0)
        # confine the y_prob to the interval (1e-10, 1)
        y_prob = np.clip(y_prob, 1e-10, 1.0)
        return -np.mean(np.sum(y_true * np.log(y_prob), axis=1))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # One-hot encode y
        # y_encoded becomes like
        # [[0, ..., 1, ..., 0],
        #  [0, ..., 1, ..., 0],
        #  ... ]
        y_encoded = np.eye(n_classes)[y]

        # initialization
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=(n_classes, n_features))
        self.b_ = np.zeros(n_classes)
        self.losses_ = []

        for _ in range(self.n_iter):
            z = X.dot(self.w_.T) + self.b_  # shape: (n_samples, n_classes)
            y_prob = self._softmax(z)       # shape: (n_samples, n_classes)

            # compute the gradient
            errors = y_prob - y_encoded     # shape: (n_samples, n_classes)

            # DLFC chapter5 p162 (5.81)
            grad_w = (errors.T.dot(X)) / n_samples
            grad_b = errors.mean(axis=0)

            # update weight
            self.w_ -= self.eta * grad_w
            self.b_ -= self.eta * grad_b

            # compute loss
            loss = self._cross_entropy_loss(y_encoded, y_prob)
            self.losses_.append(loss)

        return self

    def predict(self, X):
        """ predict which class X belongs to
        """
        z = X.dot(self.w_.T) + self.b_
        y_prob = self._softmax(z)
        return np.argmax(y_prob, axis=1)

    def predict_proba(self, X):
        """ compute the probability of each class
        """
        z = X.dot(self.w_.T) + self.b_
        return self._softmax(z)