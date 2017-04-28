import numpy as np
import scipy as sp
import scipy.linalg as splinalg
import sys


class GaussianProcessMultiClassifier:
    def __init__(self, legal_labels, seed, numsamples=1000):
        """
        Do not modify this method.
        """
        self.legalLabels = legal_labels
        self.numpRng = np.random.RandomState(seed)
        self.numsamples = numsamples
        self.posteriors = []  # Log posteriors are stored for later data analysis.

    @staticmethod
    def softmax(x):
        """
        Do not modify this method.
        """
        e = np.exp(x - np.max(x))
        det = np.sum(e, axis=1)
        return (e.T / det).T

    def initialize_hyp(self):
        """
        Initialize hyper-parameter appropriately

        Do not modify this method.
        """
        self.trainingShape = np.shape(self.trainingData)
        c = len(self.legalLabels)
        noise = 20.0
        self.hypSize = [c, 2]
        self.hyp = np.zeros(self.hypSize)
        for i in range(c):
            self.hyp[i, :] = np.array([np.log(noise) / 2, np.log(noise / 4) / 2])

    def train(self, trainingData, trainingLabels):
        """
        Do not modify this method.
        """
        self.trainingData = trainingData
        self.trainingLabels = trainingLabels
        self.initialize_hyp()

        mode, _ = self.find_mode(self.trainingData, self.trainingLabels, self.hyp)
        Kcs = self.calculate_covariance(self.trainingData, self.hyp)
        t, self.tc = self.training_labels2t(self.trainingLabels)
        _, _, self.valuesForPrediction = self.calculate_intermediate_values(t, mode, Kcs)

    def cov_ard(self, hyp, x, z=None, i=None):
        """
        Squared Exponential covariance function with isotropic distance measure. The
        covariance function is parameterized as:

        k(x^p,x^q) = sf^2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)

        where the P matrix is ell^2 times the unit matrix and sf^2 is the signal
        variance. The hyperparameters are:
        hyp = [ log(ell)
                log(sf)  ]

        Do not modify this method.
        """

        def sq_dist(a, b=None):
            D, n = np.shape(a)
            if b is None:
                mu = np.mean(a, 1)
                a = (a.T - mu).T
                b = a
                m = n
            else:
                [d, m] = np.shape(b)
                if d != D:
                    print('Error: column lengths must agree.')
                    sys.exit(1)
                mu = (m / (n + m)) * np.mean(b, 1) + (n / (n + m)) * np.mean(a, 1)
                a = (a.T - mu).T
                b = (b.T - mu).T
            return np.tile(np.sum(a * a, 0), [m, 1]).T + np.tile(np.sum(b * b, 0), [n, 1]) - 2 * a.T.dot(b)

        xeqz = z is None
        ell = np.exp(hyp[0])
        sf2 = np.exp(2 * hyp[1])
        if xeqz:
            K = sq_dist(x.T / ell)
        else:
            K = sq_dist(x.T / ell, z.T / ell)
        if i is not None:
            if i == 0:
                K = sf2 * np.exp(-K / 2) * K
            elif i == 1:
                K = 2 * sf2 * np.exp(-K / 2)
            else:
                print('Unkown parameter!')
                sys.exit(1)
        else:
            K = sf2 * np.exp(-K / 2)
        return K

    def calculate_covariance(self, trainingData, hyp):
        """
        Do not modify this method.
        """
        Ks = []
        c = len(self.legalLabels)
        for i in range(c):
            Ks.append(self.cov_ard(hyp[i, :], trainingData))
        return Ks

    def training_labels2t(self, trainingLabels):
        """
        Do not modify this method.
        """
        t = []
        n = np.shape(trainingLabels)[0]
        c = len(self.legalLabels)
        for i in range(n):
            temp = np.zeros([c, 1])
            temp[trainingLabels[i]] = 1
            t.append(temp)
        ttot = np.concatenate(t)
        tc = np.reshape(ttot, [n, c])
        ttot = np.reshape(tc.T, [n * c, 1])

        return ttot, tc

    def block_diag(self, args):
        """
        Do not modify this method.
        """
        return splinalg.block_diag(*args)

    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.

        Do not modify this method.
        """
        guesses = list()

        pi, Ecs, M, R, K = self.valuesForPrediction
        tc = self.tc

        for datum in testData:
            logposterior = self.calculate_predictive_distribution(datum, pi, Ecs, M, R, tc)
            guesses.append(np.argmax(logposterior))
            self.posteriors.append(logposterior)

        return guesses

    def find_mode(self, trainingData, trainingLabels, hyp):
        n, d = self.trainingShape
        c = len(self.legalLabels)

        Kcs = self.calculate_covariance(trainingData, hyp)
        t, _ = self.training_labels2t(trainingLabels)

        """
        You should implement this method:

        Read pdf file.
        """
        ############
        t = t[:, 0]
        a = np.zeros(c * n)
        converge = 0.0
        while True:
            [_, b, logdet, K], _, _ = self.calculate_intermediate_values(t, a, Kcs)
            b = b.reshape(c*n)
            a = np.dot(K, b)

            _a = np.exp(a.reshape(c, n).T)
            log_sum = np.log(_a.sum(axis=1)).sum()
            objective = -0.5 * np.dot(b.T, a) + np.dot(t.T, a) - log_sum

            if np.allclose(objective, converge):
                break
            else:
                converge = objective
        ############
        Z = np.zeros((1, 1))
        Z[0][0] = objective - logdet
        a = a.reshape(c*n, 1)
        return a, Z  # Do not modify this line.

    def calculate_intermediate_values(self, t, a, Kcs):
        """
        You should implement this method:

        Read pdf file.
        """
        n, d = self.trainingShape
        c = len(self.legalLabels)

        ############
        t = t.reshape(c*n)
        a = a.reshape(c*n)
        _a = a.reshape(c, n)
        pi_c = np.exp(_a - np.amax(_a, axis=0))
        pi_c = pi_c / pi_c.sum(axis=0)
        pi = pi_c.reshape(c * n)
        pi_c = pi_c.T

        K = self.block_diag(Kcs)
        D = np.diag(pi)

        logdet = 0.0
        Ecs = []
        prob = []
        for cls in range(c):
            Dc = np.diag(pi_c[:, cls])
            prob.append(Dc)
            L = np.linalg.cholesky(np.identity(n) + np.dot(np.dot(np.sqrt(Dc), Kcs[cls]), np.sqrt(Dc)))
            Ec = np.dot(np.sqrt(Dc), np.linalg.solve(L.T, np.linalg.solve(L, np.sqrt(Dc))))
            Ecs.append(Ec)
            logdet += np.sum(np.log(np.diag(L)))
        M = np.linalg.cholesky(sum(Ecs))
        E = self.block_diag(Ecs)
        logdet += np.sum(np.log(np.diag(M)))

        prob = np.concatenate(prob)

        W = D - np.dot(prob, prob.T)
        R = np.dot(np.linalg.inv(D), prob)
        _c = np.dot(W, a) + t - pi
        d = np.dot(np.dot(E, K), _c)
        b = _c - d + np.dot(E, np.dot(R, np.linalg.solve(M.T, np.linalg.solve(M, np.dot(R.T, d)))))
        ############
        b = b.reshape(c*n, 1)
        pi = pi.reshape(c*n, 1)
        # Do not modify below lines.
        valuesForModes = [W, b, logdet, K]
        valuesForDerivatives = [E, M, R, b, pi, K]
        valuesForPrediction = [pi, Ecs, M, R, K]
        return valuesForModes, valuesForDerivatives, valuesForPrediction

    def calculate_predictive_distribution(self, datum, pi, Ecs, M, R, tc):
        """
        You should implement this method:

        Read pdf file.
        """
        ############
        n, d = self.trainingShape
        c = len(self.legalLabels)
        X = np.vstack((self.trainingData, datum))

        Ks = self.calculate_covariance(X, self.hyp)
        kns = []
        knns = []
        for cls in range(c):
            kns.append(Ks[cls][:, -1][:-1])
            knns.append(Ks[cls][:, -1][-1])

        Rcs = np.array(np.split(R, c, axis=0))

        pi_c = pi.reshape(c, n).T
        mu = []
        sigma = np.zeros((c, c))
        for cls in range(c):
            t_c = tc[:, cls]
            pic = pi_c[:, cls]

            mu.append(np.dot((t_c - pic).T, kns[cls]))
            f = np.dot(Ecs[cls], kns[cls])
            x1 = np.linalg.solve(M, np.dot(Rcs[cls].T, f))
            x2 = np.linalg.solve(M.T, x1)
            g = np.dot(Rcs[cls], x2)
            g = np.dot(Ecs[cls], g)

            sigma[cls, :] = np.dot(kns, g)
            sigma[cls][cls] += knns[cls] - np.dot(f.T, kns[cls])
        mu = np.array(mu)
        ############

        # Do not modify below lines.
        samples = self.numpRng.multivariate_normal(mu.ravel(), sigma, self.numsamples)
        predict = self.softmax(samples)
        return np.mean(predict, 0)


class Datum:
    """
    Do not modify this class.

    A datum is a pixel-level encoding of digits or face/non-face edge maps.
    Digits are from the MNIST dataset.
    Each digit is 28x28 pixels, each pixel can take the following values:
        0: no edge (blank)
        1: gray pixel (+) [used for digits only]
        2: edge [for face] or black pixel [for digit] (#)
    Pixel data is stored in the 2-dimensional array pixels, which
    maps to pixels on a plane according to standard euclidean axes
    with the first dimension denoting the horizontal and the second
    the vertical coordinate:

        28 # # # #            #    #
        27 # # # #            #    #
         .
         .
         .
         3 # # + #            #    #
         2 # # # #            #    #
         1 # # # #            #    #
         0 # # # #            #    #
             0 1 2 3 ... 27 28

    For example, the + in the above diagram is stored in pixels[2][3], or
    more generally pixels[column][row].
    The contents of the representation can be accessed directly
    via the getPixel and getPixels methods.
    """

    def __init__(self, data, width, height):
        """
        Create a new datum from file input (standard MNIST encoding).
        """

        def convertToInteger(data):
            def IntegerConversionFunction(character):
                if character == ' ':
                    return 0
                elif character == '+':
                    return 1
                elif character == '#':
                    return 2

            if not isinstance(data, list):
                return IntegerConversionFunction(data)
            else:
                return list(map(convertToInteger, data))

        self.height = height
        self.width = width
        if data is None:
            data = [[' ' for i in range(width)] for j in range(height)]
        self.pixels = self.arrayInvert(convertToInteger(data))

    def getPixel(self, column, row):
        """
        Returns the value of the pixel at column, row as 0, or 1.
        """
        return self.pixels[column][row]

    def getPixels(self):
        """
        Returns all pixels as a list of lists.
        """
        return self.pixels

    def __str__(self):
        def asciiGrayscaleConversionFunction(value):
            if value == 0:
                return ' '
            elif value == 1:
                return '+'
            elif value == 2:
                return '#'

        rows = []
        data = self.arrayInvert(self.pixels)
        for row in data:
            ascii = map(asciiGrayscaleConversionFunction, row)
            rows.append("".join(ascii))
        return "\n".join(rows)

    @staticmethod
    def arrayInvert(array):
        """
        inverts a matrix stored as a list of lists.
        """
        result = [[] for _ in array]
        for outer in array:
            for inner in range(len(outer)):
                result[inner].append(outer[inner])
        return result


def basic_feature_extractor(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is white (0) or gray/black (1)

    Do not modify this function.
    """
    features = dict()
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x, y)] = 1
            else:
                features[(x, y)] = 0
    return features


def readlines(filename):
    """
    Do not modify this function.
    """
    return [l[:-1] for l in open(filename).readlines()]


def load_datafile(filename, width, height):
    """
    Reads images from a file and returns a list of Datum objects.

    Do not modify this function.
    """
    fin = readlines(filename)
    fin.reverse()
    items = []
    while len(fin) > 0:
        data = []
        for j in range(height):
            data.append(list(fin.pop()))
        items.append(Datum(data, width, height))
    return items


def load_labelsfile(filename):
    """
    Reads labels from a file and returns a list of integers.

    Do not modify this function.
    """
    fin = readlines(filename)
    labels = []
    for line in fin:
        if line == '':
            break
        labels.append(int(line))
    return labels


def feature_data_to_array(basicFeatureData):
    """
    Convert basic feature data to N x d numpy array

    Do not modify this function.
    """
    basicFeatureData = list(basicFeatureData)
    N = len(basicFeatureData)
    D = len(basicFeatureData[0])
    keys = basicFeatureData[0].keys()
    sorted_keys = sorted(list(keys))

    data = np.zeros((N, D))
    for i in range(N):
        for j in range(D):
            data[i, j] = basicFeatureData[i][sorted_keys[j]]

    return data


def main():
    classifier = GaussianProcessMultiClassifier(range(10), 108)

    # Load the data
    print("Loading the data")

    trainingData = feature_data_to_array(map(basic_feature_extractor,
                                             load_datafile("./data/digit_image_train_100.csv", DIGIT_DATUM_WIDTH,
                                                           DIGIT_DATUM_HEIGHT)))
    trainingLabels = load_labelsfile("./data/digit_label_train_100.csv")
    validationData = feature_data_to_array(map(basic_feature_extractor,
                                               load_datafile("./data/digit_image_validation_100.csv", DIGIT_DATUM_WIDTH,
                                                             DIGIT_DATUM_HEIGHT)))
    validationLabels = load_labelsfile("./data/digit_label_validation_100.csv")

    # Conduct training and validating
    print("Training")
    classifier.train(trainingData, trainingLabels)
    print("Validating")
    guesses = classifier.classify(validationData)
    correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
    print(str(correct),
          ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (100.0 * correct / len(validationLabels)))


if __name__ == "__main__":
    DIGIT_DATUM_WIDTH = 28
    DIGIT_DATUM_HEIGHT = 28

    import time

    st = time.time()

    main()

    print("The main function takes {} seconds.".format(time.time() - st))
