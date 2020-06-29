import numpy as np
from scipy.stats import norm, zscore
from pyts.approximation import SymbolicAggregateApproximation, PiecewiseAggregateApproximation
from typing import List, Dict
from collections import defaultdict
import string


class HotSax:
    def __init__(self, window_size: int, alphabet_size: int = 0, mode: str = 'brute',
                 multiple_discords: bool = False, nb_discords: int = 1):
        # Todo: differentiate between window_size and paa segment size
        self.window_size = window_size
        self.nb_bins = alphabet_size
        self.mode = mode.lower()
        self.data = None
        self._cutoffs = self._calculate_sax_cutoff()
        self._dist_matrix = self._create_distance_matrix()
        self._mapping = self._alphabet_mapping()
        self._length = None
        self._split = None
        self._segments = None
        self._norm_data = None
        self.paa_data = None
        self.sax_data = None
        self.sax_word_list = None
        self._methods = {'brute': 'Brute force method', 'hot': 'Heuristics based method'}
        self._best_dist = 0
        self.best_loc = np.nan
        self.multiple_discords = multiple_discords
        self.nb_discords = nb_discords
        self.discords_location = defaultdict()
        if self.multiple_discords:
            self.all_discords = defaultdict()
        else:
            self.all_discords = None

        if self.mode not in self._methods.keys():
            raise ValueError(f"Incorrect argument: '{mode}', only 'brute' and 'hot' can be used.")

    def _calculate_sax_cutoff(self) -> List:
        """
        Calculate the cutoffs on the Normal distribution used by the SAX algorithm. For example, from 3
        bins there are 2 cutoff points that correspond to the Normal CDF being lower than 1/3 for
        the first cutoff and lower than 2/3 for the second cutoff. This is done using a normalized
        distribution (i.e. mean of 0 and variance of 1).

        :return: a list object containing the different cutoffs
        """
        bins_prop = [i / self.nb_bins for i in range(1, self.nb_bins)]
        cutoff = norm.ppf(bins_prop)
        return cutoff

    def _create_distance_matrix(self):
        """
        Create a matrix that will be used to measure the distance between SAX digits. The matrix is
        symmetrical.
        :return: the matrix of distances
        """
        cutoffs = self._calculate_sax_cutoff()
        dist_matrix = np.zeros((self.nb_bins, self.nb_bins))
        for i in range(self.nb_bins):
            for j in range(i + 2, self.nb_bins):
                dist_matrix[i, j] = (cutoffs[i] - cutoffs[j - 1]) ** 2
                dist_matrix[j, i] = dist_matrix[i, j]

        return dist_matrix

    def _alphabet_mapping(self) -> Dict:
        """
        Map a SAX alphabet to a dict containing each letter with an index
        :return: the mapping as a dictionary
        """
        mapping = defaultdict()
        letters = list(string.ascii_lowercase)[:self.nb_bins]
        for i, l in enumerate(letters):
            mapping[l] = i

        return mapping

    def _calculate_distance_between_digit(self, digit1: str, digit2: str) -> float:
        """
        Calculates the distance between two digits from the SAX alphabet
        :param digit1: a digit from the SAX alphabet
        :param digit2: a digit from the SAX alphabet
        :return: distance between digits as a float
        """
        i, j = self._mapping[digit1], self._mapping[digit2]
        distance = self._dist_matrix[i, j]

        return distance

    def _calculate_distance(self, input1: str, input2: str):
        """
        Calculate the distance between two SAX same length words from the same alphabet.
        :param input1: a string consisting of letters from the SAX alphabet
        :param input2: a string consisting of letters from the SAX alphabet
        :return: distance between the two strings as a float
        """
        if isinstance(input1, str) and isinstance(input2, str):
            if len(input1) != len(input2):
                raise InputError(f"Discrepancies between length of {input1} which is {(len(input1))} "
                                 f"and length of {input2} which is {(len(input2))}")

            length = len(input1)
            distance = 0
            for i in range(length):
                distance += self._calculate_distance_between_digit(input1[i], input2[i])

            return np.sqrt((self.window_size / self._length) * distance)
        else:
            raise InputError("Can only calculate distance between strings.")

    def _euclidean_distance(self, input1, input2):
        """
        Calculate the Euclidean distance between two numpy arrays
        :param input1: a numpy array
        :param input2: a numpy array
        :return: the Euclidean distance between the two input arrays multiplied by the square root of the ratio of
        the window length to the length of the time series.
        """
        if isinstance(input1, np.ndarray) and isinstance(input2, np.ndarray):
            if input1.shape != input2.shape:
                raise InputError(
                    f"Mismatch in the Euclidean distance calculation: first input has shape {input1.shape} "
                    f"and second input has shape {input2.shape}")

            distance = np.linalg.norm(input1 - input2, ord=2)
            distance *= np.sqrt(self.window_size / self._length)
            return distance
        else:
            raise InputError("Can only calculate distance between numpy arrays.")

    def _load_data(self, data):
        """
        Save data into the class and ensure the data provided has the following shape: (1,n)
        where n is the length of the time series.

        Parameters
        ----------
        data: numpy array containing the dataset

        Returns
        -------
        Nothing
        """

        if not isinstance(data, np.ndarray):
            raise InputError('Data must in a numpy array')

        self.data = data
        self.data.shape = (1, -1)

    def _normalize_data(self):
        """
        Normalize the input data
        :return: nothing, normalized data is stored internally to the class instance.
        """
        self._norm_data = zscore(self.data, axis=1)

    def _paa(self):
        # Todo: rework docstring
        """
        Takes a Numpy array (ndarray) and apply the Piecewise Aggregate Approximation
        algorithm (PAA) on it.
        This is a wrapper around the PiecewiseAggregateApproximation() class from the pyts
        package.
        :return: nothing, all objects are stored internally in the class.
        """
        paa = PiecewiseAggregateApproximation(window_size=self.window_size)

        self.paa_data = paa.fit_transform(self.data)

    def _sax(self):
        # Todo: rework docstring
        """
        Computes the Symbolic Aggregate Approximation of a time series using the 'normal' strategy.
        This is a wrapper around the SymbolicAggregateApproximation() class from the pyts
        package.
        :return: nothing, all objects are stored internally in the class.
        """
        sax = SymbolicAggregateApproximation(n_bins=self.nb_bins, strategy='normal')

        self.sax_data = sax.fit_transform(self.paa_data)

    def _get_words_from_sax(self):
        total_data_length = self.sax_data.shape[1]
        data = self.sax_data[0]
        nb_words = total_data_length - self.window_size + 1

        # Create sliding window over alphabetical time series
        self.sax_word_list = [''.join(list(data[i:i + self.window_size]))
                              for i in range(0, nb_words)]

    def _brute_force_ad_detection(self):
        """
        Detect discords in the time series using the brute force anomaly detection algorithm.
        :return: nothing, discords are stored internally in the class instance.
        """
        for i, p in enumerate(self._segments):
            nearest_dist = np.inf
            for j, q in enumerate(self._segments):
                if np.abs(i - j) >= self.window_size:
                    dist = self._euclidean_distance(p, q)
                    if dist < nearest_dist:
                        nearest_dist = dist

            if self.multiple_discords:
                # self.all_discords[self._segments.index(p)] = nearest_dist
                self.all_discords[i] = nearest_dist

            if nearest_dist > self._best_dist:
                self._best_dist = nearest_dist
                # self.best_loc = self._segments.index(p)
                self.best_loc = i

    def list_anomalies(self):
        if self.mode == 'brute':
            if self.multiple_discords:
                for d in enumerate(self.all_discords):
                    if d[0] <= self.nb_discords - 1:
                        print(f"Discord {(d[0] + 1)} located at index {d[1]}")
                    else:
                        break
            else:
                print(f"Discord located at index {self.best_loc}")
        else:
            raise NotImplementedError('Only Brute force method is implemented so far!')

    def fit(self, data):
        """
        This method will compute all the data necessary to identify the anomalous discords. At the moment, only
        the brute force method is implemented.
        :param data: a numpy array with a shape of (1,n) where n is the length of the time series
        :return: nothing, all objects are stored internally in the class.
        """
        self._load_data(data)
        self._normalize_data()
        self._length = self._norm_data.shape[1]

        if self.mode == 'brute':
            nb_of_segments = self._length - self.window_size + 1
            self._segments = [self._norm_data[:, i:i + self.window_size] for i in range(nb_of_segments)]
        else:
            self._paa()
            self._sax()
            self._get_words_from_sax()

    def transform(self):
        """
        This method will identify the discords in the time series.
        Anomalies can be listed through 3 different methods:
            - by call .list_anomalies()
            - by calling .best_loc (only when multiple_discords is set to False)
            - by calling .all_discords (only when multiple_discords is set to True)
        :return: Nothing.
        """
        if self.mode == 'brute':
            self._brute_force_ad_detection()
        else:
            raise NotImplementedError("Only Brute force method is implemented so far!")

        if self.multiple_discords:
            self.all_discords = {
                k: v for k, v in sorted(self.all_discords.items(),
                                        key=lambda x: -x[1])
            }

    def fit_transform(self, data):
        """
        This method calls the .fit() and the .transform() methods together.
        :param data: a numpy array with a shape of (1,n) where n is the length of the time series
        :return: nothing. See .transform()
        """
        self.fit(data)
        self.transform()

    def __str__(self):
        message = f"Discord size: {self.window_size}, \n " \
                  f"Method: {self._methods[self.mode]}."

        return message

    def __repr__(self):
        return f"{self.__class__.__name__}({self.window_size}, {self.nb_bins})"


class InputError(Exception):
    def __init__(self, message):
        self.message = message


def main():
    DATALENGTH = 500
    W_SIZE = 5

    window_size = W_SIZE
    data = np.array([1] * DATALENGTH)
    data[368:375] = 4
    data[219:245] = 6
    data[127:131] = 10
    hotsax = HotSax(window_size=window_size, mode="brute", multiple_discords=True, nb_discords=20)
    hotsax.fit(data=data)

    hotsax.transform()
    hotsax.list_anomalies()


if __name__ == '__main__':
    main()
