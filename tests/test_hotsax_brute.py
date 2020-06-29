from hotsax import HotSax, InputError
from pytest import raises, approx
import pytest
import numpy as np

DATALENGTH = 50
W_SIZE = 5


@pytest.fixture()
def setup_brute():
    window_size = W_SIZE
    data = [i for i in range(1, DATALENGTH + 1)]
    hotsax = HotSax(window_size=window_size, mode='brute')
    return hotsax, data


def test_data_load_shape(setup_brute):
    hotsax, data = setup_brute
    hotsax._load_data(data=np.array(data))

    assert hotsax.data.shape == (1, DATALENGTH)


def test_data_load_as_numpy(setup_brute):
    hotsax, data = setup_brute

    with raises(InputError):
        hotsax._load_data(data=data)


def test_data_normalization(setup_brute):
    hotsax, data = setup_brute

    hotsax._load_data(data=np.array(data))
    hotsax._normalize_data()

    m = np.mean(data)
    sd = np.std(data)
    data = np.array(data).reshape(1, -1)
    norm_data = (data - m) / sd

    assert hotsax._norm_data == approx(norm_data)


def test_segments(setup_brute):
    hotsax, data = setup_brute
    hotsax.fit(data=np.array(data))

    assert len(hotsax._segments) == DATALENGTH - W_SIZE + 1
    for s in hotsax._segments:
        assert s.shape == (1, W_SIZE)


def test_euclidean_distance_inputs(setup_brute):
    hotsax, _ = setup_brute

    inputs = [('abc', 123), (123, 'abc'), (np.array([1, 2, 3]), np.array([1, 2, 3, 4]))]
    for inpt in inputs:
        with raises(InputError):
            hotsax._euclidean_distance(inpt[0], inpt[1])


@pytest.mark.parametrize("test_input1, test_input2 ,expected",
                         [(np.array([0, 0, 0]), np.array([0, 0, 0]), 0),
                          (np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1]), 0),
                          (np.array([1, 2, 3]), np.array([7, 6, 5]),
                           np.sqrt((W_SIZE / DATALENGTH) * np.sum((np.array([1, 2, 3]) - np.array([7, 6, 5])) ** 2)))])
def test_euclidean_distance(setup_brute, test_input1, test_input2, expected):
    hotsax, data = setup_brute
    hotsax.fit(data=np.array(data))
    assert hotsax._euclidean_distance(test_input1, test_input2) == approx(expected)


def test_anomaly_detection_brute_force():
    window_size = W_SIZE
    data = np.array([1] * DATALENGTH)
    data[37:45] = 4
    hotsax = HotSax(window_size=window_size, mode="brute", multiple_discords=True, nb_discords=5)
    hotsax.fit_transform(data=data)

    results = list(hotsax.all_discords.keys())[:5]
    assertion = [i in results for i in range(35, 38)]
    assert any(assertion)
