# pyhotsax
Implementation of the HotSax anomaly detection algorithm in Python

Reference paper: HOT SAX: Finding the Most Unusual Time Series Subsequence: Algorithms and Applications: https://cs.gmu.edu/~jessica/HOT_SAX_long_ver.pdf

## Using the algorithm

Only the brute force algorithm is implemented so far. To detect anomalies into a time series, the data must be into a numpy array.

Here is an example of the code which can be found into the `hotsax.py` file:

```
DATALENGTH = 500
W_SIZE = 5

window_size = W_SIZE
data = np.array([1] * DATALENGTH)
data[368:375] = 4
data[219:245] = 6
data[127:131] = 10
hotsax = HotSax(window_size=window_size, mode="brute", multiple_discords=True, nb_discords=5)
hotsax.fit(data=data)

hotsax.transform()
hotsax.list_anomalies()
```
