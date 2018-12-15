import numpy as np
import decimal

double = np.vectorize(decimal.Decimal)
floor = np.vectorize(int)

