import numpy as np
import math

class Series:
    def __init__(self, x: np.ndarray, y: np.ndarray, y_std: np.ndarray = None):
        self.x = x
        self.y = y
        if y_std is None:
            self.y_std = np.zeros_like(y)
        else:
            self.y_std = y_std
    def mean(self):
        return self._mean(self.x, self.y)
    def rms(self):
        return np.sqrt(self._mean(self.x, self.y**2))
    @staticmethod
    def _mean(x, y):
        area = np.trapz(y, x)
        x_range = np.ptp(x)
        return area/x_range

class Statistic:
    def __init__(self):
        pass
    def __call__(self, input: np.ndarray) -> Series:
        raise AttributeError('Must define function for generating output!')
    @property
    def name(self):
        return self.__class__.__name__


def density_series(input: np.ndarray, bins, data_range=None):
    histogram, bin_edges = np.histogram(input, bins=bins, range=data_range)
    bin_means = (bin_edges[:-1]+bin_edges[1:])/2
    return Series(bin_means, histogram)

def expspace(min, max, num=50):
    return np.exp(np.linspace(min, max, num))

def difference_series(series1: Series, series2: Series) -> Series:
    # find the combined x coordinates of the new series
    combined_x = np.union1d(series1.x, series2.x)
    # use linear interpolation to find the corresponding y coordinates
    series1_y_interp = np.interp(combined_x, series1.x, series1.y)
    series2_y_interp = np.interp(combined_x, series2.x, series2.y)
    series1_y_std_interp = np.interp(combined_x, series1.x, series1.y_std)
    series2_y_std_interp = np.interp(combined_x, series2.x, series2.y_std)
    # create a vector function for the relative difference between the two series
    @np.vectorize
    def relative_diff(point1, point2):
        if point1==point2==0:
            return 0
        else:
            return (point1-point2)/np.sqrt(point1**2 + point2**2)
    
    @np.vectorize
    def relative_diff_std(point1, point2, std1, std2):
        if std1==std2==0 or point1==point2==0:
            return 0
        else:
            return np.abs(point1+point2)*np.sqrt((point1**2+point2**2)**-3)*np.sqrt((point1*std2)**2 + (point2*std1)**2)
    # use the vector function to calculate a new difference series
    diff_series = Series(combined_x, relative_diff(series1_y_interp, series2_y_interp), relative_diff_std(series1_y_interp, series2_y_interp, series1_y_std_interp, series2_y_std_interp))
    return diff_series