import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np

"""
below is implementation of mean, median, mode, standard deviation and variance as per the formulas.
"""


def mean_of(data):
    N = len(data)
    if N == 0:
        return 0
    sum_of_all_values = sum(data)
    return sum_of_all_values / N


def standard_deviation(data):
    if len(data) == 0:
        return 0
    mean = mean_of(data)
    return math.sqrt(1.0 / len(data) * sum([(x - mean) ** 2 for x in data]))


def variance_of(data):
    return standard_deviation(data) ** 2


def median(data):
    length_of_data = len(data)
    if length_of_data == 0:
        return None
    else:
        sorted_data = sorted(data)
        if length_of_data % 2 == 0:

            return (sorted_data[int(length_of_data / 2)] + sorted_data[int(length_of_data / 2) - 1]) / 2
        else:
            return sorted_data[math.floor(length_of_data / 2)]


def mode(data):
    return max(set(data), key=data.count)


def otsu(data):
    best_mixed_variance = 9999999
    best_threshold = max(data)
    for threshold in np.arange(min(data), max(data),
                               0.01):

        wt_under = len([x for x in data if x <= threshold]) / len(data)
        var_under = variance_of([y for y in data if y <= threshold])
        wt_over = len([x for x in data if x > threshold]) / len(data)
        var_over = variance_of([y for y in data if y > threshold])

        mixed_variance = wt_under * var_under + wt_over * var_over
        if mixed_variance < best_mixed_variance:
            best_mixed_variance = mixed_variance
            best_threshold = threshold
        # else:
        #     if mixed_variance == best_mixed_variance:
        #         best_threshold.append(threshold)

    return best_threshold


"""
function to process mystery data.
"""


def process_mystery_data(data):
    exploratory_data_analysis = {'mean': mean_of(data), 'median': median(data), 'mode': mode(data),
                                 'standard deviation': standard_deviation(data),
                                 'mid-range': (min(data) + max(data)) / 2}
    print("average of mystery data: ",mean_of(data) )
    print("median of mystery data: ", median(data))
    print("mode of mystery data: ", mode(data))
    print("standard deviation of mystery Data:", standard_deviation(data))
    print("mid range of mystery data:", (min(data) + max(data)) / 2)

    return exploratory_data_analysis


"""
The main function perfoems the tasks step by step as per the guildlines in the HW.
"""


def main():
    """
    Read the data file and rename the headers for easy access, then calculate the stiffness coefficient and then calculate the
    quantized coefficients. Plot the histogram of the quantized coefficients.
    """
    df = pd.read_csv("CSCI420_2161_HW02_Unclassified_Data.csv", header=0)
    df = df.rename(columns={'Force (Newtons)': 'Force', 'Deflection (mm)': 'Deflection'})
    df['Stiffness coefficient'] = df['Deflection'] / df['Force']
    df['Quantized coefficient'] = pd.Series([round(x * 50) / 50 for x in df['Stiffness coefficient'].tolist()]).values

    fig = plt.figure(1)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(df['Force'].tolist(), df['Deflection'].tolist(), label="Data")
    ax.set_title("Data Force vs Deflection")
    ax.legend(loc='lower right')
    ax.set_xlabel('Force')
    ax.set_ylabel('Deflection')

    fig = plt.figure(2)
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(df['Quantized coefficient'].tolist(), label='Quantized Stiffness Coefficient')
    ax.set_title("Histogram")
    ax.legend(loc='lower right')

    """
    Get the quantized data and pass it to otsu's algorithm to get the list of thresholds back
    """
    data = df['Quantized coefficient'].tolist()
    threshold = otsu(data)
    print("first cluster threshold:", threshold)
    data = [x for x in data if x >= (threshold)]
    print("second cluster threshold:", otsu(data), "\n")

    df_mystery = pd.read_csv("Mystery_Data.csv", header=0)
    df_mystery = df_mystery.rename(columns={'Measures (Mystery Units)': 'Measures'})
    data = df_mystery['Measures'].tolist()

    # print(df_mystery)

    before_removal = process_mystery_data(data)
    print("\n")
    data = data[0:-1]

    print("After removal of last element")
    after_removal = process_mystery_data(data)

    """
    Find the least changed central tendency parameter.
    """

    minimum = 100000
    measure = ''
    for key, value in before_removal.items():
        percentage_change = abs(value - after_removal[key]) / value
        if percentage_change < minimum:
            measure = [key]
            minimum = percentage_change
        else:
            if percentage_change ==minimum:
                measure.append(key)
    print("\nThe least change in central tendency is: ", measure)
    plt.show()


if __name__ == '__main__':
    main()
