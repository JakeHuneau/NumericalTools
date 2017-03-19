import numpy as np
from scipy.signal import butter, lfilter, cheby1, cheby2, filtfilt
from scipy.optimize import curve_fit

pi = np.pi

#Numerical Tools
def two_point_diff(x,y):
    '''
    2 point numerical derivate. does dy/dx
    
    -Input-
    x: array. that y data is dependent of. 
    y: array. y data that is being differentiated with respect to x data.
    
    -Returns-
    dyf: array of y derivates.
    '''
    dyf = np.zeros(len(x))
    for i in range(0,len(x)-1):
        dyf[i] = (y[i+1] - y[i]) / (x[i+1] - x[i])
    dyf[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    return dyf

def three_point_diff(x,y):
    '''
    3 point numerical derivate. does dy/dx
    
    -Input-
    x: array. that y data is dependent of. 
    y: array. y data that is being differentiated with respect to x data.
    
    -Returns-
    dyf: array of y derivates.
    '''
    dyf = np.zeros(len(x))
    for i in range(0,len(x)-2):
        dyf[i] = (-1.5 * y[i] + 2 * y[i+1] - .5 * y[i+2]) / (x[i+1] - x[i])
    dyf[-2] = (1.5 * y[-2] - 2 * y[-3] + .5 * y[-4]) / (x[-2] - x[-3])
    dyf[-1] = (1.5 * y[-1] - 2 * y[-2] + .5 * y[-3]) / (x[-1] - x[-2])
    return dyf

def four_point_diff(x,y):
    '''
    4 point numerical derivate. does dy/dx
    
    -Input-
    x: array. that y data is dependent of. 
    y: array. y data that is being differentiated with respect to x data.
    
    -Returns-
    dyf: array of y derivates.
    '''
    dyf = np.zeros(len(x))
    for i in range(0, len(x)-3):
        dyf[i] = (-11.0/6.0 * y[i] + 3 * y[i+1] - 1.5 * y[i+2] + 1.0/3.0 *\
                 y[i+3]) / (x[i+1] - x[i])
    dyf[-3] = (11.0/6.0 * y[-3] - 3 * y[-4] + 1.5 * y[-5] - 1.0/3.0 * y[-6])\
              / (x[2] - x[1])
    dyf[-2] = (11.0/6.0 * y[-2] - 3 * y[-3] + 1.5 * y[-4] - 1.0/3.0 * y[-5])\
              / (x[2] - x[1])
    dyf[-1] = (11.0/6.0 * y[-1] - 3 * y[-2] + 1.5 * y[-3] - 1.0/3.0 * y[-4])\
              / (x[2] - x[1])
    return dyf

def five_point_diff(x,y):
    '''
    5 point numerical derivate. does dy/dx
    
    -Input-
    x: array. that y data is dependent of. 
    y: array. y data that is being differentiated with respect to x data.
    
    -Returns-
    dyf: array of y derivates.
    '''
    dyf = np.zeros(len(x))
    for i in range(0, len(x) - 4):
        dyf[i] = -25.0/12.0 * y[i] + 4 * y[i+1] - 3 * y[i+2] +4.0/3.0 *\
                 y[i+3] -.25 * y[i+4]
    dyf[-4] = 25.0/12.0 * y[-4] - 4 * y[-5] + 3 * y[-6] - 4.0/3.0 * y[-7] +\
              .25 * y[-8]
    dyf[-3] = 25.0/12.0 * y[-3] - 4 * y[-4] + 3 * y[-5] - 4.0/3.0 * y[-6] +\
              .25 * y[-7]
    dyf[-2] = 25.0/12.0 * y[-2] - 4 * y[-3] + 3 * y[-4] - 4.0/3.0 * y[-5] +\
              .25 * y[-6]
    dyf[-1] = 25.0/12.0 * y[-1] - 4 * y[-2] + 3 * y[-3] - 4.0/3.0 * y[-4] +\
              .25 * y[-5]
    return dyf

def average_block(data, time_step, sample_rate = 1):
    '''Calculates non-overlapping averaging blocks of time in seconds. To 
    calculate by number of items, keep sample_rate = 1 and set time_step to
    be equal to number of items to average together.'''
    step = int(time_step * sample_rate)
    run_av = []
    n = 0
    while n < len(data):
        av = np.mean(data[n:n+step])
        run_av.append(av)
        n+= (step)
    return np.array(run_av)

def arctan_unwrap(phase_data):
    '''Unwraps the angle output of arctan in numpy'''
    phase = [2 * (x + np.pi/2) for x in phase_data]
    phase = np.unwrap(phase)
    phase = [(x/2.0 - pi/2) for x in phase]
    return phase
   
#Fits
def exp_dec(t, A, tau):
    '''
    Exponential decay equation.
    
    --Inputs--
    t: (float) time for exponential decay
    A: (float) Initial amplitude of exponenital decay
    tau: (float) Decay constant
    
    --Returns--
    (float) exponential decay at time t.
    '''
    return A * np.exp(-t / tau)
    
def exp_dec2(t, A, tau, offset):
    '''
    Exponential decay equation with an offset.
    
    --Inputs--
    t: (float) time for exponential decay
    A: (float) Initial amplitude of exponenital decay
    tau: (float) Decay constant
    offset: (float) Offset for decay
    
    --Returns--
    (float) exponential decay at time t.
    '''
    return A * np.exp(-t / tau) + offset
    
def fit_exp(x_data, y_data, offset = False):
    '''
    For fitting an exponential decay to a dataset
    
    --Inputs--
    x_data: (array) array of x-data for fit
    y_data: (array) array of y-data for fit
    offset: (boolean) Set true if there is an offset for the fit
    
    --Returns--
    popt: (array) [amplitude fit, tau fit, (offset fit if offset true)]
    perr: (array) [amplitude error, tau error, (offset error if offset true)]
    '''
    max_y = max(y_data)
    guess_tau_value = max_y / np.e
    tau_loc = (np.abs(y_data - guess_tau_value)).argmin()
    guess_tau = x_data[tau_loc]
    if not offset:
        popt, pcov = curve_fit(exp_dec, x_data, y_data, p0 =\
                     [max_y, guess_tau]) #pcov can make error
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    else: #with offset
        offset_guess = y_data[-1]
        popt, pcov = curve_fit(exp_dec2, x_data, y_data, p0 =\
                     [max_y, guess_tau, offset_guess]) #pcov can make error
        perr = np.sqrt(np.diag(pcov))
        return popt, perr

#List Modifiers
def get_chopped_time_indices(chopped_time, time):
    '''
    Used for getting indices when trying to chop a time series. For example,
    if it is desired for a time series that is normally 5000 seconds long, 
    this can be used to find the indeces for making the time series from 
    300-1200 seconds. 
    
    --Inputs--
    chopped_time: (list, int or float) If a list then list is 
                  [start time, end time]. If it is a single value then the
                  time is just a start time and the end time is the end of the
                  time series.
    time: (list) time array.
    
    --Returns--
    chop_indices: (list) a list of the indices of the beginning and end times
                  for the chop.
    '''
    if type(chopped_time) == int or type(chopped_time) == float:
        start_time = chopped_time
        time_loc_start = min(range(len(time)), key=lambda i: \
                         abs(time[i]-start_time))
        return [time_loc_start, len(time)]
        
    if type(chopped_time) == list and len(chopped_time) == 2:
        start_time, end_time = chopped_time
        time_loc_start = min(range(len(time)), key=lambda i: \
                         abs(time[i]-start_time))
        time_loc_end = min(range(len(time)), key=lambda i: \
                       abs(time[i]-end_time))
        return [time_loc_start, time_loc_end]


#Filters
def cheby_lowpass_filter(data, cutoff, fs, order = 5, filter_number = 1):
    '''Creates a lowpass filter using chebyshev type 1 filter'''
    nyq = 0.5 * fs
    normalized_cutoff = cutoff / nyq
    if filter_number == 1:
        b, a = cheby1(order, 1, normalized_cutoff, btype='low', analog=False)
    elif filter_number == 2:
        b, a = cheby2(order, 1, normalized_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def butter_lowpass_filter(data, cutoff, fs, order=5):
    '''Creates a lowpass filter for the data'''
    nyq = 0.5 * fs
    normalized_cutoff = cutoff / nyq
    b, a = butter(order, normalized_cutoff, btype='low', analog=False) 
    y = lfilter(b, a, data) #filters data
    return y
    
def butter_bandpass(data, lowcut, highcut, fs, order=5):
    '''Creates a bandpass for the data'''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    y = lfilter(b,a,data)
    return y
    
def butter_highpass_filter(data, cutoff, fs, order=5):
    '''Creates a highpass filter for the data'''
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False) 
    y = lfilter(b, a, data) #filters data
    return y
    
#statistics
def chi_square(expected_array, actual_array, uncertainty):
    '''
    Calculates chi-squared value to show how good a fit is. Found on page 268
    of Taylor's Error Analysis
    --Inputs--
    expected_array: array of points calculated from fit to data set.
    actual_array: array of points that fit was calculated for.
    uncertainty: uncertainty of points of actual_array
    --Returns--
    chi_square: chi-square value showing goodness of fit. Unitless
    '''
    return np.sum([((act - exp) / uncertainty) ** 2 for act,exp in \
    zip(expected_array, actual_array)])
    
def reduced_chi_square(chi_square, number_of_points, contraints):
    '''
    Calculates the reduced chi-squared, described in detail on page 268 of
    Taylor's Error Analysis. Can be used to show how far chi-square is from
    degrees of freedom, which is what it should be equal to. If less than an
    order of 1, then the fit is likely correct. If more than an order of 1, 
    then fit is likely to be incorrect.
    --Inputs--
    chi_square: float of chi-square value of data
    number_of_points: int of number of data points
    constraints: int of number of contraints for fit applied to data
    --Returns--
    reduced_chi_square: float of reduced chi sqaure. Unitless
    '''
    dof = number_of_points - contraints #degrees of freedom
    return chi_square / dof
    
def estimate_uncertainty_chi(expected_array, actual_array, constraints):
    '''
    estimates the uncertainty in the data by finding the uncertainty where
    the chi-square is equal to the degree of freedom.
    --Inputs--
    expected_array: array of points calculated from fit to data set.
    actual_array: array of points that fit was calculated for.
    constraints: int of number of contraints for fit applied to data
    --Returns--
    uncertainty: float of estimated uncertainty 
    '''
    chi = chi_square(expected_array, actual_array, 1)
    n = len(actual_array)
    reduced_chi = reduced_chi_square(chi, n, constraints)
    return np.sqrt(reduced_chi)
    
def calc_noise(data, fit_data, p):
    '''
    Calculates the noise level of the data from the residuals of a fit
    applied to the data. Formula is noise = sqrt( 1/(n-p) sum( y - y_fit) )
    where n is the number of points, p is the number of free variables,
    y is a datapoint and y_fit is a fit datapoint. This formula was found at
    http://www.stat.colostate.edu/regression_book/chapter9.pdf
    
    --Inputs--
    data: (array) data the fit was applied to
    fit_data: (array) expected data from a fit
    p: (int) number of variables in fit
    
    --Returns--
    noise: (float) noise level
    '''
    residual_total = np.sum([(y - y_fit)**2 for y, y_fit in\
                     zip(data, fit_data)])
    return np.sqrt(residual_total/(len(data)-p))
    