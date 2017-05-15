# Model to predict totals, which is not dependent on data being from a specific day. 
from trainingset import *
from prediction import *
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
import math
import tensorflow as tf
    
def create_zscore_reversal_func(outputset):
    values = list(map(lambda x: x[0], outputset.data))
    mean = sum(values)/len(values)
    variance = sum(map(lambda x: (x-mean)*(x-mean), values))/len(values)
    stddev = math.sqrt(variance)
    return lambda x: x*stddev+mean

# The range of days from to take data to train on.
STARTDAY = 1 
ENDDAY = 8

# Datafile
details = readfile('../2004data.csv', 'utf-8', ';')
tickets = readfile('../2004tickets.csv', 'utf-8', ';')
weather = readfile('../2004rel_weather.csv', 'utf-8', ';')
dates = readfile('../2004rel_dates.csv', 'utf-8', ';')
budgets = readfile('../2004budget.csv', 'utf-8', ';')
competition = readfile('../2004competition.csv', 'utf-8', ';')
trends = readfile('../2004trends.csv', 'utf-8', ';')

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
inputs = [
    NominalListAttribute(details, 'genres'),
    NumericAttribute(budgets, 'budget'),
    LanguageAttribute(details, 'language'),
    NumericAttribute(details, 'sequel_gross'),
    NominalAttribute(details, 'agerating'),
    NumericAttribute(details, 'actor1_starred_movies'),
    NumericAttribute(details, 'actor1_starred_success'),
    NumericAttribute(details, 'actor2_starred_movies'),
    NumericAttribute(details, 'actor2_starred_success'),
    NumericAttribute(details, 'actor3_starred_movies'),
    NumericAttribute(details, 'actor3_starred_success'),
    NumericAttribute(details, 'director_movies'),
    NumericAttribute(details, 'director_success'),
    NumericAttribute(details, 'length'),
    NumericAttribute(trends, 'maxtrend'),
    NumericAttribute(trends, 'avg_trend'),
    MeanAttribute(weather, 'weather_rain+', 8),
    MeanAttribute(weather, 'weather_sun+', 8),
    MeanAttribute(weather, 'weather_lowtemp+', 8),
    MeanAttribute(weather, 'weather_hightemp+', 8),
    NumericAttribute(dates, 'year+0'),
    NumericAttribute(dates, 'weeknum+0'),
    MeanAttribute(dates, 'holiday_effect+', 7),
]

# Add competition data
for day in range(5):
    inputs.append(NumericAttribute(competition, 'day_diff_'+str(day)))
    inputs.append(NumericAttribute(competition, 'budget_'+str(day)))
    
outputs = [ 
    NumericAttribute(tickets, 'post_release')
]

trainingsets = []
for day in range(STARTDAY, ENDDAY):
    newinputs = list(inputs)
    newoutputs = list(outputs)

    newinputs.append(NumericAttribute(trends, 'day'+str(day-1)))
    newinputs.append(NumericAttribute(trends, 'day'+str(day-2)))
    newinputs.append(NumericAttribute(trends, 'day'+str(day-3)))
    newinputs.append(MeanAttribute(trends, 'day', day-3, range_start=-14))
    if day >= 0:
        newinputs.append(NumericAttribute(tickets, 'pre_release'))
        newinputs.append(NumericAttribute(tickets, 'pre_release_showings'))
    else:
        newinputs.append(NumericAttribute(tickets, 'pre_release_tickets'+str(day)))
        newinputs.append(NumericAttribute(tickets, 'pre_release_showings'+str(day)))
    if day >= 0:
        newinputs.append(NumericAttribute(tickets, 'agg_tickets+'+str(day)))
        newinputs.append(NumericAttribute(tickets, 'agg_showings+'+str(day)))
    else:
        newinputs.append(ConstantAttribute(tickets, 'tickets', 0))
        newinputs.append(ConstantAttribute(tickets, 'showings', 0))
    newinputs.append(ConstantAttribute(trends, 'days_to_premiere', -day))
    trainingsets.append(Trainingset.describe(newinputs, newoutputs))

newinput = Dataset(trainingsets[0].input.headers, 
                   np.concatenate(list(map(lambda t: t.input.data, trainingsets))))
newoutput = Dataset(trainingsets[0].output.headers,
                    np.concatenate(list(map(lambda t: t.output.data, trainingsets))))
trainingset = Trainingset(newinput, newoutput)
reversal_func = create_zscore_reversal_func(trainingset.output)
trainingset.normalize_zscore()

splits = trainingset.split_stratified_buckets(10)
inputs, outputs = zip(*splits)
inputshape = inputs[0].shape[1]
outputshape = 1

graphs = {}
for i in range(10):
    print('Running {}'.format(i))
    training_in = np.concatenate(inputs[:i]+inputs[(i+1):])
    training_out = np.concatenate(outputs[:i]+outputs[(i+1):])
    test_in = val_in = inputs[i]
    test_out = val_out = outputs[i]
    
    dropout = 0.1
    model = Sequential([ 
        Dense(200, input_dim = inputshape),
        Activation('relu'),
        Dropout(dropout),
        Dense(190, input_dim = inputshape),
        Activation('relu'),
        Dropout(dropout),
        Dense(180, input_dim = inputshape),
        Activation('relu'),
        Dropout(dropout),
        Dense(170, input_dim = inputshape),
        Activation('relu'),
        Dropout(dropout),
        Dense(160, input_dim = inputshape),
        Activation('relu'),
        Dropout(dropout),
        Dense(150, input_dim = inputshape),
        Activation('relu'),
        Dropout(dropout),
        Dense(140, input_dim = inputshape),
        Activation('relu'),
        Dropout(dropout),
        Dense(130, input_dim = inputshape),
        Activation('relu'),
        Dropout(dropout),
        Dense(120, input_dim = inputshape),
        Activation('relu'),
        Dropout(dropout),
        Dense(outputshape,activation = 'linear')
    ])
    sgd = SGD(lr = 0.0015, momentum = 0.1, decay = 0, nesterov = True) #defaults: lr=0.01, 0, 0, False
    adam = Adam(lr=0.00005, decay = 0.00) #defaults lr = 0.001, decay = 0.0
    model.compile(
        loss      = 'mean_squared_error',
        optimizer = sgd,
        metric    = 'accuracy'
    )
    model.fit(training_in, training_out,
        batch_size      = 32,
        nb_epoch        = 1000,
        verbose         = 0,
        validation_data = (val_in, val_out))

    predict = model.predict(test_in, batch_size=32)
    graph = list(zip(map(lambda l: l[len(l)-1], test_in), 
                        map(lambda x: reversal_func(x[0]), predict),
                        map(lambda x: reversal_func(x[0]), test_out)))
    buckets = {}
    for day, pred, true in graph:
        if day in buckets:
            buckets[day].append((pred, true))
        else:
            buckets[day] = [(pred, true)]

    for day, graph in buckets.items():
        if day in graphs:
            graphs[day] += graph
        else:
            graphs[day] = graph

for day, graph in sorted(graphs.items()):
    print('GRAPH DAY {}'.format(day))
    print(graph)
    plot_in, plot_out = zip(*sorted(graph))
    plt.plot(plot_in, 'r.')
    plt.plot(plot_out, 'b.')
    plt.show()
