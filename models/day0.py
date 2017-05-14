from trainingset import *
from prediction import *
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.regularizers import l1, l2
import keras
import math
import tensorflow as tf

details = readfile('../2004data.csv', 'utf-8', ';')
tickets = readfile('../2004tickets.csv', 'utf-8', ';')
weather = readfile('../2004rel_weather.csv', 'utf-8', ';')
dates = readfile('../2004rel_dates.csv', 'utf-8', ';')
budgets = readfile('../2004budget.csv', 'utf-8', ';')
competition = readfile('../2004competition.csv', 'utf-8', ';')
trends = readfile('../2004trends.csv', 'utf-8', ';')
cinemas = readfile('../2004premiere_cinemas.csv', 'utf-8', ';')

TARGET_DAY = 0

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
inputs = [
    NominalListAttribute(details, 'genres'),
    NumericAttribute(budgets, 'budget'),
    LanguageAttribute(details, 'language'),
    NumericAttribute(details, 'sequel_gross'),
    #DateAttribute(details, 'first_showing'),
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
    NumericAttribute(tickets, 'pre_release_tickets'+str(TARGET_DAY)),
    NumericAttribute(tickets, 'pre_release_showings'+str(TARGET_DAY)),
#    NumericAttribute(tickets, 'cinemas'),
    MeanAttribute(weather, 'weather_rain+', 8),
    MeanAttribute(weather, 'weather_sun+', 8),
    MeanAttribute(weather, 'weather_lowtemp+', 8),
    MeanAttribute(weather, 'weather_hightemp+', 8),
    NumericAttribute(dates, 'year+0'),
    NumericAttribute(dates, 'weeknum+0'),
    NumericAttribute(dates, 'day_in_week+0'),
    MeanAttribute(dates, 'holiday_effect+', 7),
    NumericAttribute(trends, 'day'+str(TARGET_DAY-1)),
    NumericAttribute(trends, 'day'+str(TARGET_DAY-2)),
    NumericAttribute(trends, 'day'+str(TARGET_DAY-3)),
#    NumericAttribute(trends, 'delta'+str(TARGET_DAY-1)),
    MeanAttribute(trends, 'day', TARGET_DAY-3, range_start=-14)
]

for day in range(0, TARGET_DAY):
    inputs.append(NumericAttribute(tickets, 'tickets+'+str(day)))
    inputs.append(NumericAttribute(tickets, 'showings+'+str(day)))

# Add trend data
#for day in range(-14, TARGET_DAY):
#   inputs.append(NumericAttribute(trends, 'day'+str(day)))

# Add competition data
for day in range(5):
    inputs.append(NumericAttribute(competition, 'day_diff_'+str(day)))
    inputs.append(NumericAttribute(competition, 'budget_'+str(day)))

# Add date data
#for day in range(7):
#    inputs.append(NumericAttribute(dates, 'year+' + str(day)))
#    inputs.append(NumericAttribute(dates, 'weeknum+' + str(day)))
#    inputs.append(NumericAttribute(dates, 'day_in_week+' + str(day)))
#    inputs.append(NumericAttribute(dates, 'holiday_effect+' + str(day)))

# Add weather data
#for day in range(8):
#    inputs.append(NumericAttribute(weather, 'weather_rain+' + str(day)))
#    inputs.append(NumericAttribute(weather, 'weather_sun+' + str(day)))
#    inputs.append(NumericAttribute(weather, 'weather_lowtemp+' + str(day)))
#    inputs.append(NumericAttribute(weather, 'weather_hightemp+' + str(day)))
    
outputs = [ 
    NumericAttribute(tickets, 'post_release')
    #NumericAttribute(tickets, 'agg_tickets+4')
    #NumericAttribute(tickets, 'agg_tickets+40')
    #NumericAttribute(tickets, 'avg_tickets_per_showing+40')
]

def create_zscore_reversal_func(outputset):
    values = list(map(lambda x: x[0], outputset.data))
    mean = sum(values)/len(values)
    variance = sum(map(lambda x: (x-mean)*(x-mean), values))/len(values)
    stddev = math.sqrt(variance)
    rev_func = lambda x: x*stddev+mean
    def error_func(y_true, y_pred):
        tf_mean = tf.cast(tf.constant(mean), tf.float32)
        act_true = tf.multiply(tf.add(y_true, tf_mean), tf.constant(stddev))
        act_pred = tf.multiply(tf.add(y_pred, tf_mean), tf.constant(stddev))
        return tf.reduce_mean(tf.abs(tf.subtract(act_true, act_pred)))
    return (rev_func, error_func)

trainingset = Trainingset.describe(inputs, outputs)
reversal_func, actual_percentage_error = create_zscore_reversal_func(trainingset.output)

trainingset.normalize_zscore()
output_trainingset(trainingset)

splits = trainingset.split_stratified(10)
inputs, outputs = zip(*splits) 
inputshape = inputs[0].shape[1]
outputshape = 1
#http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw


def create_model(training_in, training_out, val_in, val_out):
    dropout = 0.1
    regularization_level = 0.00000
    model = Sequential([ 
        Dense(200, input_dim = inputshape),
        Activation('relu'),
        Dropout(dropout),
        Dense(190),
        Activation('relu'),
        Dropout(dropout),
        Dense(180),
        Activation('relu'),
        Dropout(dropout),
        Dense(170),
        Activation('relu'),
        Dropout(dropout),
        Dense(160),
        Activation('relu'),
        Dropout(dropout),
        Dense(150),
        Activation('relu'),
        Dropout(dropout),
        Dense(140),
        Activation('relu'),
        Dropout(dropout),
        Dense(130),
        Activation('relu'),
        Dropout(dropout),
        Dense(120),
        Activation('relu'),
        Dropout(dropout),
        Dense(outputshape,activation = 'linear')
    ]) 
    sgd = SGD(lr = 0.0015, momentum = 0.2, decay = 0, nesterov = True) #defaults: lr=0.01, 0, 0, False
    adam = Adam(lr=0.00005, decay = 0.00) #defaults lr = 0.001, decay = 0.0
    model.compile(
        loss      = 'mean_squared_error',#actual_percentage_error,
        optimizer = sgd,
        metric    = 'accuracy'
    )
    model.fit(training_in, training_out,
        batch_size      = 16,
        nb_epoch        = 2000,
        verbose         = 1,
        validation_data = (val_in, val_out),
        #callbacks = [keras.callbacks.EarlyStopping(
        #    monitor='val_loss', min_delta=0, patience=400, verbose=1, mode='auto')]
    )
    return model

allinputs = []
alloutputs = []
graph = []
for i in range(2,3):
    print('Running {}'.format(i))
    training_in = np.concatenate(inputs[:i]+inputs[(i+1):])
    training_out = np.concatenate(outputs[:i]+outputs[(i+1):])
    #training_in = np.concatenate(inputs[:])
    #training_out = np.concatenate(outputs[:])
    val_in = test_in = inputs[i]
    val_out = test_out = outputs[i]
   
    model1 = create_model(training_in, training_out, val_in, val_out)
#    model2 = create_model(training_in, training_out, val_in, val_out)
#    model3 = create_model(training_in, training_out, val_in, val_out)
    # training_in, training_out = jitter(training_in, training_out, 
    #     [(0, 1, 0)], 0.0)
    #    [(0, 0.5, 1), (0.5, 6/7, 2), (6/7, 1, 4)], 0.05)
    #score = model.evaluate(test_in, test_out)
    predict = model1.predict(test_in, batch_size=32)
#    predict1 = model1.predict(test_in, batch_size=32)
#    predict2 = model2.predict(test_in, batch_size=32)
#    predict3 = model3.predict(test_in, batch_size=32)
#    predict = list(map(lambda t: (t[0]+t[1]+t[2])/3, zip(predict1, predict2, predict3)))
    subgraph = list(zip(map(lambda x: reversal_func(x[0]), test_out), 
                        map(lambda x: reversal_func(x[0]), predict)))
    #print('Score: {}'.format(score))
    absolute_diff = sum(map(lambda p: abs(p[1]-p[0]), subgraph))/len(subgraph)
    print('mean absolute deviation: {}\n'.format(absolute_diff))
    percentage_diff = sum(map(lambda p: abs(p[1]-p[0])/p[0], subgraph))/len(subgraph)
    print('mean percentage deviation: {}\n'.format(percentage_diff))
    graph = graph + subgraph
    #profile_network(model, training_in, headers=trainingset.input.headers, reverse_func = reversal_func)
print(graph) 
#print(graph)
plot_inputs, plot_outputs = zip(*sorted(graph))
plt.plot(plot_inputs, 'r.')
plt.plot(plot_outputs, 'b.')
plt.show()

