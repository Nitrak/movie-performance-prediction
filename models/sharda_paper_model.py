#
# Running the model from Sharda and Delen (2006) on our dataset.
#
from trainingset import *
from prediction import *
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.regularizers import l2

detailsfile = readfile('../2004data.csv', 'utf-8', ';')
ticketsfile = readfile('../2004tickets.csv', 'utf-8', ';')
budgetsfile = readfile('../2004budget.csv', 'utf-8', ';')
starfile = readfile('../starpower.csv', 'utf-8', ';')

def competition(release):
    match = re.match('\d{4}-(\d{2})-\d{2}', release)
    if match:
        month = int(match.group(1))
        if month == 6 or month == 11: return 'high'
        if month == 5 or month == 7 or month == 12: return 'medium'
        return 'low'
    else:
        return ''

inputs = [
    NominalAttribute(detailsfile, 'agerating'),
    NominalAttribute(detailsfile, 'release', preprocess = competition ),
    NumericAttribute(starfile, 'actor_starpower'),
    NominalListAttribute(detailsfile, 'genres'),
    NumericAttribute(budgetsfile, 'budget'),
    PresenceAttribute(detailsfile, 'sequelid'),
    NumericAttribute(ticketsfile, 'showings+7')
]
outputs = [
    FrequencyBucketAttribute(ticketsfile, 'post_release', 9)
]
trainingset = Trainingset.describe(inputs, outputs)
output_trainingset(trainingset)
trainingset.normalize_zscore()

splits = trainingset.split_stratified(10)
inputs, outputs = zip(*splits)
training_in = np.concatenate(inputs[0:8])
training_out = np.concatenate(outputs[0:8])
validation_in = inputs[8]
validation_out = outputs[8]
test_in = inputs[9]
test_out = outputs[9]

for i in range(10):
    model = Sequential([ 
        Dense(18, 
            activation = 'sigmoid',
            input_dim  = training_in.shape[1]),
        Dense(16, activation = 'sigmoid'),
        Dense(training_out.shape[1],
            activation = 'sigmoid',
            #Dont use regularizer
        ),
    ])

    sgd = SGD(lr = 1e-3, decay = 0.0000)
    model.compile(
        loss      = 'categorical_crossentropy',
        optimizer = sgd,
        metric    = 'categorical_accuracy')
    model.fit(training_in, training_out,
        nb_epoch = 2000,
        verbose  = 1,
        validation_data = (validation_in, validation_out))
    run_classification_predict(model, test_in, test_out, i)

