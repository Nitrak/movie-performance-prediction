from trainingset import * 
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
import gc

np.random.seed(123)

def main():
    details  = readfile('../2004data.csv', 'utf-8', ';')
    dstfile = readfile('../dst_movies.csv', 'utf-8', ';')
    premierfile = readfile('../2004tickets.csv','utf-8', ';')

    inputs = [
        NumericAttribute(details, 'budget'), 
        LanguageAttribute(details, 'language'),
        #DateAttribute(details,'release'),
        NominalListAttribute(details, 'genres'),
        #NominalAttribute(dstfile, 'agerating'),
        #NumericAttribute(dstfile, 'length',
        #    lambda x: (x-minlength)/(maxlength-minlength)),
        #NumericAttribute(starfile, 'actor_starpower'),
        #NumericAttribute(starfile, 'director_starpower'),
        PresenceAttribute(details, 'sequelid'),
        #DateAttribute(premierfile, 'premier_date'),
        #NumericAttribute(premierfile, 'premier_showings'),
        #NumericAttribute(premierfile, 'premier_gross',
        #    lambda x: (x-mingross)/(maxgross-mingross)),
        #FrequencyBucketAttribute(premierfile, 'premier_gross', 9)
    ]
    outputs = [
        FrequencyBucketAttribute(premierfile, 'total', 9)
        #FrequencyBucketAttribute(dstfile, 'sold', 9)
    ]

    trainingset = Trainingset.describe(inputs, outputs)
    #output_trainingset(trainingset)

    splits = trainingset.split_stratified(10)
    inputs, outputs = zip(*splits) 
    training_in = np.concatenate(inputs[0:8])
    training_out = np.concatenate(outputs[0:8])
    validation_in = inputs[8]
    validation_out = outputs[8]
    test_in = inputs[9]
    test_out = outputs[9]
    
    model = Sequential([
        #Noise Layers? GaussianNoise, GaussianDropout
        Dense(50, 
            #init       = 'uniform', 
            activation = 'sigmoid',
            W_regularizer=l2(0.000),
            input_dim  = training_in.shape[1]),
        Dropout(0.1),
        Dense(40,
            #init       = 'normal',
            activation = 'sigmoid',
            W_regularizer=l2(0.000)), 
        Dropout(0.1), 
        Dense(30, 
            activation = 'sigmoid', 
            W_regularizer=l2(0.00)),
        Dropout(0.10),
        Dense(training_out.shape[1], 
            #init       = 'uniform', 
            activation = 'sigmoid',
            #Dont use regularizer
        ),
    ])

    sgd = SGD(lr = 5e-1, momentum=1e-3, decay=0.00001, nesterov=True)
    adam = Adam(lr = 10e-4, decay = 0.000)
    model.compile(
        loss      = 'categorical_crossentropy',
        optimizer = sgd,
        #metrics   = ['accuracy']
    )

    model.fit(training_in, training_out, 
        batch_size      = 128,
        nb_epoch        = 10000,
        verbose         = 1,
        validation_data = (validation_in, validation_out))
    run_predict(model, test_in, test_out)

main()
