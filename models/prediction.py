import numpy as np
import matplotlib.pyplot as plt
import math
import csv

# Outputs the precentage of exact predictions and one-off predictions.
# Stands for acerage percent hit rate
# actual: training output and is a one hot array.
# prediction is the predicted class.
def APHR(actual, prediction):
    distances = [0]*9
    for i, pred in enumerate(prediction):
        correct_idx = np.argmax(actual[i])
        pred_idx = pred
        distances[abs(pred_idx-correct_idx)] += 1
    print('\nAPHR:')
    totalhits = 0 
    for dist, hits in enumerate(distances):
        totalhits += hits
        print('{}: {:.2f}% correct'.format(dist, totalhits/len(actual)*100))

# Writes the confusion matrix to the filename
# Takes the actual and prediction in similar fashion to APHR.
def output_confusion_matrix(filename, actual, prediction):
    buckets = len(actual[0])
    matrix = [[0]*buckets for i in range(buckets)]
    for i, pred in enumerate(prediction):
        correct_idx = np.argmax(actual[i])
        pred_idx = pred 
        matrix[pred_idx][correct_idx] += 1
    with open(filename, 'w+', encoding='utf-8', newline='\n') as outputfile:
        writer = csv.writer(outputfile, delimiter=';', quotechar = '"')
        for row in matrix:
            writer.writerow(row)

# Evaluates the model
def run_classification_predict(model, test_in, test_out, bucket):
    score = model.evaluate(test_in, test_out)
    print('\nTest score: ', score)
    
    predict = model.predict_classes(test_in, batch_size=32, verbose=1)
    APHR(test_out, predict)
    output_confusion_matrix('output/confusion_matrix{}.csv'.format(bucket), test_out, predict)

def run_numeric_predict(model, test_in, test_out):
    score = model.evaluate(test_in, test_out)
    print('\nTest score: ', score)
   
    predict = model.predict(test_in, batch_size=128, verbose=1)
    graph = list(zip(map(lambda x: x[0], test_out), map(lambda x: x[0], predict)))
    input, output = zip(*sorted(graph))
    plt.plot(input, 'r.')
    plt.plot(output, 'b.')
    plt.show()

# Profiles the sensitivity of the models to its inputs
# model: the model to profile
# data: the data the model has been trained with as a numpy array
# input_indices: the inputs to profile, None means all
# steps: number of datapoints to each curve
# splits: the quartiles to hold data constant at
# reverse_func: a function to apply to the predicted result before using. Usually to reverse normalization.
def profile_network(model, data, 
        headers = None, input_indices = None, steps = 100, splits = [0, 0.33, 0.66, 1], 
        reverse_func = lambda x: x):
    def get_quantile_value(sorted_list, quantile):
        idx = int(quantile*len(sorted_list))
        if idx == len(sorted_list): idx -= 1;
        return sorted_list[idx]
   
    num_rows, num_indices = data.shape
    if not input_indices: input_indices = list(range(num_indices))
    if not headers: headers = list(map(lambda idx: 'Input {}'.format(idx), input_indices))
    sorted_attr_data = list(map(lambda idx: list(sorted(map(lambda x: x[idx], data))), 
                                range(num_indices)))
 
    with open('profileoutput.tex', 'w+', encoding='utf-8') as outputfile:
        plot_sides = 3
        # subplots
        fig, axarr = plt.subplots(plot_sides, plot_sides, sharex=True, sharey=True)
        for plot_idx, input_idx in enumerate(input_indices):
            #print('Plotting {} of {}'.format(plot_idx, len(input_indices)))
            outputfile.write(headers[input_idx])
            plotx = plot_idx%plot_sides
            ploty = int(plot_idx/plot_sides)%3
            axarr[ploty, plotx].set_title(headers[input_idx])

            # curves
            for split in splits:
                # datapoints
                test_data = np.array([list(map(lambda idx: get_quantile_value(sorted_attr_data[idx], split), 
                                               range(num_indices)))])
                curve = []
                for point in range(steps+1):
                    quantile = point/steps
                    test_data[0][input_idx] = get_quantile_value(sorted_attr_data[input_idx], quantile)
                    output = reverse_func(model.predict(test_data)[0][0])
                    curve.append(output)
                axarr[ploty, plotx].plot(curve, color=(split, 0, 1-split))
                outputfile.write(' & {} & {}'.format(round(curve[0]), round(curve[len(curve)-1])))
            #    print('Split {} = start: {}, end: {}, increase: {}'.format(
            #        split, curve[0], curve[len(curve)-1], curve[len(curve)-1]-curve[0]))
            outputfile.write(' \\\\\n')
            if plot_idx%9 == 8: 
                plt.show()
                fig, axarr = plt.subplots(plot_sides, plot_sides, sharex=True, sharey=True)
        if len(input_indices)%9 != 0: plt.show()



# Output both the input andthe output dataset.
def output_trainingset(trainingset):
    trainingset.input.writecsv('output/training_input.csv')
    trainingset.output.writecsv('output/training_output.csv')

