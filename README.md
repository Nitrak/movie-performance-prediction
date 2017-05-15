# movie-performance-prediction
Part of the code of the bachelor project of Per Josefsen and Michael Vesterli for using deep neural networks to predict the performance of movies in cinemas. Includes the code for running and evaluating models. Note that since not all of our data can be publicized, the models cannot be run.

Most of our code was written for one time use only, however the following scripts may still be useful. All are written using python3

## Data gathering scripts
`trends/update_aggregation.py`
To get Google Trends values we have two datasets that collect the data points returned by google trends. trend\_single\_aggregation collects a list of values without comparing them, and trend\_cmp\_aggregation collects a list with comparisons. The list is stored in order to find the mean. This script updates every item in the provided dataset until a max value by querying google trend and scraping its graph. However it requires that an existing dataset is present.

`trends/create\_dataset.py`
This script uses the files created by `trends/update_aggregation` to create absolute trend values in a form that can be used by the models.

`get_weather_data.py`
This script scrapes generated images from dmi.dk to find weather data for each day from 2004 to 2013.

`fill_relative_weather_data.py`
This script creates a dataset of the weather on each day relative to the premiere of each movie.

`get_dates_data`
This script gets info about each date in the input dataset.

`fill_relative_date_data.py`
This script creates the dataset over the premiere data relative to the premiere of each movie.

`create_competition_data.py`
This script uses the budgets and premiere dates of movies to find the main competitors of each movie.

`fill_crewdata`
This script calculates the previous success of actors and directors for each movie and inserts it into the dataset.

## Models
`day0.py`
This script creates the specialized model, using data from the day given in the `TARGET_DAY` variable.

`day_diff`
This script creates the general model, which is trained on data from days between `STARTDAY` and `ENDDAY`.

`sharda_paper_model`
This script creates the model used by Sharda and Delen (2006).

`trainingset.py` 
This module allows describing a dataset using different kinds of attributes across multiple files. It also contains functions for manipulating the dataset.

`prediction.py` 
This module contains various means of evaluating the result of the models.
