# Creates dataset of the weather around premiere

import csv
import datetime
import matplotlib.pyplot as plt

DATA_FILE = 'adjusted_dates.csv'
WEATHER_FILE = 'mining_ready/2004weather.csv'
OUTPUT_FILE = 'mining_ready/2004rel_weather.csv'

DAYS_AHEAD = 14

# date (yyyy-mm-dd) -> weather
weather = {}

with open(WEATHER_FILE, encoding='utf-8') as input_file:
    reader = csv.DictReader(input_file, delimiter=';')
    for row in reader:
        weather[row['date']] = row

with open(DATA_FILE, encoding='utf-8') as input_file, \
     open(OUTPUT_FILE, 'w+', encoding='utf-8') as output_file:
    reader = csv.DictReader(input_file, delimiter=';')
    fields = ['tmdbid']
    for i in range(DAYS_AHEAD+1): 
        fields.append('weather_rain+' + str(i))
        fields.append('weather_sun+' + str(i))
        fields.append('weather_lowtemp+' + str(i))
        fields.append('weather_hightemp+' + str(i))
    writer = csv.DictWriter(output_file, 
        fieldnames = fields, delimiter=';', quotechar='"')
    writer.writeheader()

    for row in reader:
        output = {'tmdbid': row['tmdbid']}
        release = datetime.datetime.strptime(row['release'], '%Y-%m-%d')
        for day in range(DAYS_AHEAD+1):
            date = release + datetime.timedelta(days=day)
            datestr = '{}-{}-{}'.format(date.year, date.month, date.day)
            output['weather_rain+'+str(day)] = weather[datestr]['rain']
            output['weather_sun+'+str(day)] = weather[datestr]['sun']
            output['weather_lowtemp+'+str(day)] = weather[datestr]['min_temp']
            output['weather_hightemp+'+str(day)] = weather[datestr]['max_temp']
        writer.writerow(output)

