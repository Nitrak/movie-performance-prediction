import csv
import datetime
import matplotlib.pyplot as plt

DATA_FILE = 'adjusted_dates.csv'
DATE_FILE = 'mining_ready/2004dates_data.csv'
OUTPUT_FILE = 'mining_ready/2004rel_dates.csv'

DAYS_AHEAD = 14

# date (yyyy-mm-dd) -> dateinfo
dates = {}

with open(DATE_FILE, encoding='utf-8') as input_file:
    reader = csv.DictReader(input_file, delimiter=';')
    for row in reader:
        dates[row['date']] = row

with open(DATA_FILE, encoding='utf-8') as input_file, \
     open(OUTPUT_FILE, 'w+', encoding='utf-8') as output_file:
    reader = csv.DictReader(input_file, delimiter=';')
    fields = ['tmdbid']
    for i in range(DAYS_AHEAD+1): 
        fields.append('year+' + str(i))
        fields.append('weeknum+' + str(i))
        fields.append('day_in_week+' + str(i))
        fields.append('holiday+' + str(i))
        fields.append('holiday_effect+' + str(i))
    writer = csv.DictWriter(output_file, 
        fieldnames = fields, delimiter=';', quotechar='"')
    writer.writeheader()

    for row in reader:
        output = {'tmdbid': row['tmdbid']}
        release = datetime.datetime.strptime(row['release'], '%Y-%m-%d')
        for day in range(DAYS_AHEAD+1):
            date = release + datetime.timedelta(days=day)
            datestr = '{}-{}-{}'.format(date.year, date.month, date.day)
            output['year+'+str(day)] = dates[datestr]['year']
            output['weeknum+'+str(day)] = dates[datestr]['weeknum']
            output['day_in_week+'+str(day)] = dates[datestr]['day_in_week']
            output['holiday+'+str(day)] = dates[datestr]['holiday']
            output['holiday_effect+'+str(day)] = dates[datestr]['holiday_effect']
        writer.writerow(output)

