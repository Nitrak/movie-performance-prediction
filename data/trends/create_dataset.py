import csv
import ast
import datetime
import math

RELEASE_FILE = '../adjusted_dates.csv'
SINGLE_FILE = 'trend_single_aggregation.csv'
COMP_FILE = 'trend_cmp_aggregation.csv'
OUTPUT_FILE = '2004_trends.csv'
ID_MAPPING_FILE = 'googleids.csv'

STARTSCALE = 100
# Days before and after the release date
POSITIVE_DAYS = 14
NEGATIVE_DAYS = 28

def mean(values):
    return float(sum(values))/max(len(values), 1)

def average_cmp(l):
    agg = ({}, {})
    for trends_a, trends_b in l:
        for date, value in trends_a.items():
            if date not in agg[0]: agg[0][date] = []
            agg[0][date].append(value)
        for date, value in trends_b.items():
            if date not in agg[1]: agg[1][date] = []
            agg[1][date].append(value)
    res = ({}, {})
    for date, values in agg[0].items():
        res[0][date] = mean(values)
    for date, values in agg[1].items():
        res[1][date] = mean(values)
    return res

def average_single(l):
    agg = {}
    for trends in l:
        for date, value in trends.items():
            if date not in agg: agg[date] = []
            agg[date].append(value)
    res = {}
    for date, values in agg.items():
        res[date] = mean(values)
    return res

def get_releases():
    res = {}
    with open(RELEASE_FILE, encoding='utf-8') as input_file:
        reader = csv.DictReader(input_file, delimiter=';', quotechar='"')
        for row in reader:
            res[row['tmdbid']] = datetime.datetime.strptime(
                row['release'], '%Y-%m-%d')
    return res

def get_mappings():
    res = {}
    with open(ID_MAPPING_FILE, encoding='utf-8') as input_file:
        reader = csv.DictReader(input_file, delimiter=';', quotechar='"')
        for row in reader:
            res[row['tmdbid']] = row['googleid']
    return res

def get_cmp_trends():
    res = [] 
    with open(COMP_FILE, encoding='utf-8') as input_file:
        reader = csv.DictReader(input_file, delimiter=';', quotechar='"')
        for row in reader:
            row['trends'] = average_cmp(ast.literal_eval(row['trends']))
            res.append(row)
    return res

# Returns (googleid -> trends), scales
def get_single_trends():
    res = {}
    scales = {}
    with open(SINGLE_FILE, encoding='utf-8') as input_file:
        rows = []
        reader = csv.DictReader(input_file, delimiter=';', quotechar='"')
        for row in reader:
            rows.append(row)
            scales[row['googleid']] = 0
            res[row['googleid']] = average_single(ast.literal_eval(row['trends']))
        scales[rows[0]['googleid']] = STARTSCALE
    return (res, scales) 

def fill_scales(scales, cmp_trends):
    already_read = set()
    for row in cmp_trends:
        googleid = row['id_a']
        target = row['id_b']
        if googleid not in already_read and target in scales:
            already_read.add(googleid)
            trend_a, trend_b = row['trends']
            if trend_a:
                max_a = max(trend_a.values())
            else:
                max_a = 0
            if trend_b:
                max_b = max(trend_b.values())
            else:
                max_b = 0

            if max_a < max_b:
                scale = max_a/100*scales[target]
            else:
                if max_b == 0:
                    scale = math.inf
                else:
                    scale = 100/max_b*scales[target]
            if not math.isinf(scale):
                scales[googleid] = scale
            #print('{} ~~ {} in {}-{} : {}'.format(
            #    googleid, target, row['start'], row['end'], scale/scales[target]))

def write_scales(scales):
    with open(OUTPUT_FILE, 'w+', encoding='utf-8') as output_file:
        writer = csv.DictWriter(output_file, delimiter=';', quotechar='"',
                fieldnames=['googleid', 'scale'])
        writer.writeheader()
        for googleid, scale in scales.items():
            writer.writerow({
                'googleid': googleid,
                'scale': scale,
            })

def write_datafile(scales, single_trends, id_mappings, releases):
    with open(OUTPUT_FILE, 'w+', encoding='utf-8') as output_file:
        fields = ['tmdbid', 'maxtrend', 'avg_trend']
        for day in range(-NEGATIVE_DAYS, POSITIVE_DAYS):
            fields.append('day' + str(day))
            fields.append('delta' + str(day))
        writer = csv.DictWriter(output_file, delimiter=';', quotechar='"',
                fieldnames=fields)
        writer.writeheader()
        
        for tmdbid, release in releases.items():
            row = { 'tmdbid': tmdbid }
            found = False
            if tmdbid in id_mappings:
                googleid = id_mappings[tmdbid]
                if googleid in single_trends and googleid in scales:
                    found = True
                    scaledplot = {}
                    for key, value in single_trends[googleid].items():
                        scaledplot[key] = value*scales[googleid]
                    row['maxtrend'] = max(scaledplot.values())
                    row['avg_trend'] = mean(scaledplot.values())
                    for day in range(-NEGATIVE_DAYS, POSITIVE_DAYS):
                        date = (release+datetime.timedelta(day)).strftime('%Y-%m-%d')
                        prevdate = (release+datetime.timedelta(day-1)).strftime('%Y-%m-%d')
                        day_key = 'day'+str(day)
                        delta_key = 'delta'+str(day)
                        if date in scaledplot:
                            row[day_key] = scaledplot[date]
                            if prevdate in scaledplot:
                                row[delta_key] = scaledplot[date]-scaledplot[prevdate]
                            else:
                                row[delta_key] = '0'
                        else:
                            row[day_key] = row[delta_key] = '0'
            if not found:
                row['maxtrend'] = 0
                row['avg_trend'] = 0
                for day in range(-NEGATIVE_DAYS, POSITIVE_DAYS):
                    row['day'+str(day)] = row['delta'+str(day)] = '0'

            for day in range(-NEGATIVE_DAYS, POSITIVE_DAYS):
                date = (release+datetime.timedelta(day)).strftime('%Y-%m-%d')
            writer.writerow(row)


def main():
    # tmdbid -> releasedate
    releases = get_releases()
    # tmdbid -> googleid
    id_mappings = get_mappings()
    cmp_trends = get_cmp_trends()
    single_trends, scales = get_single_trends()
    fill_scales(scales, cmp_trends)
    #write_scales(scales)
    write_datafile(scales, single_trends, id_mappings, releases)

main()
