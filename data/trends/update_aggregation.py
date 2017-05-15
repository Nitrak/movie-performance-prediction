import csv
import time
import re
import datetime
import dryscrape
from bs4 import BeautifulSoup
import ast

SINGLE_FILE = 'trend_single_aggregation.csv'
CMP_FILE = 'trend_cmp_aggregation.csv'
DELAY = 12
# Max number of values to aggregate
MAX_VALUES = 3
# The file to update, 'single' | 'cmp'
FILE = 'cmp'


def get_written(filename):
    trends = []
    with open(filename, encoding='utf-8') as input_file:
        reader = csv.DictReader(input_file, delimiter=';', quotechar='"')
        for row in reader:
            values = {}
            for key in row:
                value = row[key]
                if value:
                    if key == 'trends': value = ast.literal_eval(value)
                    if key == 'start' or key == 'end':
                        if FILE == 'cmp':
                            value = datetime.datetime.strptime(
                                value, '%Y-%m-%d')
                        else:
                            value = datetime.datetime.strptime(value, '%Y-%m-%d')
                    values[key] = value
            trends.append(values)
    return trends;

PARSE_RE = re.compile('\[<path\s+d="M(.*?)".*')
# Returns a map of dates to trends from the svg path.
def parse(startdate, path):
    MINY = 17.5
    MAXY = 202.5
    def split_comma(pair):
        idx, string = pair
        l = string.split(',')
        return ((startdate+datetime.timedelta(days=idx)).strftime('%Y-%m-%d'), 
                round(100-(float(l[1])-MINY)/(MAXY-MINY)*100))
    r = PARSE_RE.match(str(path))
    res = {}
    for day, trend in map(split_comma, enumerate(r.group(1).split('L'))):
        res[day] = trend
    return res 

def visit(session, query, start, end):
    url = 'https://trends.google.com/trends/explore?geo=DK&date={} {}&q={}'.format(
        start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), query)
    session.visit(url)
    time.sleep(DELAY)
    response = session.body()
    return BeautifulSoup(response, 'lxml')

def request_comparison(session, id_a, id_b, start, end):
    response = visit(session, '{},{}'.format(id_a, id_b), start, end)
    svg_a = response.select('path[stroke="#4285f4"]')
    svg_b = response.select('path[stroke="#db4437"]')
    if svg_a:
        id_a = parse(start, svg_a)
        id_b = parse(start, svg_b)
        return (id_a, id_b)
    else:
        return ({}, {})

def request_single(session, googleid, start, end):
    response = visit(session, googleid, start, end)
    svg = response.select('path[stroke="#4285f4"]')
    if svg:
        return parse(start, svg)
    else:
        return {}

def main():
    session = dryscrape.Session()
    session.set_timeout(10)

    if FILE == 'single':
        trends = get_written(SINGLE_FILE)
        with open(SINGLE_FILE, 'w+', encoding='utf-8') as output_file:
            fields = ['googleid', 'start', 'end', 'trends']
            writer = csv.DictWriter(output_file, delimiter=';', quotechar='"',
                    fieldnames=fields)
            writer.writeheader()
   
            count = 0
            for row in trends:
                count += 1
                print('Reading row number {}'.format(count))
                #try:
                if len(row['trends']) < MAX_VALUES:
                    plot = request_single(session, 
                        row['googleid'], row['start'], row['end'])
                    row['trends'].append(plot)
                    writer.writerow({
                        'googleid': row['googleid'],
                        'start': row['start'].strftime('%Y-%m-%d'),
                        'end': row['end'].strftime('%Y-%m-%d'),
                        'trends': str(row['trends']),
                    })
                #except KeyboardInterrupt:
                #    raise KeyboardInterrupt
                #except:
                #    print('Fail')
    else:
        trends = get_written(CMP_FILE)
        with open(CMP_FILE, 'w+', encoding='utf-8') as output_file:
            fields = ['id_a', 'id_b', 'start', 'end', 'trends']
            writer = csv.DictWriter(output_file, delimiter=';', quotechar='"',
                    fieldnames=fields)
            writer.writeheader()

            count = 0
            for row in trends:
                count += 1
                print('Reading row number {}'.format(count))
                try:
                    if len(row['trends']) < MAX_VALUES:
                        pair = request_comparison(session, row['id_a'], row['id_b'],
                            row['start'], row['end'])
                        row['trends'].append(pair)
                        writer.writerow({
                            'id_a': row['id_a'],
                            'id_b': row['id_b'],
                            'start': row['start'].strftime('%Y-%m-%d'),
                            'end': row['end'].strftime('%Y-%m-%d'),
                            'trends': str(row['trends']),
                        })
                except:
                    print('Fail')
main()
