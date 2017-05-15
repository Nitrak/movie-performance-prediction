# Fetches weather data from dmi.dk by parsing dmi images and creates a new dataset.

import csv
import urllib.request
from PIL import Image
import io

OUTPUT_FILE = 'mining_ready/2004weather.csv'
FIELDS = ['date', 'max_temp', 'min_temp', 'sun', 'rain']

date_info = {}

temp_bounds = {
    (2004, 1): (-15, 10),
    (2004, 2): (-10, 15),
    (2004, 3): (-10, 20),
    (2004, 4): (-5, 20),
    (2004, 5): (0, 25),
    (2004, 6): (0, 25),
    (2004, 7): (0, 30),
    (2004, 8): (0, 30),
    (2004, 9): (0, 30),
    (2004, 10): (-5, 20),
    (2004, 11): (-10, 15),        
    (2004, 12): (-5, 10),        
    (2005, 1): (-15, 15),
    (2005, 2): (-15, 10),
    (2005, 3): (-25, 15),
    (2005, 4): (-10, 25),
    (2005, 5): (-5, 30),
    (2005, 6): (0, 30),
    (2005, 7): (0, 35),
    (2005, 8): (0, 30),
    (2005, 9): (0, 30),
    (2005, 10): (-5, 20),
    (2005, 11): (-10, 20),        
    (2005, 12): (-10, 15),        
    (2006, 1): (-20, 10),
    (2006, 2): (-15, 10),
    (2006, 3): (-20, 15),
    (2006, 4): (-5, 20),
    (2006, 5): (0, 25),
    (2006, 6): (0, 35),
    (2006, 7): (0, 35),
    (2006, 8): (0, 30),
    (2006, 9): (0, 30),
    (2006, 10): (0, 25),
    (2006, 11): (-10, 15),        
    (2006, 12): (-5, 15),        
    (2007, 1): (-15, 15),
    (2007, 2): (-10, 10),
    (2007, 3): (-5, 20),
    (2007, 4): (-5, 25),
    (2007, 5): (-5, 30),
    (2007, 6): (0, 35),
    (2007, 7): (0, 30),
    (2007, 8): (0, 30),
    (2007, 9): (0, 25),
    (2007, 10): (-5, 20),
    (2007, 11): (-5, 15),        
    (2007, 12): (-5, 15),        
    (2008, 1): (-5, 15),
    (2008, 2): (-10, 15),
    (2008, 3): (-10, 20),
    (2008, 4): (-5, 25),
    (2008, 5): (-5, 30),
    (2008, 6): (0, 30),
    (2008, 7): (0, 35),
    (2008, 8): (0, 35),
    (2008, 9): (0, 25),
    (2008, 10): (-5, 20),
    (2008, 11): (-10, 15),        
    (2008, 12): (-10, 10),        
    (2009, 1): (-15, 10),
    (2009, 2): (-15, 10),
    (2009, 3): (-10, 15),
    (2009, 4): (-5, 25),
    (2009, 5): (-5, 30),
    (2009, 6): (0, 30),
    (2009, 7): (0, 35),
    (2009, 8): (0, 30),
    (2009, 9): (0, 30),
    (2009, 10): (-5, 20),
    (2009, 11): (0, 15),        
    (2009, 12): (-15, 10),        
    (2010, 1): (-20, 10),
    (2010, 2): (-15, 10),
    (2010, 3): (-15, 20),
    (2010, 4): (-5, 25),
    (2010, 5): (-5, 25),
    (2010, 6): (0, 30),
    (2010, 7): (0, 35),
    (2010, 8): (0, 30),
    (2010, 9): (0, 25),
    (2010, 10): (-5, 20),
    (2010, 11): (-10, 15),        
    (2010, 12): (-25, 10),        
    (2011, 1): (-15, 10),
    (2011, 2): (-15, 15),
    (2011, 3): (-10, 20),
    (2011, 4): (-5, 25),
    (2011, 5): (-5, 30),
    (2011, 6): (0, 30),
    (2011, 7): (0, 30),
    (2011, 8): (0, 30),
    (2011, 9): (0, 30),
    (2011, 10): (-5, 25),
    (2011, 11): (-5, 15),        
    (2011, 12): (-5, 15),        
    (2012, 1): (-10, 10),
    (2012, 2): (-20, 15),
    (2012, 3): (-5, 20),
    (2012, 4): (-10, 25),
    (2012, 5): (-5, 30),
    (2012, 6): (0, 30),
    (2012, 7): (0, 35),
    (2012, 8): (0, 35),
    (2012, 9): (0, 30),
    (2012, 10): (-5, 25),
    (2012, 11): (-5, 15),        
    (2012, 12): (-15, 10),        
    (2013, 1): (-20, 15),
    (2013, 2): (-15, 10),
    (2013, 3): (-15, 15),
    (2013, 4): (-10, 25),
    (2013, 5): (-5, 30),
    (2013, 6): (0, 30),
    (2013, 7): (0, 35),
    (2013, 8): (0, 30),
    (2013, 9): (0, 30),
    (2013, 10): (-5, 20),
    (2013, 11): (-10, 15),        
    (2013, 12): (-10, 15),        
}

precip_bounds = {
    (2004, 1): 20,
    (2004, 2): 10,
    (2004, 3): 15,
    (2004, 4): 15,
    (2004, 5): 10,
    (2004, 6): 15,
    (2004, 7): 25,
    (2004, 8): 15,
    (2004, 9): 10,
    (2004, 10):25,
    (2004, 11):15,        
    (2004, 12):20,        
    (2005, 1): 15,
    (2005, 2): 25,
    (2005, 3): 10,
    (2005, 4): 10,
    (2005, 5): 10,
    (2005, 6): 25,
    (2005, 7): 20,
    (2005, 8): 15,
    (2005, 9): 10,
    (2005, 10):20,
    (2005, 11):10,        
    (2005, 12):15,        
    (2006, 1): 10,
    (2006, 2): 20,
    (2006, 3): 10,
    (2006, 4): 10,
    (2006, 5): 15,
    (2006, 6): 15,
    (2006, 7): 15,
    (2006, 8): 25,
    (2006, 9): 10,
    (2006, 10):15,
    (2006, 11):15,        
    (2006, 12):20,        
    (2007, 1): 20,
    (2007, 2): 20,
    (2007, 3): 10,
    (2007, 4): 10,
    (2007, 5): 40,
    (2007, 6): 40,
    (2007, 7): 45,
    (2007, 8): 15,
    (2007, 9): 20,
    (2007, 10):15,
    (2007, 11):10,        
    (2007, 12):10,        
    (2008, 1): 10,
    (2008, 2): 10,
    (2008, 3): 10,
    (2008, 4): 10,
    (2008, 5): 20,
    (2008, 6): 10,
    (2008, 7): 15,
    (2008, 8): 30,
    (2008, 9): 15,
    (2008, 10):15,
    (2008, 11):15,        
    (2008, 12):20,        
    (2009, 1): 10,
    (2009, 2): 10,
    (2009, 3): 10,
    (2009, 4): 10,
    (2009, 5): 10,
    (2009, 6): 80,
    (2009, 7): 15,
    (2009, 8): 10,
    (2009, 9): 10,
    (2009, 10):20,
    (2009, 11):15,        
    (2009, 12):15,        
    (2010, 1): 10,
    (2010, 2): 10,
    (2010, 3): 15,
    (2010, 4): 15,
    (2010, 5): 25,
    (2010, 6): 30,
    (2010, 7): 20,
    (2010, 8): 50,
    (2010, 9): 15,
    (2010, 10):15,
    (2010, 11):15,        
    (2010, 12):10,        
    (2011, 1): 10,
    (2011, 2): 15,
    (2011, 3): 10,
    (2011, 4): 15,
    (2011, 5): 15,
    (2011, 6): 15,
    (2011, 7): 35,
    (2011, 8): 35,
    (2011, 9): 15,
    (2011, 10):15,
    (2011, 11):10,        
    (2011, 12):10,        
    (2012, 1): 20,
    (2012, 2): 10,
    (2012, 3): 10,
    (2012, 4): 15,
    (2012, 5): 10,
    (2012, 6): 25,
    (2012, 7): 15,
    (2012, 8): 15,
    (2012, 9): 20,
    (2012, 10):20,
    (2012, 11):15,        
    (2012, 12):10,        
    (2013, 1): 15,
    (2013, 2): 10,
    (2013, 3): 10,
    (2013, 4): 10,
    (2013, 5): 20,
    (2013, 6): 20,
    (2013, 7): 10,
    (2013, 8): 10,
    (2013, 9): 10,
    (2013, 10):20,
    (2013, 11):10,        
    (2013, 12):10,        
}

sun_bounds = {
    (2004, 1): 10,
    (2004, 2): 10,
    (2004, 3): 15,
    (2004, 4): 15,
    (2004, 5): 15,
    (2004, 6): 20,
    (2004, 7): 20,
    (2004, 8): 15,
    (2004, 9): 15,
    (2004, 10):10,
    (2004, 11):10,        
    (2004, 12):10,        
    (2005, 1): 10,
    (2005, 2): 10,
    (2005, 3): 15,
    (2005, 4): 15,
    (2005, 5): 15,
    (2005, 6): 20,
    (2005, 7): 20,
    (2005, 8): 15,
    (2005, 9): 15,
    (2005, 10):10,
    (2005, 11):10,        
    (2005, 12):10,        
    (2006, 1): 10,
    (2006, 2): 10,
    (2006, 3): 15,
    (2006, 4): 15,
    (2006, 5): 20,
    (2006, 6): 20,
    (2006, 7): 20,
    (2006, 8): 15,
    (2006, 9): 15,
    (2006, 10):10,
    (2006, 11):10,        
    (2006, 12):10,        
    (2007, 1): 10,
    (2007, 2): 10,
    (2007, 3): 15,
    (2007, 4): 15,
    (2007, 5): 20,
    (2007, 6): 20,
    (2007, 7): 15,
    (2007, 8): 15,
    (2007, 9): 15,
    (2007, 10):10,
    (2007, 11):10,        
    (2007, 12):10,        
    (2008, 1): 10,
    (2008, 2): 10,
    (2008, 3): 15,
    (2008, 4): 15,
    (2008, 5): 20,
    (2008, 6): 20,
    (2008, 7): 20,
    (2008, 8): 15,
    (2008, 9): 15,
    (2008, 10):10,
    (2008, 11):10,        
    (2008, 12):10,        
    (2009, 1): 10,
    (2009, 2): 10,
    (2009, 3): 15,
    (2009, 4): 15,
    (2009, 5): 20,
    (2009, 6): 20,
    (2009, 7): 15,
    (2009, 8): 15,
    (2009, 9): 15,
    (2009, 10):10,
    (2009, 11):10,        
    (2009, 12):10,        
    (2010, 1): 10,
    (2010, 2): 10,
    (2010, 3): 15,
    (2010, 4): 15,
    (2010, 5): 15,
    (2010, 6): 20,
    (2010, 7): 20,
    (2010, 8): 15,
    (2010, 9): 15,
    (2010, 10):15,
    (2010, 11):10,        
    (2010, 12):10,        
    (2011, 1): 10,
    (2011, 2): 10,
    (2011, 3): 15,
    (2011, 4): 15,
    (2011, 5): 20,
    (2011, 6): 20,
    (2011, 7): 20,
    (2011, 8): 15,
    (2011, 9): 15,
    (2011, 10):15,
    (2011, 11):10,        
    (2011, 12):10,        
    (2012, 1): 10,
    (2012, 2): 10,
    (2012, 3): 15,
    (2012, 4): 15,
    (2012, 5): 20,
    (2012, 6): 20,
    (2012, 7): 20,
    (2012, 8): 15,
    (2012, 9): 15,
    (2012, 10):10,
    (2012, 11):10,        
    (2012, 12):10,        
    (2013, 1): 10,
    (2013, 2): 10,
    (2013, 3): 15,
    (2013, 4): 15,
    (2013, 5): 15,
    (2013, 6): 20,
    (2013, 7): 20,
    (2013, 8): 15,
    (2013, 9): 15,
    (2013, 10):10,
    (2013, 11):10,        
    (2013, 12):10,        
}

def average(xy):
    x, y = xy
    return (x+y)/2

# Datatype can be temp(Temperature)
def get_image(year, month, data_type):
    url = 'http://servlet.dmi.dk/vejrarkiv/servlet/vejrarkiv?region=7&year={}&month={}&country=dk&parameter={}'.format(year, month, data_type)
    return Image.open(io.BytesIO(urllib.request.urlopen(url).read()))

# Takes the image, the minimum and maximum labels and 
# a function to select the resulting value from an entire found line.
def process_image(img, minval, maxval, selector):
    # Image size
    width = 640
    height = 360
    
    # Edges of graph
    top_edge = 95
    bottom_edge = 315
    left_edge = 10
    right_edge = 599

    num_columns = 31
    col_width = (right_edge-left_edge)/num_columns

    def is_color(rgb):
        r, g, b = rgb
        rg, rb, gb = abs(r-g), abs(r-b), abs(g-b)
        return rg > 30 or rb > 30 or gb > 30
    def normalize(pair):
        def _normalize(val):
            return minval+(bottom_edge-val)*(maxval-minval)/(bottom_edge-top_edge)
        mi, ma = pair
        return (_normalize(mi), _normalize(ma))

    def values(x):
        res = []
        first = None
        for y in range(top_edge-1, bottom_edge+1):
            if is_color(img.getpixel((x, y))):
                if first == None: first = y
            elif first != None:
                res.append((first, y-1))
                first = None
        return res

    colx = left_edge-col_width/2
    res = []
    for day in range(num_columns):
        colx += col_width
        res.append(list(map(selector, map(normalize, values(colx)))))
    return res

def get_temperature(year, month, min_temp, max_temp):
    def create_mapping(temps):
        if len(temps) == 2: 
            return {'max_temp': temps[0], 'min_temp': temps[1]}
        return {}
    return map(create_mapping,
        process_image(get_image(year, month, 'temp'), min_temp, max_temp, average))

def get_rain(year, month, bound):
    def create_mapping(rain):
        if len(rain) == 1: return {'rain': rain[0]}
        return {'rain': 0}
    return map(create_mapping,
        process_image(get_image(year, month, 'precip'), 0, bound, max))

def get_sun(year, month, bound):
    def create_mapping(sun):
        if len(sun) == 1: return {'sun': sun[0]}
        else: return {'sun': 0}
    return map(create_mapping,
        process_image(get_image(year, month, 'sun'), 0, bound, max))

for time, bounds in temp_bounds.items():
    year, month = time
    mi, ma = bounds
    for day, temp in enumerate(get_temperature(year, month, mi, ma)):
        date = '{}-{}-{}'.format(year, month, day+1)
        if date not in date_info: date_info[date] = {}
        date_info[date].update(temp)

for time, bound in precip_bounds.items():
    year, month = time
    for day, rain in enumerate(get_rain(year, month, bound)):
        date = '{}-{}-{}'.format(year, month, day+1)
        date_info[date].update(rain)

for time, bound in sun_bounds.items():
    year, month = time
    for day, sun in enumerate(get_sun(year, month, bound)):
        date = '{}-{}-{}'.format(year, month, day+1)
        date_info[date].update(sun)

with open(OUTPUT_FILE, 'w+', encoding='utf-8') as output_file:
    writer=csv.DictWriter(output_file, fieldnames=FIELDS, delimiter=';',
            quotechar='"')
    writer.writeheader()
    for date, weather in date_info.items():
        row = {'date': date}
        row.update(weather)
        writer.writerow(row)
