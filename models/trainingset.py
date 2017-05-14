import copy
import numpy as np
import math
import csv
import itertools
import datetime
import re
import random

# This module uses descriptors to create datasets.  
# Each descriptor contains a "headers()" function, which returns an iterable 
# of headers, each describing the data in a single column.
# It also contains a "data" field, which is a list of dicts that it reads its data from
# It must not be the same list as another descriptor.

# The key for uniquely identifying movies. All read datasets must contain this column.
#UNIQUE_KEY = 'showingid'
UNIQUE_KEY = 'tmdbid'

# Parses the number as a float. If it is unparseable, returns nan.
def getnumber(val):
    try:
        v = float(val)
    except:
        v = math.nan
    return v

# Reads a list of the form ['a', 'b'...]
def readlist(string):
    string = string.strip('[]')
    return map(lambda s: s.strip("'"), re.split(',\s*', string))
    
# Reads a csvfile into a list of dicts.
def readfile(filename, encoding, delimiter):
    rows = []
    with open(filename, encoding=encoding) as inputfile:
        reader = csv.DictReader(inputfile, delimiter=delimiter)
        for row in reader:
            rows.append(row)
    return rows

# Returns a function that jitters all inputs in a pair of numpy arrays.
# ranges: a list of tuples (min, max, n), which says that for inputs whose
# output lies in the (min, max) percentile range, n extra inputs should be genereated.
# jitter_percentage, the range that each input can be multiplied with
def jitter(input_data, output_data, ranges, jitter_percentage):
    def jitter_input(row):
        newrow = []
        for v in row:
            jitter = random.uniform(-jitter_percentage, jitter_percentage)
            newrow.append(v*(1+jitter))
        return newrow
    entries = sorted(zip(input_data, output_data), 
            key=lambda v: max(v[1]))
    num_entries = len(entries)

    newinputs = []
    newoutputs = []
    for num_entry, (input_v, output_v) in enumerate(entries):
        n = 1
        for r in ranges:
            if num_entry/num_entries >= r[0] and \
               num_entry/num_entries <= r[1]:
                n = r[2]
        newoutputs.append(output_v)
        newinputs.append(input_v)
        for i in range(n):
            newoutputs.append(output_v)
            newinputs.append(jitter_input(input_v))
    return (np.array(newinputs), np.array(newoutputs))

# Descripes a single column in the resulting dataset.
class Header:
    def __init__(self, name, dataset, attribute, mapping, accepts_row = False):
        # Name of attribute
        self.name      = name  
        # Dataset to read from, which is an iterable of dicts
        self.dataset   = dataset
        # Attribute to read, key in dict
        self.attribute = attribute 
        # Function to parse read attribute.
        # The function does not need to handle missing data.
        self.mapping   = mapping   
        # Whether an attribute should be read or the entire row should be passed
        self.accepts_row = accepts_row

class Dataset:
    def __init__(self, headers, data):
        # list of names of columns for outputs.
        self.headers = headers 
        # A 2d numpy array.
        self.data = data       

    # Creates a dataset from a list of descriptors.
    @classmethod
    def describe(cls, descriptors):
        headers = list(itertools.chain.from_iterable( 
            map(lambda d: d.headers(), descriptors)))
        data = {}
        for (idx, header) in enumerate(headers):
            for row in header.dataset:
                id = row[UNIQUE_KEY]
                if not id in data: data[id] = [math.nan] * len(headers)
                if header.accepts_row:
                    data[id][idx] = header.mapping(row)
                else:
                    if header.attribute in row:
                        value = row[header.attribute]
                        data[id][idx] = header.mapping(value)
                    else:
                        data[id][idx] = header.mapping(0)
        names = list(map(lambda h: h.name, headers))
        data = np.array(list(map(lambda entry: entry[1], sorted(data.items()))))
        return cls(names, data)

    # Writes the dataset to the specified filename in csv format.
    def writecsv(self, filename):
        with open(filename, 'w+', encoding='utf-8', newline='\n') as outputfile:
            writer = csv.writer(outputfile, delimiter = ';', quotechar = '"')
            writer.writerow(self.headers)
            for row in self.data:
                writer.writerow(row)

    def normalize(self):
        for col in range(self.data.shape[1]):
            values = list(map(lambda x: x[col], self.data))
            minval = min(values)
            maxval = max(values)
            for row in range(self.data.shape[0]):
                if maxval-minval == 0:
                    self.data[row][col] = 0.5
                else:  
                    self.data[row][col] = (self.data[row][col]-minval)/(maxval-minval)
    
    def normalize_zscore(self):
        for col in range(self.data.shape[1]):
            values = list(map(lambda x: x[col], self.data))
            mean = sum(values)/len(values)
            variance = sum(map(lambda x: (x-mean)*(x-mean), values))/len(values)
            stddev = math.sqrt(variance)
            for row in range(self.data.shape[0]):
                if stddev == 0:
                    self.data[row][col] = 0
                else:
                    self.data[row][col] = (self.data[row][col]-mean)/stddev

class Trainingset:
    def __init__(self, input, output):
        # Input vectors as Dataset
        self.input = input
        # Expected output vectors as Dataset
        self.output = output

    # Creates a trainingset from a list of input and output descriptors.
    # Ensures that all rows have no missing data in either input or output.
    # Jitterfunc is a function that takes a list of inputs and returns another, 
    # modified list of inputs.
    @classmethod
    def describe(cls, input_descriptors, output_descriptors):
        # Returns all the ids in the dataset that have values present.
        def _all_good_ids(dataset, attribute):
            res = set()
            for row in dataset:
                if attribute not in row:
                    # Non present attributes are defaulted to 0. 
                    # Only attributes that are present and have no value are undefined.
                    res.add(row[UNIQUE_KEY])
                elif row[attribute]:
                    res.add(row[UNIQUE_KEY])
            return res
        # Returns all ids with all values given in both datasets.
        def all_ids(input_headers, output_headers):
            first_header = next(input_headers)
            all_ids = _all_good_ids(first_header.dataset, first_header.attribute)
            for h in itertools.chain(input_headers, output_headers):
                good_ids = _all_good_ids(h.dataset, h.attribute)
                all_ids = set(filter(lambda i: i in good_ids, all_ids))
            return all_ids

        input_headers = itertools.chain.from_iterable(
                map(lambda d: d.headers(), input_descriptors))
        output_headers = itertools.chain.from_iterable(
                map(lambda d: d.headers(), output_descriptors))
        
        all_ids = all_ids(input_headers, output_headers)
        for descriptor in itertools.chain(input_descriptors, output_descriptors):
            descriptor.data = list(filter(
                lambda row: row[UNIQUE_KEY] in all_ids, descriptor.data))

        input  = Dataset.describe(input_descriptors)
        output = Dataset.describe(output_descriptors)
        return cls(input, output)
  
    def split_stratified_buckets(self, n):
        input_data = [[] for _ in range(n)]
        output_data = [[] for _ in range(n)]

        idx = 0
        last = -9999
        for entry in sorted(zip(self.input.data, self.output.data),
                                           key=lambda v: max(v[1])):
            if entry[1][0] != last:
                idx += 1
                last = entry[1][0]
            input_data[idx%n].append(entry[0])
            output_data[idx%n].append(entry[1])
        return list(map(lambda entry: (np.array(entry[0]), np.array(entry[1])),
                        zip(input_data, output_data)))

    # Split into n pairs of numpy arrays.
    def split_stratified(self, n):
        input_data = [[] for _ in range(n)]
        output_data = [[] for _ in range(n)]
        for idx, entry in enumerate(sorted(zip(self.input.data, self.output.data), 
                                           key=lambda v: max(v[1]))):
            input_data[idx%n].append(entry[0])
            output_data[idx%n].append(entry[1])
        return list(map(lambda entry: (np.array(entry[0]), np.array(entry[1])), 
            zip(input_data, output_data)))

    # Mutably normalizes the data to the range [0..1]
    def normalize(self):
        self.input.normalize()
        self.output.normalize()

    def normalize_zscore(self):
        self.input.normalize_zscore()
        self.output.normalize_zscore()

# Creates a binary field for each unique occurrence of the attributes.
class NominalAttribute:
    def __init__(self, data, attribute, preprocess = lambda x: x):
        self.data    = data
        self.attribute = attribute
        self.preprocess = preprocess

    def _create_header(self, name):
        return Header(
            self.attribute + '_' + name.lower(),
            self.data,
            self.attribute,
            lambda v: 1 if self.preprocess(v) == name else 0
        )

    def headers(self):
        names = set()
        for row in self.data:
            attr = self.preprocess(row[self.attribute])
            if attr: names.add(attr)
        return map(self._create_header, names)

# Creates a binary field for each occurence of the attributes.
# A field can contain multiple attributes.
class NominalListAttribute:
    def __init__(self, data, attribute):
        self.data      = data
        self.attribute = attribute
    
    def _create_header(self, name):
        def _mapping(v):
            if not v: return math.nan
            return 1 if name in v else 0
        return Header(
            self.attribute + '_' + name.lower(),
            self.data,
            self.attribute,
            lambda v: 1 if name in v else 0
        )

    def headers(self):
        names = set()
        for row in self.data:
            for name in readlist(row[self.attribute]):
                if name: names.add(name)
        return map(self._create_header, names)

# Creates a number of regions that a language can fit into.
class LanguageAttribute:
    def __init__(self, data, attribute):
        self.data      = data
        self.attribute = attribute

    def _create_header(self, name_func):
        name, func = name_func
        return Header(
            self.attribute + '_' + name,
            self.data,
            self.attribute,
            func
        )

    def headers(self):
        nordic = set(['da', 'no', 'sv', 'nb'])
        european = set(['sh', 'sr', 'sk', 'es', 'bs', 'sl', 'hr', 'hu', 'ro',
                'sq', 'el', 'et', 'no', 'sv', 'fr', 'pt', 'nl', 'it', 'is', 'cs',
                'nb', 'ru', 'fi', 'de', 'dk'])
        asian = set(['cn', 'mn', 'vi', 'bo', 'ch', 'zh', 'pl', 'ko', 'id', 'ja'])

        data = [
            ('dk', lambda v: 1 if v == 'da' else 0),
            ('nordic', lambda v: 1 if v in nordic else 0),
            ('european', lambda v: 1 if v in european else 0),
            ('asian', lambda v: 1 if v in asian else 0),
            ('english', lambda v: 1 if v == 'en' else 0)
        ]
        return map(self._create_header, data)

# Returns the attribute as a numeric value.
# Optionally takes a normalization function to apply.
class NumericAttribute:
    def __init__(self, data, attribute):
        self.data      = data
        self.attribute = attribute
    
    def headers(self):
        header = Header(
            self.attribute,
            self.data,
            self.attribute,
            lambda v: getnumber(v)
        )
        return [header]

class MeanAttribute:
    def __init__(self, data, attribute, range_end, range_start = 0):
        self.data = data
        self.attribute = attribute
        self.range_end = range_end
        self.range_start = range_start

    def headers(self):
        def func(row):
            res = 0.0
            for idx in range(self.range_start, self.range_end):
                res += getnumber(row[self.attribute+str(idx)])
            return res/(self.range_end-self.range_start)

        header = Header(
            self.attribute,
            self.data,
            self.attribute,
            func,
            accepts_row = True
        )
        return [header]

# An attribute that is 1 if data is present, otherwise 0.
# Data is present if the value is not -1.
class PresenceAttribute:
    def __init__(self, data, attribute):
        self.data      = data
        self.attribute = attribute

    def headers(self):
        header = Header(
            self.attribute,
            self.data,
            self.attribute,
            lambda v: 1 if v != '-1' else 0
        )
        return [header]

# Returns the year, month and day of a date given in YYYY-MM-DD format.
class DateAttribute:
    def __init__(self, data, attribute):
        self.data      = data
        self.attribute = attribute
    
    def headers(self):
        def verifyDay(value):
            try:
                return datetime.datetime.strptime(value,"%Y-%m-%d").day
            except:
                return math.nan
        def verifyMonth(value):
            try:
                return datetime.datetime.strptime(value,"%Y-%m-%d").month
            except:
                return math.nan
        def verifyYear(value):
            try:
                return datetime.datetime.strptime(value,"%Y-%m-%d").year
            except:
                return math.nan
                
        h1 = Header(
            self.attribute + '_date',
            self.data,
            self.attribute,
            verifyDay
        )
        h2 = Header(
            self.attribute + '_month',
            self.data,
            self.attribute,
            verifyMonth
        )
        h3 = Header(
            self.attribute + '_year',
            self.data,
            self.attribute,
            verifyYear
        )
        return [h1,h2,h3]
  
class CustomAttribute:
    def __init__(self, data, attribute, name, func):
        self.name = name
        self.data = data
        self.attribute = attribute
        self.func = func
    
    def headers(self):
        header = Header(
            self.name,
            self.data,
            self.attribute,
            self.func
        )
        return [ header ]

# Dataset contains the tmdbids for each row the value should be present for
class ConstantAttribute:
    def __init__(self, dataset, attribute, value):
        self.data = dataset
        self.attribute = attribute
        self.value = value

    def headers(self):
        header = Header(
            self.attribute,
            self.data,
            self.attribute,
            lambda _: self.value)
        return [ header ]

# Splits a numeric attribute into n binary buckets of equal size.
class FrequencyBucketAttribute:
    def __init__(self, data, attribute, num_buckets):
        self.data        = data
        self.attribute   = attribute
        self.num_buckets = num_buckets

    def headers(self):
        ordered = list(sorted(
            filter(lambda row: not math.isnan(getnumber(row[self.attribute])), 
                self.data),
            key = lambda x: float(x[self.attribute])))
        elements_per_bucket = len(ordered)/self.num_buckets 

        maxvals = [-math.inf]
        for i in range(1, self.num_buckets):
            maxvals.append(
                float(ordered[int(i*elements_per_bucket-1)][self.attribute]))
        maxvals.append(math.inf)

        headers = []
        for i in range(self.num_buckets):
            def isinrange(v, i=i):
                num = getnumber(v)
                if math.isnan(num): return num
                return 1 if num >= maxvals[i] and num < maxvals[i+1] else 0

            header = Header(
                self.attribute + '_bucket_' + str(i),
                self.data,
                self.attribute,
                isinrange
            )
            headers.append(header)
        return headers


