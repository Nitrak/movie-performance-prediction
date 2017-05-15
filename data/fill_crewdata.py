# Used to fill actor and director data in 2004data.csv

import csv
import datetime

STARRING_FILE = 'mining_ready/data.csv'
INPUT_FILE = 'mining_ready/2004data.csv'
OUTPUT_FILE = 'mining_ready/2004data2.csv'
TICKET_FILE = 'mining_ready/dst_movies.csv'

# tickets sold per tmdbid
tickets_sold = {}
# data about all movies from data.csv
all_movies = []

def get_date(row, key):
    if row[key]:
        return datetime.datetime.strptime(row[key], '%Y-%m-%d')
    else:
        return datetime.datetime(1900, 1, 1)

def get_starring_date(row):
    return get_date(row, 'release')

def get_or_default(dic, key, default):
    if key in dic:
        return dic[key]
    else:
        return default

def get_star_value(tmdbid):
    tickets = 0
    if tmdbid in tickets_sold: tickets = int(tickets_sold[tmdbid])
    if tickets < 29000: return 0
    if tickets < 57000: return 1
    if tickets < 130000: return 2
    return 3

def get_crew(actors, directors):
    actors = actors.strip('[]').split(', ')
    directors = directors.strip('[]').split(', ')
    for idx, name in enumerate(actors):
        actors[idx] = name.strip("'")
    for idx, name in enumerate(directors):
        directors[idx] = name.strip("'")
    director = ''
    if len(directors) > 0: director = directors[0]
    return (actors, director)

with open(TICKET_FILE, encoding='utf-8') as input_file:
    reader = csv.DictReader(input_file, delimiter=';', quotechar='"')
    for row in reader:
        tickets_sold[row['tmdbid']] = row['sold']

with open(STARRING_FILE, encoding='utf-8') as input_file:
    reader = csv.DictReader(input_file, delimiter=';', quotechar='"')
    for row in reader:
        all_movies.append(row)
    all_movies.sort(key = get_starring_date)


current_idx = 0
number_starred = {}
actor_success = {}
number_directed = {}
director_success = {}
with open(INPUT_FILE, encoding='utf-8') as input_file, \
     open(OUTPUT_FILE, 'w+', encoding='utf-8') as output_file:
    reader = csv.DictReader(input_file, delimiter=';', quotechar='"')
    writer = csv.DictWriter(output_file, fieldnames=reader.fieldnames,
            delimiter=';', quotechar='"')
    writer.writeheader()

    for row in reader:
        actors, director = get_crew(row['actors'], row['director'])

        # Update historic data
        while(get_date(all_movies[current_idx], 'release') < get_date(row, 'first_showing')):
            hist_actors, hist_director = get_crew(
               all_movies[current_idx]['actors'], all_movies[current_idx]['director'])
            for name in hist_actors:
                number_starred[name] = get_or_default(number_starred, name, 0)+1
                actor_success[name] = get_or_default(actor_success, name, 0)+get_star_value(all_movies[current_idx]['tmdbid'])
            number_directed[hist_director] = get_or_default(
                number_directed, hist_director, 0)+1
            director_success[hist_director] = get_or_default(
                director_success, hist_director, 0) + get_star_value(all_movies[current_idx]['tmdbid']) 
            current_idx += 1

        # Init for missing data
        for idx in range(3):
            row['actor' + str(idx+1) + '_starred_movies'] = 0
            row['actor' + str(idx+1) + '_starred_success'] = 0
        row['director_movies'] = 0
        row['director_success'] = 0

        # Fill present data
        for idx, name in enumerate(actors):
            if name:
                row['actor' + str(idx+1) + '_starred_movies'] = get_or_default(
                        number_starred, name, 0)
                row['actor' + str(idx+1) + '_starred_success'] = get_or_default(
                        actor_success, name, 0)
        if director:
            row['director_movies'] = get_or_default(number_directed, director, 0)
            row['director_success'] = get_or_default(director_success, director, 0)
        writer.writerow(row) 


