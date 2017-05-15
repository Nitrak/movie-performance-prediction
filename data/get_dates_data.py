import csv
import datetime

OUTPUT_FILE = 'mining_ready/2004dates_data.csv'
INPUT_FILE = 'mining_ready/2004dates.csv'
FIELDS = ['date', 'year', 'weeknum', 'day_in_week', 'holiday', 'holiday_effect']

HOLIDAY_EFFECT = {
    'jul': 9.626,
    'vinter': 14.766,
    'påske': 4.779,
    'bededag': -18.438,
    'himmelfart': -21.082,
    'pinse': -9.916,
    'sommer': 12.885,
    'efterår': 11.596,
    '': 0
}

#Holidays
SUMMER = {
    2004: ((19, 6), (2, 8)),
    2005: ((18, 6), (7, 8)),
    2006: ((24, 6), (6, 8)),
    2007: ((30, 6), (12, 8)),
    2008: ((28, 6), (12, 8)),
    2009: ((27, 6), (11, 8)),
    2010: ((26, 6), (10, 8)),
    2011: ((25, 6), (11, 8)),
    2012: ((30, 6), (12, 8)),
    2013: ((29, 6), (13, 8)),
}

AUTUMN = {
    2004: ((9, 10), (17, 10)),
    2005: ((15, 10), (23, 10)),
    2006: ((14, 10), (22, 10)),
    2007: ((13, 10), (21, 10)),
    2008: ((11, 10), (19, 10)),
    2009: ((10, 10), (18, 10)),
    2010: ((16, 10), (24, 10)),
    2011: ((17, 10), (23, 10)),
    2012: ((13, 10), (21, 10)),
    2013: ((12, 10), (20, 10)),
}

CHRISTMAS = {
    2004: ((22, 12), (2, 1)),
    2005: ((22, 12), (3, 1)),
    2006: ((21, 12), (7, 1)),
    2007: ((20, 12), (2, 1)),
    2008: ((20, 12), (4, 1)),
    2009: ((23, 12), (3, 1)),
    2010: ((18, 12), (2, 1)),
    2011: ((19, 12), (1, 1)),
    2012: ((21, 12), (2, 1)),
    2013: ((21, 12), (1, 1)),
}

WINTER = {
    2004: ((7, 2), (15, 2)),
    2005: ((12, 2), (20, 2)),
    2006: ((11, 2), (19, 2)),
    2007: ((10, 2), (18, 2)),
    2008: ((9, 2), (17, 2)),
    2009: ((7, 2), (15, 2)),
    2010: ((13, 2), (21, 2)),
    2011: ((19, 2), (27, 2)),
    2012: ((11, 2), (19, 2)),
    2013: ((9, 2), (17, 2)),
}

EASTER = {
    2004: ((3, 4), (12, 4)),
    2005: ((19, 3), (28, 3)),
    2006: ((8, 4), (17, 4)),
    2007: ((31, 3), (9, 4)),
    2008: ((15, 3), (24, 3)),
    2009: ((4, 4), (13, 4)),
    2010: ((27, 3), (5, 4)),
    2011: ((16, 4), (25, 4)),
    2012: ((31, 3), (9, 4)),
    2013: ((23, 3), (1, 4)),
}

BEDEDAG = {
    2004: ((7, 5), (9, 5)),
    2005: ((22, 5), (24, 5)),
    2006: ((12, 5), (14, 5)),
    2007: ((4, 5), (6, 5)),
    2008: ((18, 4), (20, 4)),
    2009: ((8, 5), (10, 5)),
    2010: ((30, 4), (2, 5)),
    2011: ((20, 5), (22, 5)),
    2012: ((4, 5), (6, 5)),
    2013: ((26, 4), (28, 4)),
}

HIMMELFART = {
    2004: ((20, 5), (23, 5)),
    2005: ((5, 5), (8, 5)),
    2006: ((25, 5), (28, 5)),
    2007: ((17, 5), (20, 5)),
    2008: ((1, 5), (4, 5)),
    2009: ((21, 5), (24, 5)),
    2010: ((13, 5), (16, 5)),
    2011: ((2, 6), (5, 6)),
    2012: ((17, 5), (20, 5)),
    2013: ((9, 5), (12, 5)),
}

PINSE = {
    2004: ((29, 5), (31, 5)),
    2005: ((14, 5), (16, 5)),
    2006: ((3, 6), (5, 6)),
    2007: ((26, 5), (28, 5)),
    2008: ((10, 5), (12, 5)),
    2009: ((30, 5), (1, 6)),
    2010: ((22, 5), (24, 5)),
    2011: ((11, 6), (13, 6)),
    2012: ((26, 5), (28, 5)),
    2013: ((18, 5), (20, 5)),
}

# Date -> year, weeknumber, weekday
iso_dates = {}
holiday = {}

def is_between(limits, month, day):
    (sday, smonth), (eday, emonth) = limits
    if emonth == smonth:
        return month == smonth and day >= sday and day <= eday
    if emonth > smonth:
        if month == smonth: return day >= sday
        if month == emonth: return day <= eday
        return month > smonth and month < emonth
    if month == smonth: return day >= sday
    if month == emonth: return day <= eday
    return month > smonth or month < emonth

def get_holiday(date):
    if is_between(SUMMER[date.year], date.month, date.day): return 'sommer'
    if is_between(AUTUMN[date.year], date.month, date.day): return 'efterår'
    if is_between(CHRISTMAS[date.year], date.month, date.day): return 'jul'
    if is_between(WINTER[date.year], date.month, date.day): return 'vinter'
    if is_between(EASTER[date.year], date.month, date.day): return 'påske'
    if is_between(BEDEDAG[date.year], date.month, date.day): return 'bededag'
    if is_between(HIMMELFART[date.year], date.month, date.day): return 'himmelfart'
    if is_between(PINSE[date.year], date.month, date.day): return 'pinse'
    return '' 

with open(INPUT_FILE, encoding='utf-8') as input_file:
    reader = csv.DictReader(input_file, delimiter=';', quotechar='"')
    for row in reader:
        date = row['date']
        try:
            parsed = datetime.datetime.strptime(date, '%Y-%m-%d')
            iso_dates[date] = parsed.isocalendar()
            holiday[date] = get_holiday(parsed)
        except ValueError:
            pass

with open(OUTPUT_FILE, 'w+', encoding='utf-8') as output_file:
    writer=csv.DictWriter(output_file, fieldnames=FIELDS, delimiter=';',
            quotechar='"')
    writer.writeheader()
    for date in iso_dates:
        row = {
            'date': date,
            'year': iso_dates[date][0],
            'weeknum': iso_dates[date][1],
            'day_in_week': iso_dates[date][2],
            'holiday': holiday[date],
            'holiday_effect': HOLIDAY_EFFECT[holiday[date]],
        }
        writer.writerow(row)
