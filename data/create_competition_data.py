import csv
import datetime

RELEASE_FILE = 'adjusted_dates.csv'
BUDGET_FILE = 'mining_ready/2004budget.csv'
OUTPUT_FILE = 'mining_ready/2004competition.csv'
NUM_COMPETITORS = 5
DAY_RANGE = 18

def get_releases():
    res = {}
    with open(RELEASE_FILE, encoding='utf-8') as input_file:
        reader = csv.DictReader(input_file, delimiter=';', quotechar='"')
        for row in reader:
            res[row['tmdbid']] = datetime.datetime.strptime(
                    row['release'], '%Y-%m-%d')
    return res

def get_budgets():
    res = {}
    with open(BUDGET_FILE, encoding='utf-8') as input_file:
        reader = csv.DictReader(input_file, delimiter=';', quotechar='"')
        for row in reader:
            res[row['tmdbid']] = int(row['budget'])
    return res

def get_greatest_competitors(releases, budgets):
    res = {}
    for tmdbid, date in releases.items():
        res[tmdbid] = []
        for i in range(NUM_COMPETITORS):
            maxbudget = 0
            max_comp = -1
            for competitor, budget in budgets.items():
                if competitor in releases \
                        and abs((date-releases[competitor]).days) <= DAY_RANGE \
                        and budget > maxbudget and competitor != tmdbid \
                        and competitor not in res[tmdbid]:
                    maxbudget = budget
                    max_comp = competitor
            res[tmdbid].append(max_comp)
    return res
    
def write_outputs(competitors, releases, budgets):
    with open(OUTPUT_FILE, 'w+', encoding='utf-8') as output_file:
        fields = ['tmdbid']
        for comp in range(NUM_COMPETITORS):
            fields.append('day_diff_'+str(comp))
            fields.append('budget_'+str(comp))
        writer = csv.DictWriter(output_file, delimiter=';', quotechar='"',
            fieldnames=fields)
        writer.writeheader()

        for tmdbid, comps in competitors.items():
            row = { 'tmdbid': tmdbid }
            for idx, comp in enumerate(comps):
                if comp in releases and comp in budgets:
                    row['day_diff_'+str(idx)] = \
                        str((releases[comp]-releases[tmdbid]).days)
                    row['budget_'+str(idx)] = str(budgets[comp])
                else:
                    row['day_diff_'+str(idx)] = '0'
                    row['budget_'+str(idx)] = '0'
            writer.writerow(row)
            
def main():
    releases = get_releases()
    budgets = get_budgets()
    closest = get_greatest_competitors(releases, budgets)
    write_outputs(closest, releases, budgets)

main()
