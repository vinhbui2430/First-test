import csv 

with open('VN_weather.csv', 'r') as csvfile:
    # handle header line, save it for writing to output file
    # and skip it for further processing
    header = csvfile.readline()    
    # create reader object
    reader = csv.reader(csvfile, delimiter=',')
    results = filter(lambda row: row[0] == 'Ha Noi' , reader)
    # store filtered data in a list
    filtered_data = list(results)

# create writer object
with open('Hanoi_weather.csv', 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    # write a row to the CSV file
    writer.writerow(['province', 'max', 'min', 'wind', 	'wind_d', 'rain', 'humidi',	'cloud', 'pressure', 'date'])
    # write rows to the CSV file
    writer.writerows(filtered_data)