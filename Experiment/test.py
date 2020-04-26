import csv
print("Hello world yaaaasss")
columnNames = {
    'TIME': 0,
    'ACC1': 1,
    'ACC2': 2,
    'ACC2': 3,
    'ACC2': 4,
    'AUDIO1': 5,
    'AUDIO2': 6,
    'AUDIO3': 7,
    'AUDIO4': 8,
    'SCRN1': 9,
    'SCRN2': 10,
    'SCRN3': 11,
    'SCRN4': 12,
    'NOTIF1': 13,
    'NOTIF2': 14,
    'NOTIF3': 15,
    'NOTIF4': 16,
    'LIGHT1': 17,
    'LIGHT2': 18,
    'LIGHT3': 19,
    'LIGHT4': 20,
    'APP_VID1': 21,
    'APP_VID2': 22,
    'APP_VID3': 23,
    'APP_VID4': 24,
    'APP_COMM1': 25,
    'APP_COMM2': 26,
    'APP_COMM3': 27,
    'APP_COMM4': 28,
    'APP_OTHER1': 29,
    'APP_OTHER2': 30,
    'APP_OTHER3': 31,
    'APP_OTHER4': 32,

}

with open('../../testData/testData.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            # print(f'Column names are {", ".join(row)}')
            print(row)
            line_count += 1
        else:
            print(row["APP_OTHER4"])
            line_count += 1
    print(line_count)
