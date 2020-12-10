import json

f = open('result.json')

# from Camera-3_2020-06-12_09:45:06-check to Camera-3_2020-06-12_11:45:14-check
# person, backpack, handbag, laptop, book
personAnswer   = [0, 0, 1, 4, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 2, 2, 0, 1, 6, 1, 4, 0, 1, 3, 1, 0, 1, 2, 1, 0, 2, 2, 1, 1, 4, 2, 0, 1, 2, 0]
backpackAnswer = [1, 0, 1, 4, 0, 3, 2, 1, 0, 3, 2, 0, 0, 2, 3, 3, 1, 3, 4, 2, 1, 2, 3, 2, 0, 2, 3, 2, 0, 2, 1, 2, 0, 2, 3, 2, 0, 2, 1, 1]
handbagAnswer  = [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
laptopAnswer   = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 2, 1, 0, 1, 3, 1, 0, 1, 2, 1, 1, 1, 1, 1, 2, 0, 1, 1, 1, 0, 0]
bookAnswer     = [0, 1, 0, 2, 1, 0, 1, 0, 1, 0, 1, 0, 2, 1, 1, 0, 2, 1, 1, 0, 1, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 0, 2, 1, 0, 1, 1, 1, 0, 0]

personCount = [0]*len(personAnswer)
backpackCount = [0]*len(personAnswer)
handbagCount = [0]*len(personAnswer)
laptopCount = [0]*len(personAnswer)
bookCount = [0]*len(personAnswer)

'''wrongPersonCount = 0
wrongBackpackCount = 0
wrongHandbagCount = 0
wrongLaptopCount = 0
wrongBookCount = 0'''

openedJson = json.load(f)
# stat = []
jsonIterateStart = 0

checkResultJson = []

for j in openedJson:
    if (j['filename'].__contains__('Camera-3_2020-06-12_09:45:06-check')):
        jsonIterateStart = int(j['frame_id']) - 1
        break

for i in range(jsonIterateStart, jsonIterateStart + len(personAnswer)):
    objects = openedJson[i]['objects']
    for o in objects:
        if o['name'] == 'person':
            personCount[i - jsonIterateStart] += 1
        elif o['name'] == 'backpack':
            backpackCount[i - jsonIterateStart] += 1
        elif o['name'] == 'handbag':
            handbagCount[i - jsonIterateStart] += 1
        elif o['name'] == 'laptop':
            laptopCount[i - jsonIterateStart] += 1
        elif o['name'] == 'book':
            bookCount[i - jsonIterateStart] += 1
        # temp[int(o['name'])] += 1
    
    '''if personCount[i - jsonIterateStart] == personAnswer[i - jsonIterateStart]:
        wrongPersonCount += 1
    if backpackCount[i - jsonIterateStart] == backpackAnswer[i - jsonIterateStart]:
        wrongBackpackCount += 1
    if handbagCount[i - jsonIterateStart] == handbagAnswer[i - jsonIterateStart]:
        wrongHandbagCount += 1
    if laptopCount[i - jsonIterateStart] == laptopAnswer[i - jsonIterateStart]:
        wrongLaptopCount += 1
    if bookCount[i - jsonIterateStart] == bookAnswer[i - jsonIterateStart]:
        wrongBookCount += 1 '''
    
    dictionary = {
        "file_name": openedJson[i]['filename'],
        "wrong_person_count": abs(personCount[i - jsonIterateStart] - personAnswer[i - jsonIterateStart]),
        "wrong_backpack_count": abs(backpackCount[i - jsonIterateStart] - backpackAnswer[i - jsonIterateStart]),
        "wrong_handbag_count": abs(handbagCount[i - jsonIterateStart] - handbagAnswer[i - jsonIterateStart]),
        "wrong_laptop_count": abs(laptopCount[i - jsonIterateStart] - laptopAnswer[i - jsonIterateStart]),
        "wrong_book_count": abs(bookCount[i - jsonIterateStart] - bookAnswer[i - jsonIterateStart])
    }
    checkResultJson.append(dictionary)
    
    '''wrongPersonCount += abs(personCount[i - jsonIterateStart] - personAnswer[i - jsonIterateStart])
    wrongBackpackCount += abs(backpackCount[i - jsonIterateStart] - backpackAnswer[i - jsonIterateStart])
    wrongHandbagCount += abs(handbagCount[i - jsonIterateStart] - handbagAnswer[i - jsonIterateStart])
    wrongLaptopCount += abs(laptopCount[i - jsonIterateStart] - laptopAnswer[i - jsonIterateStart])
    wrongBookCount += abs(bookCount[i - jsonIterateStart] - bookAnswer[i - jsonIterateStart])'''

'''print('wrong person_count: ' + str(wrongPersonCount))
print('wrong backpack_count: ' + str(wrongBackpackCount))
print('wrong handbag_count: ' + str(wrongHandbagCount))
print('wrong laptop_count: ' + str(wrongLaptopCount))
print('wrong book_count: ' + str(wrongBookCount)) '''

print('person   detected: ' + str(sum(personCount)) + ', ground_truth: ' + str(sum(personAnswer)) + ', wrong_count: ' + str(abs(sum(personAnswer)-sum(personCount))))
print('backpack detected: ' + str(sum(backpackCount)) + ', ground_truth: ' + str(sum(backpackAnswer)) + ', wrong_count: ' + str(abs(sum(backpackAnswer)-sum(backpackCount))))
print('handbag  detected: ' + str(sum(handbagCount)) + ', ground_truth: ' + str(sum(handbagAnswer)) + ', wrong_count: ' + str(abs(sum(handbagAnswer)-sum(handbagCount))))
print('laptop   detected: ' + str(sum(laptopCount)) + ', ground_truth: ' + str(sum(laptopAnswer)) + ', wrong_count: ' + str(abs(sum(laptopAnswer)-sum(laptopCount))))
print('book     detected: ' + str(sum(bookCount)) + ', ground_truth: ' + str(sum(bookAnswer)) + ', wrong_count: ' + str(abs(sum(bookAnswer)-sum(bookCount))))

checkResultJsonDump = json.dumps(checkResultJson, indent=2)
with open('checkResult.json', 'w') as outfile:
    outfile.write(checkResultJsonDump)
print('\n\ncheck result file saved successfully!')