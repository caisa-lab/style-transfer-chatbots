import pandas as pd
from math import ceil

def getAverageAssignmentDuration(times):
    durations = []
    for time1, time2 in times:
        diff = time2 - time1
        durations.append(diff)
    
    durations = pd.Series(durations)
    meanDuration = durations.mean()
    medianDuration = durations.median()
    return meanDuration, medianDuration

def formatTimeDelta(td):
    minutes, seconds = divmod(td.seconds + td.days * 86400, 60)
    hours, minutes = divmod(minutes, 60)
    return '{:02d}:{:02d}'.format(minutes, seconds)

df = pd.read_csv('hit-output/caisa-test-1.assignments.csv', index_col=0)
df['acceptTime'] = pd.to_datetime(df['acceptTime'], infer_datetime_format=True) 
df['submitTime'] = pd.to_datetime(df['submitTime'], infer_datetime_format=True) 
meanDuration, medianDuration = getAverageAssignmentDuration(df[['acceptTime', 'submitTime']].values)

print('Expert test results:')
print('Mean duration:', formatTimeDelta(meanDuration))
print('Median duration:', formatTimeDelta(medianDuration))


tax = 0.2
secondsPerHit = 65
assignmentsPerHit = 3
captchaRate = 1 / 10
# usd $$$
hourlyRate = 8.3
numOfHits = 50 * 9

print('=== Input ===')
print('Tax: {}%'.format(tax * 100))
print('Seconds per HIT:', secondsPerHit)
print('Assignments per HIT:', assignmentsPerHit)
print('Captcha rate: {}%'.format(captchaRate * 100))
print('Hourly worker rate: {}$'.format(hourlyRate))
print('Number of HITs:', numOfHits)

# calculations start here
numOfAssignments = numOfHits + int(captchaRate * numOfHits)
numOfAssignments *= assignmentsPerHit

pricePerHit = secondsPerHit * (hourlyRate / (60 * 60))
pricePerHit = round(pricePerHit, 2)
cost = pricePerHit * numOfAssignments
cost += cost * tax

print('=== Output ===')
print('Number of HITs*Assignments:', numOfAssignments)
print('Hit reward (w/o tax): {}$'.format(pricePerHit))
print('Hit reward (w/ tax): {}$'.format(pricePerHit + pricePerHit * tax))
print('Cost (w/ tax): {}$'.format(cost))