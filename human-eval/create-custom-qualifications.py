from mturk.client import MturkClient
from mturk.quali_requirements import generateRequirement, Comparator
import mturk.env
import jsonpickle

client = MturkClient(mturkEnv=mturk.env.live)

def getTemplates(folder):
    with open('templates/qualification/{}/quali-test.xml'.format(folder), 'r', encoding='utf-8') as f: 
        questionTemplate = f.read()

    with open('templates/qualification/{}/quali-params.json'.format(folder)) as f:
        qualiParams = jsonpickle.decode(f.read())

    return qualiParams, questionTemplate

def createDemographicsTest():
    qualiParams, questionTemplate = getTemplates('demographics')

    # 15 minutes
    testDuration = 15*60
    # 1 minute
    retryDelay = 60
    myQualification = client.createQualificationType(
        name=qualiParams['name'], description=qualiParams['description'], keywords=qualiParams['keywords'],
        test=questionTemplate, testDurationInSeconds=testDuration, retryDelayInSeconds=retryDelay
    )

    print('Qualification Name:', qualiParams['name'])
    print('Qualification Type Id:', myQualification['QualificationTypeId'])

def createWarmupTest():
    qualiParams, questionTemplate = getTemplates('warmup')

    with open('templates/qualification/{}/answer-key.xml'.format('warmup'), 'r', encoding='utf-8') as f:
        answerKey = f.read()

    # 20 minutes
    testDuration = 20*60
    # 1 hour
    retryDelay = 60*60
    myQualification = client.createQualificationType(
        name=qualiParams['name'], description=qualiParams['description'], keywords=qualiParams['keywords'],
        test=questionTemplate, answerKey=answerKey,
        testDurationInSeconds=testDuration, retryDelayInSeconds=retryDelay
    )

    print('Qualification Name:', qualiParams['name'])
    print('Qualification Type Id:', myQualification['QualificationTypeId'])


createDemographicsTest()
createWarmupTest()