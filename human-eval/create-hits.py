from captcha import getCaptcha
import pandas as pd
import numpy as np
import jsonpickle
import json
import sys
import os
import random
from xml.sax.saxutils import escape
from math import ceil
from mturk.client import MturkClient, createDirs
from mturk.quali_requirements import Comparator, WorkerLocale, WorkerNumberHitsApproved, WorkerPercentAssignmentsApproved, generateRequirement
import mturk.env
from hit import Hit
from generate_hit_params import generateHitParamsStyle, generateHitParamsCoherence

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)

with open('templates/hit-params.json') as f:
    hitParams = jsonpickle.decode(f.read())

currentTask = 'style'
with open(os.path.join(dname, 'templates', 'hit', currentTask, 'hit-stance.html'), 'r') as f:
    fileTemplate = f.read()
with open(os.path.join(dname, 'templates', 'hit', currentTask, 'turn.html'), 'r') as f:
    turnTemplate = f.read()

def generateHits(df, generateParamsFunc, captchaRate: float = 0.1, expertHits = []):
    hitList = []
    generatedTasks = 0
    generatedCaptchas = 0
    addedExpertHits = 0

    allHitParams = generateParamsFunc(df)
    for parameters in allHitParams:
        hitList.append(Hit(parameters=parameters))
        
    generatedTasks += len(hitList)
    numOfCaptchas = ceil(generatedTasks * captchaRate)
    captchaCandidates = random.sample(hitList, numOfCaptchas)
    for hit in captchaCandidates:
        hitList.append(getCaptcha(hit))

    generatedCaptchas += numOfCaptchas

    # TODO: add matching expert HITs
    addedExpertHits = 0
    for hit in expertHits:
        params = hit.parameters
        # create copy
        params = dict(params)
        # add expert hit marker / old hit id
        #params['expertHitId'] = hit.amtResponse['HITId']
        params['workerHitId'] = hit.amtResponse['HITId']
        hitList.append(Hit(parameters=params))
        addedExpertHits += 1
    
    generatedTasks = len(allHitParams)
    print('Generated tasks:', generatedTasks)
    print('Generated captchas:', generatedCaptchas)
    print('Added expert HITs:', addedExpertHits)
    print('Total number of HITs:', generatedTasks + generatedCaptchas + addedExpertHits)
    return hitList

def saveHits(hits, runName=None, link=None):
    if (runName == None):
        print('Enter run name:')
        runName = sys.stdin.readline().replace('\n', '')

    path = 'hit-data/{}.json'.format(runName)
    createDirs(path)
    with open(path, 'w') as f:
        jsonString = jsonpickle.encode(hits)
        f.write(jsonString)      
    print('Saved HIT data in ', path)

    if (link == None): 
        return
    path = 'hit-data/links.csv'
    createDirs(path)
    with open(path, 'a') as f:
        f.write('\n{0},{1}'.format(runName, link))

def getCustomRequirement(client, folderName: str, minValue: int):
    with open(os.path.join('templates', 'qualification', folderName, 'quali-params.json')) as f:
        qualiParams = json.load(f)
    qualiTest = client.getQualificationType(qualiParams['name'])
    # 100% correct answers
    customRq = generateRequirement(
        qualiTypeId=qualiTest['QualificationTypeId'],
        comparator=Comparator.GreaterThanOrEqualTo,
        integerValues=[minValue,]
    )
    return customRq

def deployHits(hits, runName: str, hitParams=hitParams, fileTemplate: str = fileTemplate, turnTemplate: str = turnTemplate,
    maxAssignments: int = 3, lifetimeInSeconds: int = 600, assignmentDurationInSeconds : int = 600,
    autoApprovalDelayInSeconds: int = 5*24*60*60, reward: str = '0.00', 
    mturkEnv: mturk.env.MturkEnv = mturk.env.sandbox, profileName=None):
    """
    Follow AWS best practices for setting up credentials here:
    http://boto3.readthedocs.io/en/latest/guide/configuration.html
    """
    client = MturkClient(mturkEnv=mturkEnv)

    # Worker qualification requirements
    requirements = [] 
    requirements.append(WorkerPercentAssignmentsApproved(97))

    if (mturkEnv == mturk.env.sandbox):
        # make sure we can test the HIT in the sandbox.
        pass
    else:
        requirements.append(getCustomRequirement(client, 'warmup', 85))
        requirements.append(getCustomRequirement(client, 'demographics', 100))
        requirements.append(WorkerLocale(['US', 'CA']))
        requirements.append(WorkerNumberHitsApproved(1000))

    questions = []
    requesterAnnotations = []
    random.shuffle(hits)
    for hit in hits:
        # Make sure the form leads to the appropiate endpoint
        hit.parameters['postAction'] = mturkEnv.postAction
        # Create the HIT on AMT
        questions.append(hit.generateHtmlQuestion(fileTemplate, turnTemplate))

        requesterAnnotation = { 
            'runName': runName
        }
        # no captchas yet
        if (hit.isCaptcha()):
            captcha = dict(hit.parameters['captcha'])
            # make sure we stay in requester annotation limits.
            for turn in captcha.values():
                del turn['replaceMessage']
            requesterAnnotation['captcha'] = captcha
        if (hit.isExpertHit()):
            requesterAnnotation['expertHitId'] = hit.parameters['expertHitId']
        requesterAnnotation = jsonpickle.encode(requesterAnnotation)
        requesterAnnotations.append(requesterAnnotation)


    responses = client.createHits(questions, maxAssignments=maxAssignments, lifetimeInSeconds=lifetimeInSeconds, 
        assignmentDurationInSeconds=assignmentDurationInSeconds, reward=reward,
        autoApprovalDelayInSeconds=autoApprovalDelayInSeconds,
        title=hitParams['title'], keywords=hitParams['keywords'], description=hitParams['description'],
        qualificationRequirements=requirements, savePath='hit-data/{}.mturk.json'.format(runName),
        requesterAnnotations=requesterAnnotations
    )

    for hit, response in zip(hits, responses):
        hit.amtResponse = response
    
    return client.getWorkerLink(response['HITTypeId'])


# ==== settings start ====
mturkEnv = mturk.env.sandbox
# 2 weeks
lifetime = 2*7*24*60*60
# 12 minutes
assignmentDuration = 12*60
# 5 days
autoApprovalDelay = 5*24*60*60
#maxAssignments = 3
maxAssignments = 7
#captchaRate = 0.1
captchaRate = 0
reward = '0.15'
runName = 'debate-expert-hits-{}'.format(mturkEnv.name)
# ==== settings end ====

# Load data.
with open('hit-data/debate-expert-sample.json', 'r') as f:
    expertHits = jsonpickle.decode(f.read(), keys=True, classes=Hit)
# Style
#data = pd.read_csv(os.path.join(
#    dname, 'paraphrases', runName.replace('-{}'.format(mturkEnv.name), ''), 'output.csv'))
#hits = generateHits(df=data, generateParamsFunc=generateHitParamsStyle, captchaRate=captchaRate, expertHits=[])
hits = generateHits(df=pd.DataFrame().assign(conversationId=pd.NA), generateParamsFunc=generateHitParamsStyle, captchaRate=captchaRate, expertHits=expertHits)
# Coherence
#data = pd.read_csv('paraphrases/coherence-1/input.csv', index_col=0)
#hits = generateHits(df=data, generateParamsFunc=generateHitParamsCoherence, expertHits=expertHits)
# make a backup copy of HITs in case of errors when deploying
saveHits(hits, runName=runName)
link = deployHits(
    hits, runName=runName, 
    maxAssignments=maxAssignments, 
    lifetimeInSeconds=lifetime, 
    assignmentDurationInSeconds=assignmentDuration, 
    autoApprovalDelayInSeconds=autoApprovalDelay,
    reward=reward, mturkEnv=mturkEnv
)
saveHits(hits, runName=runName, link=link)
print('done.')