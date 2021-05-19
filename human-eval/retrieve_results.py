import pandas as pd
import numpy as np
import jsonpickle
from math import floor
from mturk.client import MturkClient
import mturk.env as env
from hit import Hit
from hit_list import orderHitsById
from captcha import scoreCaptcha

mturkEnv = env.sandbox
client = MturkClient(mturkEnv=mturkEnv, confirmEnv=False)

runName = 'debate-expert-hits-{}'.format(mturkEnv.name)
with open('hit-data/{}.json'.format(runName), 'r') as f:
    hits = jsonpickle.decode(f.read(), keys=True, classes=Hit)

answerColumns = [
    'moreNatural', 
    'moreLikeable',
    'personalPreference',
    'bot1Incoherent[]',
    'bot2Incoherent[]',
    'bot1Offensive[]',
    'bot2Offensive[]',
    'reason'
]
if (hits[0].getScenario() == 'debate'):
    answerColumns.append('agreesWithIssue')

def getAssignments(hitIds):
    responses = client.listAssignmentsForHits(hitIds)
    return responses

def extractAnnotations(assignments, hit):
    annotations = []

    for assignment in assignments:
        answer = dict(assignment['AnswerDict'])
        hitId = assignment['HITId']
        if (hitId != hit.getHitId()):
            raise ValueError('Given HIT does not match the assignment.')

        answer['hitType'] = hit.getHitType()        
        answer['hitId'] = hitId
        answer['workerId'] = assignment['WorkerId']
        answer['assignmentId'] = assignment['AssignmentId']
        answer['acceptTime'] = assignment['AcceptTime']
        answer['submitTime'] = assignment['SubmitTime']
        answer['scenario'] = hit.getScenario()

        annotations.append(answer)
    
    return annotations

def transformAnnotations(assignments, hits):
    df = assignments
    df = df.loc[df['hitType'] == 'task']
    annotations = []
    for hit in hits:
        hitId = hit.getHitId()
        curr = df.loc[df['hitId'] == hitId]
        numOfVotes = curr.shape[0]
        annotation = { 
            'hitId': hitId,
            'hitType': hit.getHitType(),
            'scenario': hit.getScenario(),
            'assignments': numOfVotes
        }

        conversationId = hit.parameters.get('conversationId', '')
        if (conversationId != ''):
            annotation['conversationId'] = conversationId
        style1 = hit.parameters.get('style1', '')
        if (style1 != ''):
            style2 = hit.parameters.get('style2', '')
            annotation['style1'] = style1
            annotation['style2'] = style2

        if (numOfVotes > 0):
            error, errorAgree = getVoteResult(curr['error'], dropNaN=False)
            annotation['error'] = error
            annotation['errorAgree'] = errorAgree
            if (error == 'true'):
                reason, reasonAgree = getVoteResult(curr['reason'], dropNaN=False)
                annotation['reason'] = reason
                annotation['reasonAgree'] = reasonAgree
                annotations.append(annotation)
                continue
            # else
            curr = curr.loc[curr['error'].isnull()]
            for key in answerColumns:
                votes = curr[key]
                answer, answerAgree = getVoteResult(votes, numOfVotes=numOfVotes, dropNaN=False)
                annotation[key] = answer
                annotation[key + 'Agree'] = answerAgree
        
        annotations.append(annotation)
    
    return pd.DataFrame(annotations)

def getVoteResult(series, numOfVotes: int = None, dropNaN=True):
    votes = series.value_counts(dropna=dropNaN)
    if (numOfVotes == None):
        numOfVotes = len(series)
    agreement = votes.iat[0] / numOfVotes
    return votes.index[0], agreement

def formatTimeDelta(td):
    minutes, seconds = divmod(td.seconds + td.days * 86400, 60)
    hours, minutes = divmod(minutes, 60)
    return '{:02d}:{:02d}'.format(minutes, seconds)

def getAverageAssignmentDuration(times):
    durations = []
    for time1, time2 in times:
        diff = time2 - time1
        durations.append(diff)
    
    durations = pd.Series(durations)
    meanDuration = formatTimeDelta(durations.mean())
    medianDuration = formatTimeDelta(durations.median())
    return meanDuration, medianDuration

def getWorkerStats(annotations, hits, expertResults):
    # generate per worker stats
    workerIds = annotations['workerId'].unique()
    hitIndex = orderHitsById(hits)

    workerStats = []
    for workerId in workerIds:
        df = annotations
        df = df.loc[df['workerId'] == workerId].sort_values(by='submitTime')
        timeBetweenSubmits = []
        lastSubmit = None
        for idx, row in df.iterrows():
            submittedAt = row['submitTime']
            if (lastSubmit != None):
                timeBetweenSubmits.append(submittedAt - lastSubmit)
            lastSubmit = submittedAt
        if (len(timeBetweenSubmits) > 1):
            timeBetweenSubmits = pd.Series(timeBetweenSubmits)
            timeBetweenSubmits = formatTimeDelta(timeBetweenSubmits.median())
        else:
            timeBetweenSubmits = ''

        meanDuration, medianDuration = getAverageAssignmentDuration(df[['acceptTime', 'submitTime']].values)
        numOfAssignments = df.shape[0]
        captchas = df.loc[df['hitType'] == 'captcha']
        expertHits = df.loc[df['hitType'] == 'expert']
        numOfCaptchas = captchas.shape[0]
        captchasCorrect = None
        if (numOfCaptchas > 0):
            captchasCorrect = 0

            for index, row in captchas.iterrows():
                captchasCorrect += scoreCaptcha(row, hitIndex)
                
            captchasCorrect /= 2 * numOfCaptchas
            captchasCorrect = round(captchasCorrect, 3)
        numOfExpertHits = expertHits.shape[0]
        votesCorrect = None
        numOfVotes = None
        if (numOfExpertHits > 0):
            votesCorrect = 0
            numOfVotes = 0
            threshold = 0.7
            for index, row in expertHits.iterrows():
                hit = hitIndex[row['hitId']]
                expert = expertResults.loc[expertResults['hitId'] == hit.parameters['expertHitId']].iloc[0]
                if (expert['bestAgree'] >= threshold):
                    numOfVotes += 1
                    if (int(row['best']) == int(expert['best'])):
                        votesCorrect += 1
                if (expert['worstAgree'] >= threshold):
                    numOfVotes += 1
                    if (int(row['worst']) == int(expert['worst'])):
                        votesCorrect += 1
            if (numOfVotes > 0):
                votesCorrect /= numOfVotes      

        workerStats.append({
            'workerId': workerId,
            'numOfAssignments': numOfAssignments,
            'numOfCaptchas': numOfCaptchas,
            'captchasCorrect': captchasCorrect,
            'meanDuration': meanDuration,
            'medianDuration': medianDuration,
            'timeBetweenSubmits': timeBetweenSubmits,
            'numOfExpertHits': numOfExpertHits,
            'exportVotesCorrect': votesCorrect,
            'votesChecked': numOfVotes
        })
    
    workerStats = pd.DataFrame(workerStats)
    workerStats = workerStats.sort_values(by='numOfAssignments', ascending=False)
    return workerStats
    
def retrieveResults():
    hitIds = [hit.getHitId() for hit in hits]
    assignments = getAssignments(hitIds)

    annotations = []
    for index, assignment in enumerate(assignments):
        annotations += extractAnnotations(assignment, hits[index])

    annotations = pd.DataFrame(annotations)
    # make sure error column exists for further transformations
    if ('error' not in annotations.columns):
        annotations = annotations.assign(error=np.nan)
    for answerColumn in answerColumns[-5:]:
        if (answerColumn not in annotations.columns):
            annotations = annotations.assign(**{
                answerColumn: np.nan
            })
    if ('agreesWithIssue' in answerColumns):
        annotations['agreesWithIssue'] = annotations['agreesWithIssue'] == 'true'
    annotations.to_csv('hit-output/{}.assignments.csv'.format(runName))

    #expertResults = pd.read_csv('hit-output/ukp-test-1.results.csv', index_col=0)
    expertResults = pd.DataFrame()
    workerStats = getWorkerStats(annotations, hits, expertResults)
    workerStats.to_csv('hit-output/{}.workers.csv'.format(runName))
    results = transformAnnotations(annotations, hits)
    if ('error' not in results.columns):
        results = results.assign(error=np.nan)
    results.to_csv('hit-output/{}.results.csv'.format(runName))

    numOfResults = len(results)
    results = results.loc[results['error'].isnull()]
    numOfResultsNoErrors = len(results)
    print('errors/problems:', numOfResults - numOfResultsNoErrors)
    results = results.loc[results['hitType'] == 'task']
    results = results.loc[results['assignments'] >= 2]

    print('assignments:', len(annotations))
    for scenario in results['scenario'].unique():
        df = results
        df = df.loc[df['scenario'] == scenario]
        print('===========================')
        print('scenario:', scenario)
        print('sample size:', len(df))
        for column in df.columns:
            if ('agree' not in column.lower()):
                continue
            
            print(column, 'mean:', df[column].mean())
        
retrieveResults()