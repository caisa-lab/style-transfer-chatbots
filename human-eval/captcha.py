# ===========================================================
# Contains methods to generate new captchas based on the files in `template/captcha/`
# and to score workers' responses to captchas.
# ===========================================================

from hit import Hit
from copy import deepcopy
from collections import defaultdict
import random
import numpy as np
import pandas as pd

offensiveList = pd.read_csv('templates/offensive-sentences.csv', index_col=0)['summary'].tolist()

def getCaptcha(hit: Hit):
    """
    Generate a new captcha from an existing HIT object.
    """
    hit = Hit(deepcopy(hit.parameters))
    botTurns = hit.getBotTurns(isConst=False)
    # leave the greeting intact
    possibleTurns = set(botTurns.keys()) - set([0])
    replaceTurnId1 = random.choice(list(possibleTurns))
    possibleTurns -= set([replaceTurnId1])
    replaceTurnId2 = random.choice(list(possibleTurns))

    # replace bot messages
    def replaceTurn(hit, turnId, botPrefix):
        captchaMessage = random.choice(offensiveList)
        hit.parameters['turns'][turnId]['{}Message'.format(botPrefix)] = captchaMessage
        return captchaMessage
    
    replaceSender = random.choice(['bot1', 'bot2'])
    replacement1 = replaceTurn(hit, replaceTurnId1, replaceSender)
    replacement2 = replaceTurn(hit, replaceTurnId2, replaceSender)

    expectedFlags = ['offensive', 'incoherent']
    hit.parameters['captcha'] = {
        replaceTurnId1: {
            'replaceSender': replaceSender,
            'replaceMessage': replacement1,
            'expectedFlags': expectedFlags
        },
        replaceTurnId2: {
            'replaceSender': replaceSender,
            'replaceMessage': replacement2,
            'expectedFlags': expectedFlags
        }
    }

    return hit

def scoreCaptcha(captchaAssignment, hitIndex):  
    """
    Score answers for a captcha:
    Each correct option gives 1 point.
    Minumum score: 0 (all options wrong)
    Maximum score: 2 (all options correct)

    Parameters
    ----------
    captchaAssignment: Series (row)
        Single assignment for the captcha. Needs to contain `best` and `worst` fields (worker's answers).
    hitIndex: dict
        Lookup table for Hits based on Hit Id.
    
    Returns
    -------
    score: numeric    
    """
    row = captchaAssignment
    hit = hitIndex[row['hitId']]
    captcha = hit.parameters['captcha']
    correctAnswers = 0
    
    flags = defaultdict(dict)
    for prefix in ['bot1', 'bot2']:
        for flag in ['incoherent', 'offensive']:
            columnName = '{}{}[]'.format(prefix, flag.title())
            turnIds = []
            if ((columnName in row) and pd.notna(row[columnName])):
                turnIds = str(row[columnName]).split('|')
            flags[prefix][flag] = set(turnIds)

    for turnId, captchaInfo in captcha.items():
        prefix = captchaInfo['replaceSender']
        expectedFlags = captchaInfo['expectedFlags']
        for flag in expectedFlags:
            if (turnId in flags[prefix][flag]):
                # Not all replaceMessages are offensive and incoherent. one flag is enough
                correctAnswers += 1
                break
    
    return correctAnswers

def getOffensiveSentences():
    df = pd.read_csv('/data/daten/python/master/gyafc-classifier/offensive-reviews-roberta.csv', index_col=0)
    df = df.loc[df['summary'].str.len() >= 50]
    df = df['reviewerID,asin,reviewerName,summary'.split(',')]
    df.to_csv('templates/offensive-reviews.csv')
    offensiveList = df['summary'].tolist()
    offensiveList = [s.replace('...', '.') for s in offensiveList[:]]
    df = pd.DataFrame({ 'summary': offensiveList}).to_csv('templates/offensive-sentences.csv')
