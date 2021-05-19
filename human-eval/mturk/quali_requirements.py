import mturk.env as env
from enum import auto as Enum

class ActionsGuarded(Enum):
    Accept = 'Accept'
    PreviewAndAccept = 'PreviewAndAccept'
    DiscoverPreviewAndAccept = 'DiscoverPreviewAndAccept'

class Comparator(Enum):
    LessThan = 'LessThan'
    LessThanOrEqualTo = 'LessThanOrEqualTo'
    GreaterThan = 'GreaterThan'
    GreaterThanOrEqualTo = 'GreaterThanOrEqualTo'
    EqualTo = 'EqualTo'
    NotEqualTo = 'NotEqualTo'
    Exists = 'Exists'
    DoesNotExist = 'DoesNotExist'
    In = 'In'
    NotIn = 'NotIn'

def WorkerLocale(countryCodes):
    qualiTypeId = '00000000000000000071'
    locales = []
    for country in countryCodes:
        locales.append({
            'Country': country
        })
    return generateRequirement(qualiTypeId, comparator=Comparator.In, localeValues=locales)

def Masters(mturkEnv: env.MturkEnv, actionsGuarded=ActionsGuarded.Accept):
    if (mturkEnv.name == 'sandbox'):
        qualiTypeId = '2ARFPLSP75KLA8M8DH1HTEQVJT3SY6'
    else:
        # production
        qualiTypeId = '2F1QJWKUDD8XADTFD2Q0G6UTO95ALH'
    
    return generateRequirement(qualiTypeId, comparator=Comparator.Exists, actionsGuarded=actionsGuarded)

def WorkerNumberHitsApproved(numOfHits: int, comparator=Comparator.GreaterThanOrEqualTo, actionsGuarded=ActionsGuarded.Accept):
    qualiTypeId = '00000000000000000040'
    return generateRequirement(qualiTypeId, comparator=comparator, integerValues=(numOfHits,), actionsGuarded=actionsGuarded)

def WorkerPercentAssignmentsApproved(approvedPercent: int, comparator=Comparator.GreaterThanOrEqualTo, actionsGuarded=ActionsGuarded.Accept):
    qualiTypeId = '000000000000000000L0'
    return generateRequirement(qualiTypeId, comparator=comparator, integerValues=(approvedPercent,), actionsGuarded=actionsGuarded)
    

def generateRequirement(qualiTypeId, comparator, integerValues=None, localeValues=None, actionsGuarded=ActionsGuarded.Accept):
    requirement = {
        'QualificationTypeId': qualiTypeId,
        'Comparator': comparator,
        'ActionsGuarded': actionsGuarded
    }

    if (integerValues != None):
        requirement['IntegerValues'] = integerValues
    if(localeValues != None):
        requirement['LocaleValues'] = localeValues

    return requirement