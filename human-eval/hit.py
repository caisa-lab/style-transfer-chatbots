import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)

with open(os.path.join(dname, 'templates', 'question.xml')) as f:
    questionTemplate = f.read()

with open(os.path.join(dname, 'templates', 'hit', 'style', 'message.html')) as f:
    messageTemplate = f.read()


def replace(template, key, value, varBlackList=['captcha', 'expertHitId']) -> str:
    if (hasattr(value, 'keys') or key in varBlackList):
        return template
    return template.replace('%{}%'.format(key), str(value))

def replaceParameters(template: str, params: dict, varBlackList=['captcha', 'expertHitId']) -> str:
    result = template
    for key, value in params.items():
        result = replace(result, key, value, varBlackList)
    
    return result

def getTurnHtml(turnTemplate: str, messageTemplate: str, turnId, params: dict) -> str:
    tempParams = {
        'userMessage': '',
        'bot1Message': '',
        'bot2Message': '',
    }
    for key in params:
        value = params[key]
        if (key != 'userMessage'):
            botPrefix = key.lower().replace('message', '')
            value = getMessageHtml(messageTemplate, value, turnId, botPrefix)
        tempParams[key] = value

    return replaceParameters(turnTemplate, tempParams)

def getMessageHtml(messageTemplate: str, messageText, turnId, botPrefix):
    return replaceParameters(messageTemplate, {
        'messageText': messageText,
        'turnId': turnId,
        'prefix': botPrefix
    })

class Hit(object):
    def __init__(self, parameters):
        self.parameters = parameters
        self.amtResponse = { }

    def generateHtmlQuestion(self, htmlTemplate: str, turnTemplate: str):
        result = replace(questionTemplate, 'htmlContent', htmlTemplate)
        result = replaceParameters(result, self.parameters)
        turnsHtml = ''
        for turnId, params in sorted(self.parameters['turns'].items(), key=lambda x: int(x[0])):
            turnsHtml += getTurnHtml(turnTemplate, messageTemplate, turnId, params)
            turnsHtml += '\n'

        result = replace(result, 'turns', turnsHtml)
        return result
  
    def getHitId(self):
        return self.amtResponse['HITId']

    def getScenario(self):
        return self.parameters['scenario']

    def getHitType(self):
        if (self.isCaptcha()):
            return 'captcha'
        elif (self.isExpertHit()):
            return 'expert'
        else:
            return 'task'

    def isCaptcha(self):
        return 'captcha' in self.parameters

    def isExpertHit(self):
        return 'expertHitId' in self.parameters
    
    def getBotTurns(self, isConst: bool = None):
        results = {}
        for turnId, turn in self.parameters['turns'].items():
            if (turn['sender'] == 'bot'):
                # if == None, return any bot turn.
                # if isConst is bool, return only matching ones
                if ((isConst == None) or (turn['isConst'] == isConst)):
                    results[turnId] = turn
        
        return results