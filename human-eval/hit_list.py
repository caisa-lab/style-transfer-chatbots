from hit import Hit
import jsonpickle

def orderHitsByType(hits):
    captchas = []
    expertHits = []
    bwsQuestions = []

    for hit in hits:
        if (hit.isCaptcha()):
            captchas.append(hit)
        elif (hit.isExpertHit()):
            expertHits.append(hit)
        else:
            bwsQuestions.append(hit)
    
    return {
        'all': hits,
        'captcha': captchas,
        'expert': expertHits,
        'bws': bwsQuestions
    }

def orderHitsByTopic(hits):
    result = { }
    for hit in hits:
        topic = hit.parameters['topicName']
        if (not (topic in result.keys())):
            result[topic] = []
        result[topic].append(hit)
    
    return result

def orderHitsById(hits):
    result = { }
    for hit in hits:
        result[hit.getHitId()] = hit
    
    return result

def testHits():
    with open('data/batch-1-live.json', 'r') as f:
        hits = jsonpickle.decode(f.read(), keys=True, classes=Hit)
    
    hits = orderHitsByType(hits)
    hits = orderHitsByTopic(hits['bwsQuestions'])
    print(hits)
