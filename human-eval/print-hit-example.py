from hit import Hit
import os
import jsonpickle

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)

runName = 'phone-conversations-1-live'
with open('hit-data/{}.json'.format(runName), 'r') as f:
    hits = jsonpickle.decode(f.read(), keys=True, classes=Hit)

currentTask = 'style'
with open(os.path.join(dname, 'templates', 'hit', currentTask, 'hit.html'), 'r') as f:
    fileTemplate = f.read()
with open(os.path.join(dname, 'templates', 'hit', currentTask, 'turn.html'), 'r') as f:
    turnTemplate = f.read()
with open(os.path.join(dname, 'templates', 'question.xml')) as f:
    questionTemplate = f.read()


hit = hits[4]
html = hit.generateHtmlQuestion(fileTemplate, turnTemplate)
for part in questionTemplate.split('%htmlContent%'):
    html = html.replace(part, '')

with open('templates/hit/style/sample-phone.html', 'w') as f:
    f.write(html)

