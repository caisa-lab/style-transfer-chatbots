import pandas as pd
import random

df = pd.read_csv('short_candidates.csv', index_col=0)

def replaceWhitespace(text):
    text = text.replace('\\n', ' ')
    return ' '.join(text.split())

results = []
conversationId = 1000
for idx, row in df.iterrows():
    newDf = []
    issueText = str(row['issue'])
    issue = ''
    if (' police ' in issueText):
        issue = 'police brutality'
    elif (' death penalty ' in issueText):
        issue = 'the death penalty'
    elif (' unions ' in issueText):
        issue = 'public sector unions'
    else:
        raise ValueError('Unknown issue: ' + issueText)
    
    hiList = ['Hi', 'Hello', 'Hey']
    botGreetings = [
        '{}, what do you think about {}?',
        '{}, what is your opinion on {}?'
    ]
    hi = random.choice(hiList)
    botGreeting = random.choice(botGreetings)
    botGreeting = botGreeting.format(hi, issue)

    newDf.append({
        'text': botGreeting,
        'sender': 'bot',
        'isConst': False
    })

    text = replaceWhitespace(issueText)
    text += ' Do you agree?'
    sender = 'user'
    isConst = True
    newDf.append({
        'text': text,
        'sender': sender,
        'isConst': isConst
    })

    for column in '"round_1","round_2_part","round_3"'.split(','):
        column = column.replace('"', '')
        text = replaceWhitespace(row[column])
        sender = 'user' if sender == 'bot' else 'bot'
        isConst = True if sender == 'user' else False

        newDf.append({
            'text': text,
            'sender': sender,
            'isConst': isConst
        })
    
    newDf = pd.DataFrame(newDf)
    newDf = newDf.assign(
        turnId=list(range(len(newDf))),
        scenario='debate',
        targetStyle='original',
        conversationId=conversationId
    )
    results.append(newDf)
    conversationId += 1

df = pd.concat(random.sample(results, 50), ignore_index=True)
df.to_csv('debate-conversations.csv')