import pandas as pd

scenario = 'phone'
df = pd.read_csv('paraphrases/{}-conversations-1/input.csv'.format(scenario), index_col=0)

convId = 863
df = df.loc[df['conversationId'] == convId]
outDf = []
for idx, row in df.iterrows():
    sender = 'Chatbot:' if row['sender'] == 'bot' else 'User:'
    outDf.append({
        'Sender': sender,
        'Message': row['text']
    })

outDf = pd.DataFrame(outDf)
outDf.to_csv('conv-sample.csv')
