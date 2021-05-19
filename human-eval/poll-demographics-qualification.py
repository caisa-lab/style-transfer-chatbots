import json
import os
import pandas as pd
import mturk.env
from mturk.client import MturkClient, convertAnswer

with open('templates/qualification/demographics/quali-params.json') as f:
    qualiParams = json.load(f)

for mturkEnv in [mturk.env.sandbox, mturk.env.live]:
    client = MturkClient(mturkEnv=mturkEnv, confirmEnv=False)
    botoClient = client.client
    demoQualification = client.getQualificationType(qualiParams['name'])
    demoQualificationTypeId = demoQualification['QualificationTypeId']

    #print('Qualification Name:', qualiParams['name'])
    #print('Qualification Type Id:', demoQualificationTypeId)

    df = pd.DataFrame()
    dfPath = 'hit-output/demographics-{}.csv'.format(mturkEnv.name)
    if (os.path.isfile(dfPath)):
        df = pd.read_csv(dfPath, index_col=0)

    qualificationRequests = botoClient.list_qualification_requests(QualificationTypeId=demoQualificationTypeId)
    numOfRequests = qualificationRequests['NumResults']
    for request in qualificationRequests['QualificationRequests']:
        answer = convertAnswer(request['Answer'])
        requestCopy = dict(request)
        del requestCopy['Test']
        del requestCopy['Answer']
        del requestCopy['QualificationTypeId']
        requestCopy.update(answer)
        requestCopy['mturkEnv'] = mturkEnv.name

        # only save the latest entry for each worker.
        if (len(df) > 0):
            df = df.loc[df['WorkerId'] != requestCopy['WorkerId']]
        dfBackup = df
        currentDf = pd.DataFrame([requestCopy])
        df = pd.concat([dfBackup, currentDf], ignore_index=True).drop_duplicates()
        df.to_csv(dfPath)
        score = 100
        result = botoClient.accept_qualification_request(
            QualificationRequestId=requestCopy['QualificationRequestId'],
            IntegerValue=score
        )
        print('Granted qualification to worker', requestCopy['WorkerId'])