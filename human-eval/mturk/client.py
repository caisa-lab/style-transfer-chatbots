import boto3
from botocore.config import Config
import mturk.env as env
import jsonpickle
import sys
import os
from hashlib import sha256
import xmltodict

class MturkClient:
    def __init__(self, mturkEnv: env.MturkEnv = env.sandbox, confirmEnv: bool = True, profileName: str = None):
        self.session = boto3.Session(profile_name=profileName)
        self.client = self.session.client(
            service_name='mturk',
            region_name='us-east-1',
            endpoint_url=mturkEnv.endpoint,
            config=Config(connect_timeout=5, read_timeout=60, retries={'max_attempts': 20})
        )
        
        self.mturkEnv = mturkEnv
        print('Selected mturk environment:', mturkEnv.name)
        if (confirmEnv == True):
            print('Press {Enter} to continue...')
            sys.stdin.readline()
        
        userBalance = self.getAccountBalance()
        print("Your account balance is {}".format(userBalance['AvailableBalance']))

    def getAccountBalance(self):
        return self.client.get_account_balance()

    def getWorkerLink(self, hitTypeId: str):
        return self.mturkEnv.preview + "?groupId={}".format(hitTypeId)

    def createHits(self, questions, maxAssignments: int, lifetimeInSeconds: int, assignmentDurationInSeconds: int,
        reward: str, title: str, keywords: str, description: str, qualificationRequirements, 
        autoApprovalDelayInSeconds: int = 24*60*60, requesterAnnotations=None, savePath: str = None):
        """
        Create a number of hits with the given mturk parameters.
        If savePath is set, the responses will be saved as json in that file.

        Parameters
        ----------
        questions: list of str
            List of questions in the correct format (XML) as described in the mturk documentation
        maxAssignments: int
            Number of times each single hit will be assigned to workers
        lifetimeInSeconds:
            Duration the new hits will be assignable to workers
        assignmentDurationInSeconds:
            Maximum time a worker can take to complete the assignment
        reward: str
            Reward money for each single hit in $ (USD)
        qualificationRequirements: list of dict
            List of requirements each worker has to fulfill to be able to work on this hit as described in the mturk docs
        requesterAnnotations: list of str
            List of requester annotations for each hit, only visible for you (the requester)
        savePath: str
            Set this path to save the responses from mturk for each HIT in the given file path as json.

        Returns
        -------
        responses: list of dict
            Responses from mturk
        """
        if (savePath != None):
            tempDir = createDirs(os.path.dirname(savePath) + '/temp/')
            print('Will save temporary response files in:', tempDir)
        
        response = self.client.create_hit_type(
            AutoApprovalDelayInSeconds=autoApprovalDelayInSeconds,
            AssignmentDurationInSeconds=assignmentDurationInSeconds,
            Reward=reward, Title=title, Keywords=keywords, Description=description,
            QualificationRequirements=qualificationRequirements
        )

        hitTypeId = response['HITTypeId']
        responses = []
        numOfQuestions = len(questions)
        for i, question in enumerate(questions):
            if (requesterAnnotations == None):
                requesterAnnotation = ''
            else:
                requesterAnnotation = requesterAnnotations[i]
            
            response = self.client.create_hit_with_hit_type(
                HITTypeId=hitTypeId,
                MaxAssignments=maxAssignments,
                LifetimeInSeconds=lifetimeInSeconds,
                Question=question,
                RequesterAnnotation=requesterAnnotation,
                UniqueRequestToken=sha256((title + question).encode('utf8')).hexdigest()
            )
            response = response['HIT']
            # delete html question to reduce file size. We can always recreate the html question with the parameters.
            del response['Question']
            responses.append(response)
            
            hitId = response['HITId']
            print('Created HIT {0}/{1}:'.format(i + 1, numOfQuestions), hitId)
            if (savePath != None):
                with open('{0}/{1}.json'.format(tempDir, hitId), 'w') as f:
                    f.write(jsonpickle.encode(response))

        print('Created {} HITs.'.format(len(responses)))
        if (savePath != None):
            with open(savePath, 'w') as f:
                f.write(jsonpickle.encode(responses))
            print('Saved raw mturk responses in', savePath)
            for response in responses:
                hitId = response['HITId']
                os.remove('{0}/{1}.json'.format(tempDir, hitId))
            print('Removed temporary files.')  

        workerLink = self.getWorkerLink(hitTypeId)
        print("You can work the HIT here:")
        print(workerLink)
        print("And see results here:")
        print(self.mturkEnv.manage)
        return responses
    
    def listAssignmentsForHits(self, hitIds, assignmentStatuses=['Submitted', 'Approved'], convertAnswers: bool = True, savePath=None):
        """
        List all assignments for the given hit ids. You can filter results by setting assignmentStatuses.
        
        Parameters
        ----------
        hitIds: list of str
            List of HIT ids to retrieve assignments
        assignmentStatuses: list of str
            Filters assignments by status in this list: ['Submitted', 'Approved', 'Rejected']
        convertAnswers: bool
            If set to True, then the answers will be converted from XML to key-value dict. Accessible in assignment['AnswerDict']
        savePath: str
            Set this path to save the responses from mturk for each HIT in the given file path as json.

        Returns
        -------
        responses: list of dict
            List of responses from mturk (all assignments for the given HIT ids)
        """
        responses = []
        numOfHits = len(hitIds)
        for i, hitId in enumerate(hitIds):
            response = self.client.list_assignments_for_hit(
                HITId=hitId,
                MaxResults=50,
                AssignmentStatuses=assignmentStatuses
            )
            
            print('HIT {0}/{1}:'.format(i + 1, numOfHits), hitId)
            print('Assignments:', response['NumResults'])
            response = response['Assignments']
            for assignment in response:
                answerDict = convertAnswer(assignment['Answer'])
                assignment['AnswerDict'] = answerDict
            responses.append(response)

            if (savePath != None):
                with open(savePath, 'w') as f:
                    f.write(jsonpickle.encode(responses))
        
        return responses
    
    def createQualificationType(self, name: str, description: str, keywords: str, 
        test: str = None, answerKey: str = '', testDurationInSeconds: int = None, 
        retryDelayInSeconds: int = None, qualificationTypeStatus='Active', updateIfExists=True):
        try:
            # try to update if qualification exists
            qualification = self.getQualificationType(name)
            callParams = {}
            callParams.update(
                QualificationTypeId=qualification['QualificationTypeId'],
                RetryDelayInSeconds=retryDelayInSeconds,
                QualificationTypeStatus=qualificationTypeStatus,
                Description=description,
                Test=test,
                TestDurationInSeconds=testDurationInSeconds
            )
            if (answerKey != '' and answerKey != None):
                callParams['AnswerKey'] = answerKey
            result = self.client.update_qualification_type(**callParams)
        except ValueError as ex:
            # create new qualification
            callParams = {}
            callParams.update(
                Name=name, Description=description, Keywords=keywords,
                RetryDelayInSeconds=retryDelayInSeconds,
                QualificationTypeStatus=qualificationTypeStatus,
                Test=test,  
                TestDurationInSeconds=testDurationInSeconds
            )
            if (answerKey != '' and answerKey != None):
                callParams['AnswerKey'] = answerKey
            result = self.client.create_qualification_type(**callParams)
        
        result = result['QualificationType']        
        return result
    
    def rejectAssignments(self, assignmentIds, feedback):
        length = len(assignmentIds)
        for index, assignmentId in enumerate(assignmentIds):
            print('Rejecting assignment {}/{}: {}'.format(index + 1, length, assignmentId))
            self.client.reject_assignment(
                AssignmentId=assignmentId,
                RequesterFeedback=feedback
            )

    def createAdditionalAssignmentsForHit(self, hitId: str, addAssignments: int):
        result = self.client.create_additional_assignments_for_hit(
            HITId=hitId, NumberOfAdditionalAssignments=addAssignments
        )
        #result = result['']

        return result

    def getQualificationType(self, name: str):
        results = self.client.list_qualification_types(Query=name, MustBeRequestable=True)
        if (results['NumResults'] <= 0):
            raise ValueError('No qualifications found. Query: {}'.format(name))
        
        for result in results['QualificationTypes']:
            if (result['Name'] == name):
                return result
        # else
        raise ValueError('Qualification with name {} not found'.format(name))

def convertAnswer(xml: str):
    xmlAnswer = xmltodict.parse(xml)
    answers = { }
    for field in xmlAnswer['QuestionFormAnswers']['Answer']:
        answerKey = ''
        for key in field.keys():
            if (key != 'QuestionIdentifier'):
                answerKey = key
                break
        answers[field['QuestionIdentifier']] = field[answerKey]

    return answers

def createDirs(filePath: str):
    """Creates all missing directories for the given file path."""
    dirName = os.path.dirname(filePath)
    if not os.path.exists(dirName):
        os.makedirs(dirName)

    return dirName