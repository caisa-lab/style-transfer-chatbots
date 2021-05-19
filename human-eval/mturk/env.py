class MturkEnv:
    def __init__(self, name, endpoint, preview, manage, postAction):
        self.name = name
        self.endpoint = endpoint
        self.preview = preview
        self.manage = manage
        self.postAction = postAction

sandbox = MturkEnv(
    name='sandbox',
    endpoint='https://mturk-requester-sandbox.us-east-1.amazonaws.com',
    preview='https://workersandbox.mturk.com/mturk/preview',
    manage='https://requestersandbox.mturk.com/mturk/manageHITs',
    postAction='https://workersandbox.mturk.com/mturk/externalSubmit'
)

live = MturkEnv(
    name='live',
    endpoint='https://mturk-requester.us-east-1.amazonaws.com',
    preview='https://www.mturk.com/mturk/preview',
    manage='https://requester.mturk.com/mturk/manageHITs',
    postAction='https://www.mturk.com/mturk/externalSubmit'
)