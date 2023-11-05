#Exception 1
class PredictionException(Exception):
    def __init__(self, name: str):
        self.name = "Error in prediction: "

#Exception 2
class UserNotFoundException(Exception):
    def __init__(self, name = "User Not Found:"):
        self.name = name

