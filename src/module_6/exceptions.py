
class PredictionException(Exception):
    def __init__(self, name: str):
        self.name = "Error in prediction: "


class UserNotFoundException(Exception):
    def __init__(self, name = "User Not Found:"):
        self.name = name

