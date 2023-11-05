
class PredictionException(Exception):
    def __init__(self, message):
        super().__init__(message)

class UserNotFoundException(Exception):
    def __init__(self, message):
        super().__init__(message)

