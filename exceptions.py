class BaseError(Exception):
    def __init__(self, message="Error"):
        self.message = message
        super().__init__(self.message)


class ValidationError(BaseError):
    def __init__(self, message="Validation error"):
        self.message = message
        super().__init__(message=self.message)
