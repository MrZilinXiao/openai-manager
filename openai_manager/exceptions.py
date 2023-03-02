class ManagerException(Exception):
    def __init__(self, message) -> None:
        super().__init__(message)


class NoAvailableAuthException(ManagerException):
    pass


class AttemptsExhaustedException(ManagerException):
    pass
