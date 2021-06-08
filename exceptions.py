class InferException(Exception):
    """For failed inferences in Dataset."""
    pass

class DatabaseException(InferException):
    """Data not found in database or some other problem."""
    pass             

class InvalidEncodingException(InferException):
    """Invalid encoding of level or transition name."""
    pass
