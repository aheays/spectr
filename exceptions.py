class InferException(Exception):
    """For failed inferences in Dataset."""
    pass

class MissingDataException(InferException):
    """Data not found in database."""
    pass             

class InvalidEncodingException(InferException):
    """Invalid encoding of level or tranistion name."""
    pass
