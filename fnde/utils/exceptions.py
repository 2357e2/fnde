class ParameterError(Exception):
    def __init__(self, parameter, value):
        super().__init__(f"ParameterError: parameter {parameter} given invalid value {value}")