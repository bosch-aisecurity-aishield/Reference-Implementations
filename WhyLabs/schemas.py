from pydantic import BaseModel


class FeatureVector_IRIS(BaseModel):
    sepal_length_cm: float
    sepal_width_cm: float
    petal_length_cm: float
    petal_width_cm: float



class FeatureVector_MNIST(BaseModel): #  add upto 28*28 features
    features : list 
 