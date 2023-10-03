def HillsTransformation(x, alpha: int=1, gamma:int=1): 
    return (x ** gamma) / (x ** gamma + alpha ** gamma)