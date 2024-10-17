import numpy as np
import json
from fastapi import FastAPI
from pydantic import BaseModel
from seir import seir

app = FastAPI()

class Params(BaseModel):
    days: int
    sims: int
    beta: float
    epsilon: float
    
# class Data(BaseModel):
#     train_set: np.array
#     train_meanset: np.array
#     train_stdset: np.array
    
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

@app.get("/")
def root(params: Params):
    item_1, item_2, item_3 = seir(params.days,params.beta, params.epsilon,params.sims)
    json_dump = json.dumps({"train_set": item_1,'train_meanset':item_2,'train_stdset':item_3}, 
                       cls=NumpyEncoder)
    return json_dump

