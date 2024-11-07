import numpy as np
import json
from fastapi import Depends, FastAPI
from pydantic import BaseModel
from app.seir import seir, full_seir    
from os import environ
from app.auth import get_user

class Params(BaseModel):
    days: int
    sims: int
    beta: float
    epsilon: float
    
class ListParams(BaseModel):
    days: int
    sims: int
    beta_epsilon: list
    
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    
app = FastAPI()

@app.get("/")
def test(user: dict = Depends(get_user)):
    return user

@app.post("/original")
def original(params: Params, user:dict = Depends(get_user)):
    item_1, item_2, item_3 = seir(params.days,params.beta, params.epsilon,params.sims)
    json_dump = json.dumps({"train_set": item_1}, 
                       cls=NumpyEncoder)
    return json_dump

@app.post("/multiple")
def multiple(params: ListParams, user:dict = Depends(get_user)):
    beta_epsilon = np.array(params.beta_epsilon)
    
    item_1 = full_seir(params.days,beta_epsilon,params.sims)
    json_dump = json.dumps({"train_set": item_1}, 
                       cls=NumpyEncoder)
    return json_dump

