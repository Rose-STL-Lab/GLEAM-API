import numpy as np
import json
from fastapi import Depends, FastAPI
from pydantic import BaseModel
from app.seir import seir, full_seir    
from os import environ
from app.auth import get_user
from google.cloud.firestore_v1 import DocumentReference, DocumentSnapshot
from app.compute import create_instance_with_docker
import time
from app.dcrnn_model.dcrnn import DCRNNModel
import torch

class Params(BaseModel):
    days: int
    sims: int
    beta: float
    epsilon: float
    
class ListParams(BaseModel):
    days: int
    sims: int
    beta_epsilon: list

class ComputeParams(BaseModel):
    days: int
    sims: int
    beta: float
    epsilon: float
    
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    
app = FastAPI()

@app.get("/")
def test(user: tuple[DocumentSnapshot, DocumentReference] = Depends(get_user)):
    num_runs = user[0].to_dict()['Num_Runs_This_Month']
    user[1].update({'Num_Runs_This_Month': num_runs + 1})
    return user[0].to_dict()

@app.post("/original")
def original(params: Params, user: tuple[DocumentSnapshot, DocumentReference] = Depends(get_user)):
    item_1, item_2, item_3 = seir(params.days,params.beta, params.epsilon,params.sims)
    json_dump = json.dumps({"train_set": item_1}, 
                       cls=NumpyEncoder)
    return json_dump

@app.post("/multiple")
def multiple(params: ListParams, user: tuple[DocumentSnapshot, DocumentReference] = Depends(get_user)):
    beta_epsilon = np.array(params.beta_epsilon)
    
    item_1 = full_seir(params.days,beta_epsilon,params.sims)
    json_dump = json.dumps({"train_set": item_1}, 
                       cls=NumpyEncoder)
    return json_dump

@app.post("/create_compute")
def create_compute(params: Params, user: tuple[DocumentSnapshot, DocumentReference] = Depends(get_user)):
    timestamp = str(int(time.time()))
    output = create_instance_with_docker(
        project_id="epistorm-gleam-api",
        zone="us-central1-a",
        instance_name=f"my-docker-vm8-{timestamp}",
        machine_type="e2-medium",
        image_family="debian-12",
        image_project="debian-cloud",
        docker_image="gcr.io/epistorm-gleam-api/seir",
        beta=params.beta,
        epsilon=params.epsilon,
        simulations=params.sims,
        days=params.days,
        bucket='seir-output-bucket-2'
        )
    return timestamp

@app.post("/stnp_model")
def stnp_model(params: Params, user: tuple[DocumentSnapshot, DocumentReference] = Depends(get_user)):
    print("started")
    device = torch.device("cpu")
    model = DCRNNModel(x_dim=2,y_dim=100,r_dim=8,z_dim=8,device=device)
    print("model created")
    model.load_state_dict(torch.load('app/dcrnn_model/weights_batch_size_1.pth'))
    print("model loaded")
    zs = torch.load("app/dcrnn_model/zs_batch_size_1.pth",weights_only=False)
    output = model.decoder(torch.tensor(np.array([[params.beta,params.epsilon]])).float(),zs).detach().numpy()
    json_dump = json.dumps({"train_set": output}, 
                       cls=NumpyEncoder)
    return json_dump
    