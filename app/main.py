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
from google.cloud import storage
from google.cloud.exceptions import NotFound
from google.api_core.exceptions import GoogleAPICallError
from fastapi import HTTPException


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


class StampParams(BaseModel):
    stamp: str
    delete: bool
    
    
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
        instance_name=f"seir-generator-{timestamp}",
        machine_type="e2-medium",
        image_family="debian-12",
        image_project="debian-cloud",
        docker_image="gcr.io/epistorm-gleam-api/seir",
        beta=params.beta,
        epsilon=params.epsilon,
        simulations=params.sims,
        days=params.days,
        bucket='seir-output-bucket-2',
        outfile=f'out-{timestamp}'
        )
    return timestamp

@app.post("/data")
def create_compute(params: StampParams, user: tuple[DocumentSnapshot, DocumentReference] = Depends(get_user)):
    try:
        client = storage.Client()
        bucket = client.bucket("seir-output-bucket-2")
    except NotFound:
        raise HTTPException(status_code=404, detail="Bucket not found.")
    except GoogleAPICallError as e:
        raise HTTPException(status_code=500, detail=f"Error accessing GCS: {str(e)}")

    try:
        blob = bucket.blob(f'out-{params.stamp}.json')

        if not blob.exists():
            raise HTTPException(status_code=404, detail=f"Data not found in the bucket.")
        
        content = blob.download_as_text()
        if params.delete:
            blob.delete()

        return content
    
    except GoogleAPICallError as e:
        raise HTTPException(status_code=500, detail=f"Error accessing blob: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}")

