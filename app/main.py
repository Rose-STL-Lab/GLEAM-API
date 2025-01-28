import numpy as np
import json
from fastapi import Depends, FastAPI, BackgroundTasks
from pydantic import BaseModel
from app.seir import seir, full_seir    
from os import environ
from app.auth import get_user
from google.cloud.firestore_v1 import DocumentReference, DocumentSnapshot
from app.compute import create_instance_with_docker, create_dummy_instance
from app.compute import create_instance_with_image
from app.compute import create_instance_and_save_image
from app.compute import upload_yaml_to_gcs
from app.compute import create_instance_with_image_config
import time
from app.dcrnn_model.dcrnn import DCRNNModel
import torch
from google.cloud import storage
from google.cloud.exceptions import NotFound
from google.api_core.exceptions import GoogleAPICallError
from fastapi import HTTPException


class Params(BaseModel):
    days: int
    sims: int
    beta: float
    epsilon: float

class StressTestParams(BaseModel):
    cpu: int
    io: int
    vm: int
    vm_bytes: str
    timeout: str

class ComputeImageParams(BaseModel):
    days: int
    sims: int
    beta: float
    epsilon: float
    image: str

class CreateImageParams(BaseModel):
    bucket_name: str
    folder_name: str
    requirements_name: str
    image_name: str

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

class ConfigParams(BaseModel):
    json_object: dict 

class ComputeWithConfig(BaseModel):
    config_file: str
    image: str
    
    
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

@app.post("/create_dummy_compute")
def create_compute(params: StressTestParams, user: tuple[DocumentSnapshot, DocumentReference] = Depends(get_user)):

    timestamp = str(int(time.time()))
    
    output = create_dummy_instance(
        project_id="epistorm-gleam-api",
        zone="us-central1-a",
        instance_name=f"stress-test-{timestamp}",
        machine_type="e2-medium",
        cpu= params.cpu,
        io= params.io,
        vm= params.vm,
        vm_bytes= params.vm_bytes,
        timeout= params.timeout
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
    
@app.post("/data")
def get_data(params: StampParams, user: tuple[DocumentSnapshot, DocumentReference] = Depends(get_user)):
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
    

@app.post("/create_compute_with_image")
def create_compute_with_image(params: ComputeImageParams, user: tuple[DocumentSnapshot, DocumentReference] = Depends(get_user)):

    timestamp = str(int(time.time()))


    output = create_instance_with_image(
        project_id="epistorm-gleam-api",
        zone="us-central1-a",
        instance_name=f"seir-generator-{timestamp}",
        machine_type="e2-medium",
        source_image= f"projects/epistorm-gleam-api/global/images/{params.image}",
        beta=params.beta,
        epsilon=params.epsilon,
        simulations=params.sims,
        days=params.days,
        bucket='seir-output-bucket-2',
        outfile=f'out-{timestamp}'
        )
    return timestamp



@app.post("/create_image")
def create_image(params: CreateImageParams, user: tuple[DocumentSnapshot, DocumentReference] = Depends(get_user)):

    timestamp = str(int(time.time()))


    BackgroundTasks.add_task(
        create_instance_and_save_image,
        project_id="epistorm-gleam-api",
        zone="us-central1-a",
        instance_name=f"image-generator-{timestamp}",
        machine_type="e2-medium",
        image_family="debian-12",
        image_project="debian-cloud",
        bucket_name=params.bucket_name,
        folder_name=params.folder_name,
        requirements_name=params.requirements_name,
        custom_image_name=params.image_name + "-" + timestamp,
    )
    return params.image_name + "-"+ timestamp

@app.post("/create_yaml")
def create_image(params: ConfigParams, user: tuple[DocumentSnapshot, DocumentReference] = Depends(get_user)):

    timestamp = str(int(time.time()))


    upload_yaml_to_gcs(params.json_object, "testscriptholder", f"config{timestamp}.yaml")
    return f"config{timestamp}.yaml"


@app.post("/compute_with_config")
def create_image(params: ComputeWithConfig, user: tuple[DocumentSnapshot, DocumentReference] = Depends(get_user)):

    timestamp = str(int(time.time()))
    
    output = create_instance_with_image_config(
        project_id="epistorm-gleam-api",
        zone="us-central1-a",
        instance_name=f"seir-generator-{timestamp}",
        machine_type="e2-medium",
        source_image= f"projects/epistorm-gleam-api/global/images/{params.image}",
        bucket='seir-output-bucket-2',
        outfile=f'out-{timestamp}',
        config= params.config_file
        )
    return timestamp


