import numpy as np
import json
from fastapi import Depends, FastAPI, BackgroundTasks
from pydantic import BaseModel
from app.seir import seir, full_seir    
from os import environ
from app.auth import get_user
from google.cloud.firestore_v1 import DocumentReference, DocumentSnapshot
from app.compute import create_instance_with_docker, create_dummy_instance, estimate_instance_cost
from app.compute import create_instance_with_image
from app.compute import create_instance_and_save_image
from app.compute import upload_yaml_to_gcs
from app.compute import create_instance_with_image_config
from app.compute import julia_create_instance_and_save_image
import time
from app.dcrnn_model.dcrnn import DCRNNModel
import torch
from google.cloud import storage
from google.cloud.exceptions import NotFound
from google.api_core.exceptions import GoogleAPICallError
from fastapi import HTTPException
from app.gleam_ml.dcrnn_supervisor import DCRNNSupervisor
import yaml
from app.gleam_ml.lib.utils import load_graph_data
from app.gleam_ml.lib import utils
from typing import Dict
import io
import zipfile
from fastapi.responses import StreamingResponse
from google.cloud import storage

storage_client = storage.Client()
bucket_name = "seir-output-bucket-2"

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

class JuliaImageParams(BaseModel):
    bucket_name: str
    folder_name: str
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
    folder: str
    delete: bool

class ConfigParams(BaseModel):
    json_object: dict 

class ComputeWithConfig(BaseModel):
    config_file: str
    image: str
    script_location: str
    
    
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

@app.post("/estimate_cost")
def estimate_cost(user: tuple[DocumentSnapshot, DocumentReference] = Depends(get_user)):
    print("called api")
    sku = estimate_instance_cost("e2-medium", 1)
    return sku

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
        timestamp = timestamp
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

@app.post("/gleam_ml")
def gleam_ml(params: Params):
    print("started gleam")
    model_tar = torch.load("app/gleam_ml/model_epo101.tar")
    with open("app/gleam_ml/dcrnn_cov.yaml") as f:
        supervisor_config = yaml.safe_load(f)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)
        i=1
        max_itr = 12 #12
        data, search_data_x, search_data_y = utils.load_dataset(**supervisor_config.get('data'))
        supervisor = DCRNNSupervisor(random_seed=i, iteration=11, max_itr = max_itr, 
                adj_mx=adj_mx, **model_tar)
        
    supervisor.epoch_num = 101
    supervisor.load_model()
    supervisor._data = data
    model = supervisor.dcrnn_model
    val_iterator = supervisor._data['{}_loader'.format('val')].get_iterator()

    for _, (x, y, x0) in enumerate(val_iterator):
        x, y, x0 = supervisor._prepare_data(x, y, x0)
        
        output,_ = model(x, y, x0, None, True, supervisor.z_mean_all, supervisor.z_var_temp_all)
        
        break
    output = output.detach().numpy()
    return json.dumps({"model_pred": output}, 
                       cls=NumpyEncoder)    
    # temp_data = np.load('app/gleam_ml/data/data/val.npz')
    # x,y,x0 = supervisor._prepare_data(temp_data['x'],temp_data['y'],None)
    # model(x
    #       ,y,None,None,test=True,z_mean_all=supervisor.z_mean_all,z_var_temp_all=supervisor.z_var_temp_all)

    
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
        prefix = f"{params.stamp}/"
        blobs = bucket.list_blobs(prefix=prefix) 
        
        content: Dict[str, str] = {}
        found_files = False
        
        for blob in blobs:
            found_files = True
            blob_name = blob.name
            blob_content = blob.download_as_text()
            content[blob_name] = blob_content
            
            if params.delete:
                blob.delete()
        
        if not found_files:
            raise HTTPException(status_code=404, detail=f"No data found in folder: {prefix}")
        
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
def create_image(params: CreateImageParams,
                 background_tasks: BackgroundTasks, 
    user: tuple[DocumentSnapshot, DocumentReference] = Depends(get_user)):

    timestamp = str(int(time.time()))


    background_tasks.add_task(
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
        script_location=params.script_location,
        bucket='seir-output-bucket-2',
        outfile=f'out-{timestamp}',
        config= params.config_file
        )
    return timestamp


@app.post("/julia_create_image")
def julia_create_image(params: JuliaImageParams,
                 background_tasks: BackgroundTasks, 
    user: tuple[DocumentSnapshot, DocumentReference] = Depends(get_user)):

    timestamp = str(int(time.time()))


    background_tasks.add_task(
        julia_create_instance_and_save_image,
        project_id="epistorm-gleam-api",
        zone="us-central1-a",
        instance_name=f"image-generator-{timestamp}",
        machine_type="e2-medium",
        image_family="debian-12",
        image_project="debian-cloud",
        bucket_name=params.bucket_name,
        folder_name=params.folder_name,
        custom_image_name=params.image_name + "-" + timestamp,
    )
    return params.image_name + "-"+ timestamp


@app.get("/download-folder")
async def download_folder(params: StampParams):
    try:
        bucket = storage_client.bucket(bucket_name)
        prefix = params.folder 
        blobs = list(bucket.list_blobs(prefix=prefix))

        if not blobs:
            raise HTTPException(status_code=404, detail=f"No files found in folder: {prefix}")

        zip_stream = io.BytesIO()

        with zipfile.ZipFile(zip_stream, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for blob in blobs:
                file_data = blob.download_as_bytes()
                zip_file.writestr(blob.name[len(prefix):], file_data) 

        zip_stream.seek(0)

        return StreamingResponse(zip_stream, media_type="application/zip",
                                 headers={"Content-Disposition": f"attachment; filename={prefix}.zip"})
    
    except NotFound:
        raise HTTPException(status_code=404, detail="Bucket not found.")
    except GoogleAPICallError as e:
        raise HTTPException(status_code=500, detail=f"Error accessing GCS: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))