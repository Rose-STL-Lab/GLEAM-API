import numpy as np
import json
from app.gleam_ml.stnp_supervisor import STNPSupervisor
from fastapi import Depends, FastAPI, BackgroundTasks, Request
from pydantic import BaseModel
from app.auth import get_user
from google.cloud.firestore_v1 import DocumentReference, DocumentSnapshot
from app.compute import  create_dummy_instance, estimate_instance_cost, create_instance_with_image, create_instance_and_save_image, upload_yaml_to_gcs
import time
import torch
from google.cloud import storage
from fastapi import HTTPException
import yaml
from app.gleam_ml.lib.utils import load_graph_data
import io
import zipfile
from google.cloud import storage
from google.oauth2 import service_account
from datetime import timedelta
import tempfile

import gc


def download_service_account_key(bucket_name: str, blob_name: str) -> str:
    """Downloads the service account key from Cloud Storage and returns the file path."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    blob.download_to_filename(temp_file.name)

    return temp_file.name 
### This is for signed urls
BUCKET_NAME = "liamsjuliabucket"
SERVICE_ACCOUNT_FILENAME = "epistorm-gleam-api-612347bc95a6.json"
service_account_path = download_service_account_key(BUCKET_NAME, SERVICE_ACCOUNT_FILENAME)

credentials = service_account.Credentials.from_service_account_file(service_account_path)
# credentials = service_account.Credentials.from_service_account_file(
#     "C:/Users/00011/Downloads/epistorm-gleam-api-612347bc95a6.json"
# )

storage_client = storage.Client(credentials=credentials)
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

class GleamParams(BaseModel):
    x: list
    x0: list

class CreateImageParams(BaseModel):
    bucket_name: str
    folder_name: str
    requirements_name: str
    image_name: str

class StampParams(BaseModel):
    folder: str
    delete: bool

class ConfigParams(BaseModel):
    json_object: dict 

class EstimateCostParams(BaseModel):
    num_gpu: int
    num_cpu: int
    num_ram: int
    hours: int
    
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    
app = FastAPI()


@app.middleware("http")
async def collect_garbage(request: Request, call_next):
    response = await call_next(request)
    gc.collect()
    return response


@app.get("/")
def test(user: tuple[DocumentSnapshot, DocumentReference] = Depends(get_user)):
    return {"API Status": "Online"}

@app.post("/estimate_cost")
def estimate_cost(params: EstimateCostParams, user: tuple[DocumentSnapshot, DocumentReference] = Depends(get_user)):
    user_info = user[0].to_dict()
    num_runs = user_info['Num_Runs_This_Month']
    cost_this_month = user_info['Cost_This_Month']
    cost_limit = user_info['Limit_Cost_Per_Month']
    estimate = estimate_instance_cost(params.num_gpu,params.num_cpu,params.num_ram,params.hours)
    if estimate + cost_this_month > cost_limit:
        return f'User has exceeded their monthly cost limit of ${cost_limit}'
    user[1].update({'Num_Runs_This_Month': num_runs + 1,'Cost_This_Month': cost_this_month + estimate})
    return {"Estimate": estimate}

@app.post("/gleam_simulation")
def create_compute(params: StressTestParams, user: tuple[DocumentSnapshot, DocumentReference] = Depends(get_user)):

    timestamp = str(int(time.time()))
    user_info = user[0].to_dict()
    num_runs = user_info['Num_Runs_This_Month']
    cost_this_month = user_info['Cost_This_Month']
    cost_limit = user_info['Limit_Cost_Per_Month']
    estimate = estimate_instance_cost(2,1,4,1)
    if estimate + cost_this_month > cost_limit:
        return f'User has exceeded their monthly cost limit of ${cost_limit}'
    
    output = create_dummy_instance(
        project_id="epistorm-gleam-api",
        zone="us-central1-a",
        instance_name=f"stress-test-{timestamp}",
        machine_type="e2-medium",
        cpu= params.cpu,
        io= params.io,
        vm= params.vm,
        vm_bytes= params.vm_bytes,
        timeout= params.timeout,
        timestamp = timestamp
        )
    user[1].update({'Num_Runs_This_Month': num_runs + 1,'Cost_This_Month': cost_this_month + estimate})
    return timestamp

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

def zip_and_upload(folder_name: str, zip_blob_name: str):
    bucket = storage_client.bucket(bucket_name)
    
    blobs = list(bucket.list_blobs(prefix=folder_name))
    if not blobs:
        raise HTTPException(status_code=404, detail="No files found in the folder.")

    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for blob in blobs:
            file_data = blob.download_as_bytes()
            file_name = blob.name[len(folder_name):]
            zipf.writestr(file_name, file_data)

    zip_buffer.seek(0)

    zip_blob = bucket.blob(zip_blob_name)

    zip_blob.chunk_size = 5 * 1024 * 1024 
    zip_blob.upload_from_file(zip_buffer, content_type="application/zip")

    return f"gs://{bucket_name}/{zip_blob_name}"


def generate_signed_url(blob_name: str, expiration_minutes=15):
    """Generates a signed URL for a GCS object."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    url = blob.generate_signed_url(
        expiration=timedelta(minutes=expiration_minutes), 
        version="v4",
    )
    return url


@app.get("/download-folder/")
async def download_folder(folder_name: str):
    """Creates a ZIP file of a GCS folder and returns a signed URL for download."""
    try:
        zip_blob_name = f"{folder_name}.zip" 
        # zip_gcs_path = zip_and_upload(folder_name, zip_blob_name)
        zip_gcs_path = f"gs://{bucket_name}/{zip_blob_name}"
        
        signed_url = generate_signed_url(zip_blob_name)
        return {"download_url": signed_url, "gcs_path": zip_gcs_path}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/prediction")
def gleam_ml(params: GleamParams):
    print("started gleam")

    x = np.array(params.x)
    x0 = np.array(params.x0)
    inputer = len(x)
    
    model_tar = torch.load("app/gleam_ml/model_epo101.tar")
    with open("app/gleam_ml/dcrnn_cov.yaml") as f:
        supervisor_config = yaml.safe_load(f)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)
    
    print("loaded graph data")
    i=1
    max_itr = 12 #12
    supervisor = STNPSupervisor(random_seed=i, iteration=101, max_itr = max_itr, 
            adj_mx=adj_mx, **model_tar)
    
    print("created model")

    supervisor.epoch_num = 101
    supervisor.load_model()

    x, x0 = supervisor._prepare_data(x, x0)
    
    z_var_all = 0.1 + 0.9 * torch.sigmoid(supervisor.z_var_temp_all)
    zs = supervisor.dcrnn_model.sample_z(supervisor.z_mean_all, z_var_all, inputer)
    outputs_hidden = supervisor.dcrnn_model.dcrnn_to_hidden(x)
    output = supervisor.dcrnn_model.decoder(x0, outputs_hidden, zs)
    
    return output.tolist()