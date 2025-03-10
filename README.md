# Gleam Model API

## Description

The goal of this project is to create an API so users can access the GLEAM simulation model and data without having to run the simulation on their local machine. This is a basic version of the API that will be used to connect the gleam model to machine learning models over the web. Instead of having a the more Complex GLEAM model accessed in this API, we instead are using the placerholder LEAM US model. There are various branches on this repository that each correspond to different objectives and explorations that we have done. 

The API is built using FastAPI and depending on the branch, has a varying amount of endpoints. Main has a single endpoint that only takes in one set of parameters. Multiple takes in various sets of parameters. API Key has API key authentication. Gcloud has google cloud specific additions to the API. Async_api has a simple NP model attached that can communicate with the api when it is launched.

Liam and Kyla worked extensively on the compute engine creation logic, including the data pipeline.


Alaa and Ethan handled the authorization and other api logistics, they also contributed to integrating endpoints.


Annirudh and Manav worked deeply on the Machine Learning aspect of our project and getting our surrogate model hosted on the cloud - code for the predictions can be found here - https://github.com/aindraga/Leam-US.


# LEAM Simulation API

This FastAPI-based API provides endpoints for running LEAM simulations, creating compute instances, and managing simulation data. It integrates with Google Cloud for compute and storage.

## Endpoints

### **Miscellaneous**
- `GET /`
  - Returns API Status
  - **Dependencies:** `get_user`

- `POST /estimate_cost`
  - Estimates the cost of running the VM
  - **Parameters:**
    - `num_gpu` (int)
    - `num_cpu` (int)
    - `num_ram` (int)
    - `hours` (int)
  - **Returns:** JSON with the cost estimate in US Dollars

### **Compute Management**
- `POST /gleam_simulation`
  - Creates a dummy stress-test compute instance.
  - **Parameters:**
    - `cpu`, `io`, `vm`, `vm_bytes`, `timeout`
  - **Returns:** Timestamp of the created instance.

- `POST /create_compute_with_image`
  - Creates a compute instance using a pre-configured image.
  - **Parameters:**
    - `days`, `sims`, `beta`, `epsilon`, `image`
  - **Returns:** Timestamp of the created instance.

- `POST /create_image`
  - Creates a new VM image in Google Cloud with config files attached based on the contents of folder_name.
  - **Parameters:**
    - `bucket_name`, `folder_name`, `base_image`, `image_name`, 
  - **Returns:** Image name with timestamp.
 
- `POST /gleam_simulation_personal_bucket`
  - Creates a dummy stress-test compute instance and pipes the data to the users storage bucket not in our gcp project.  Requires access to buck through service account
  - **Parameters:**
    - `cpu`, `io`, `vm`, `vm_bytes`, `timeout`,  `service_account`,  `service_account_key`,  `bucket_name`, `folder_name`
  - **Returns:** Timestamp of the created instance.

### **Machine Learning Models**

- `POST /prediction`
  - Runs a GLEAM ML model using a pre-trained DCRNN supervisor.
  - **Parameters:** `x`, `x0`
  - **Returns:** Model inference results.

### **Data Management**
- `GET /download_folder`
  - Retrieves simulation data from Google Cloud Storage.
  - **Parameters:**
    - `folder_name` (str)
  - **Returns:** JSON with download link

- `POST /create_yaml`
  - Uploads 2 yamls for GLEAM
  - **Parameters:**
    - `config1` (dict)
    - `config2` (dict)
  - **Returns:** Confirmation message.

## Key Dependencies

(more can be found in the requirements.txt)

- FastAPI
- NumPy
- Google Cloud SDK (Firestore, Storage)
- PyTorch
- YAML

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the API: `fastapi run app/main.py --port 80`
3. Use an API client (e.g., Postman) to interact with the endpoints.

This API supports cloud-based LEAM simulations and integrates with machine learning models for advanced epidemiological forecasting.

