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

### **User Authentication**
- `GET /`
  - Retrieves user information and updates the number of runs this month.
  - **Dependencies:** `get_user`

### **Simulations**
- `POST /original`
  - Runs a standard SEIR simulation.
  - **Parameters:**
    - `days` (int)
    - `sims` (int)
    - `beta` (float)
    - `epsilon` (float)
  - **Returns:** JSON with simulation results.

- `POST /multiple`
  - Runs a SEIR simulation for multiple beta-epsilon parameter pairs.
  - **Parameters:**
    - `days` (int)
    - `sims` (int)
    - `beta_epsilon` (list of floats)
  - **Returns:** JSON with simulation results.

### **Compute Management**
- `POST /create_compute`
  - Creates a compute instance to run LEAM simulations using Docker.
  - **Parameters:**
    - `days`, `sims`, `beta`, `epsilon`
  - **Returns:** Timestamp of the created instance.

- `POST /create_dummy_compute`
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
  - Creates a new VM image in Google Cloud.
  - **Parameters:**
    - `bucket_name`, `folder_name`, `requirements_name`, `image_name`
  - **Returns:** Image name with timestamp.
 
- `POST /compute_with_config`
  - Preferred method for running simulations. Runs LEAM with a given config and optional extra data
  - **Parameters:**
    - `bucket_name`, `folder_name`, `requirements_name`, `image_name`
  - **Returns:** Timestamp for accessing data later

### **Machine Learning Models**
- `POST /stnp_model`
  - Runs the STNP model using a trained DCRNN model.
  - **Parameters:** `days`, `sims`, `beta`, `epsilon`
  - **Returns:** JSON with model predictions.

- `POST /gleam_ml`
  - Runs a GLEAM ML model using a pre-trained DCRNN supervisor.
  - **Parameters:** `days`, `sims`, `beta`, `epsilon`
  - **Returns:** Model inference results.

### **Data Management**
- `POST /data`
  - Retrieves or deletes simulation data from Google Cloud Storage.
  - **Parameters:**
    - `stamp` (str): Identifier for the data file.
    - `delete` (bool): Whether to delete the file after retrieval.
  - **Returns:** JSON with simulation data.

- `POST /create_yaml`
  - Uploads a YAML configuration to Google Cloud Storage.
  - **Parameters:**
    - `json_object` (dict)
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
2. Run the API: `uvicorn app.main:app --reload`
3. Use an API client (e.g., Postman) to interact with the endpoints.

This API supports cloud-based LEAM simulations and integrates with machine learning models for advanced epidemiological forecasting.

