import numpy as np
from numpy.random import binomial
import time
import os
from google.cloud import storage 
import json
beta = os.getenv("BETA") or 1.0
epsilon = os.getenv("EPSILON") or 0.5
beta= float(beta)
epsilon = float(epsilon)

days = os.getenv("DAYS") or 100
simulations = os.getenv("SIMULATIONS") or 30
days= int(days)
simulations = int(simulations)
output_file_name = os.getenv("OUTFILENAME") or 'default'

def full_seir(num_days, beta_epsilon_flatten, num_simulations):
    mu = 1 #0.4
    all_cmpts = ['S', 'E', 'I', 'R', 'F']
    all_cases = ['E', 'I', 'R', 'F']

    x = range(num_days)

    ## model parameters
    ## initialization of california
    N = 100000 #39512000

    ## save the number of death (mean, std) for each senario
    train_mean_list = []
    train_std_list = []
    train_list = []

    for i in range (len(beta_epsilon_flatten)):
        init_I = int(2000) #2000
        init_R = int(0) #0
        init_E = int(2000) #2000

        ## save the number of individuals in each cmpt everyday
        dic_cmpts = dict()
        for cmpt in all_cmpts:
            dic_cmpts[cmpt] = np.zeros((num_simulations, num_days)).astype(int)

        dic_cmpts['S'][:, 0] = N - init_I - init_R - init_E
        dic_cmpts['I'][:, 0] = init_I
        dic_cmpts['E'][:, 0] = init_E
        dic_cmpts['R'][:, 0] = init_R
        

        ## save the number of new individuals entering each cmpt everyday
        dic_cases = dict()
        for cmpt in all_cmpts[1:]:
            dic_cases[cmpt] = np.zeros((num_simulations, num_days))

        ## run simulations
        for simu_id in range(num_simulations):
            for t in range(num_days-1):
                ## SEIR: stochastic
                flow_S2E = binomial(dic_cmpts['S'][simu_id, t], beta_epsilon_flatten[i,0] * dic_cmpts['I'][simu_id, t] / N)
                flow_E2I = binomial(dic_cmpts['E'][simu_id, t], beta_epsilon_flatten[i,1])
                flow_I2R = binomial(dic_cmpts['I'][simu_id, t], mu)
#                 print(t,flow_R2F)
                dic_cmpts['S'][simu_id, t+1] = dic_cmpts['S'][simu_id, t] - flow_S2E
                dic_cmpts['E'][simu_id, t+1] = dic_cmpts['E'][simu_id, t] + flow_S2E - flow_E2I
                dic_cmpts['I'][simu_id, t+1] = dic_cmpts['I'][simu_id, t] + flow_E2I - flow_I2R
                dic_cmpts['R'][simu_id, t+1] = dic_cmpts['R'][simu_id, t] + flow_I2R
                # dic_cmpts['F'][simu_id, t+1] = dic_cmpts['F'][simu_id, t] + flow_R2F

            
                ## get new cases per day
                dic_cases['E'][simu_id, t+1] = flow_S2E # exposed
                dic_cases['I'][simu_id, t+1] = flow_E2I # infectious
                dic_cases['R'][simu_id, t+1] = flow_I2R # removed
                # dic_cases['F'][simu_id, t+1] = flow_R2F # death 
        
        # rescale_cares_E = dic_cmpts['E'][...,1:]/N
        rescale_cares_I = dic_cmpts['I'][...,1:]/N*100
        # rescale_cares_R = dic_cmpts['R'][...,1:]/N

        train_list.append(rescale_cares_I)
        train_mean_list.append(np.mean(rescale_cares_I,axis=0))
        train_std_list.append(np.std(rescale_cares_I,axis=0))        

    train_meanset = np.stack(train_mean_list,0)
    train_stdset = np.stack(train_std_list,0)
    train_set = np.stack(train_list,0)
    return train_set

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to a Google Cloud Storage bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {bucket_name}/{destination_blob_name}.")

    

if __name__ == '__main__':
    beta_epsilon = np.array([[beta, epsilon]])
    result = full_seir(days, beta_epsilon, simulations)

    output_data = {"train_set": result.tolist()}

    output_file = f"{output_file_name}.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f)

    # Upload to Google Cloud Storage
    gcs_bucket = os.getenv("GCS_BUCKET", "seir-output-bucket-2")
    upload_to_gcs(gcs_bucket, output_file, f"{output_file_name}.json")
