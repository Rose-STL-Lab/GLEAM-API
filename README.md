# Gleam Model API

## Description

The goal of this project is to create an API so users can access the GLEAM simulation model and data without having to run the simulation on their local machine. This is a basic version of the API that will be used to connect the gleam model to machine learning models over the web. Instead of having a the more Complex GLEAM model accessed in this API, we instead are using the placerholder SEIR model located in SEIR.py. There are various branches on this repository that each correspond to different objectives and explorations that we have done. 

The API is built using FastAPI and depending on the branch, has a varying amount of endpoints. Main has a single endpoint that only takes in one set of parameters. Multiple takes in various sets of parameters. API Key has API key authentication. Gcloud has google cloud specific additions to the API. Async_api has a simple NP model attached that can communicate with the api when it is launched.

Ethan and Anirudh mainly worked on the Async_Api branch.
Kyla and Liam mainly worked on the gcloud branch.
Alaa and Manav mainly worked on the api_key branch.

For the final project submission, Anirudh worked on the branch clean_code. Manav worked on the branch container_v1. They both ) and combined the work from both branches (cleaned_code's containerization work is directly taken from container_v1) and deployed it to the cloud. 

## Set Up

Run this command to install libraries (For some branches like the machine learning one, some libraries like tensorflow may not be installed)

```console
pip install -r requirements.txt
```
Run this command to launch server

```console
uvicorn main:app -- reload
```

## Calling the API endpoint

Currently, the code for this api is deployed on google cloud at the link [https://gleam-seir-api-883627921778.us-west1.run.app](https://gleam-seir-api-883627921778.us-west1.run.app). In order to call the API however, one needs an API key to put in the header of their request. This can be obtained through asking the owner of the repository for one. There are currently three endpoints in the api. These endpoints are all relative to the original url.

The root endpoint (/) returns information about the user and is a get request.

The original endpoint (/original) returns data from the seir simulation based off of a set of parameters. It is a post request and requires a json in the payload that looks as follows:

```yaml
{
    days: 101,
    sims: 30,
    beta: 1.1,
    epsilon: 0.25,
}
```
With parameters of the type
```yaml
{
    days: int,
    sims: int,
    beta: float,
    epsilon: float,
}
```

The multiple endpoint (/multiple) returns sets data from the seir simulation based off of a list of parameters. The beta_epislon parameter is the same as the beta and epsilon from above except compressed into a list with the beta parameter coming first. It is a post request and requires a json in the payload that looks as follows:

```yaml
{
    days: 101,
    sims: 30,
    beta_epsilon: [[1.1,0.25],[1.2,0.26]],
}
```
With parameters of the type
```yaml
{
    days: int,
    sims: int,
    beta_epsilon: list,
}
```
