# Gleam Model API

## Description

The goal of this project is to create an API so users can access the GLEAM simulation model and data without having to run the simulation on their local machine. This is a basic version of the API that will be used to connect the gleam model to machine learning models over the web. Instead of having a the more Complex GLEAM model accessed in this API, we instead are using the placerholder SEIR model located in SEIR.py. There are various branches on this repository that each correspond to different objectives and explorations that we have done. 

The API is built using FastAPI and depending on the branch, has a varying amount of endpoints. Main has a single endpoint that only takes in one set of parameters. Multiple takes in various sets of parameters. API Key has API key authentication. Gcloud has google cloud specific additions to the API. Async_api has a simple NP model attached that can communicate with the api when it is launched.

Ethan and Anirudh mainly worked on the Async_Api branch.
Kyla and Liam mainly worked on the gcloud branch.
Alaa and Manav mainly worked on the api_key branch.

## Set Up

Run this command to install libraries (For some branches like the machine learning one, some libraries like tensorflow may not be installed)

```console
pip install -r requirements.txt
```
Run this command to launch server

```console
uvicorn main:app -- reload
```
