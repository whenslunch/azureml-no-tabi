# Task 1 : Deploy an open source HuggingFace model in AzureML

## Task description

Deploy FLUX.1 Dev NF4 as a real-time Managed Endpoint for inferencing on Azure ML.

## Motivations
- Playing with HuggingFace (HF) models on a local machine is fine, but how to serve this model perhaps privately within a studio or company setting? Try this, I set about deploying this within an Azure test account.
- the HF model I needed was FLUX.1-dev NF4, but that is not available in the Azure ML Model Catalog, which would have been very convenient. So I had to find some way to load that model from HF.
- I wanted to explore Azure ML usage both from scripted and Portal UX approaches, so you will see a mix of Python and Azure ML Studio in the steps below. There's no reason it couldn't be completely one or the other.
- I wanted to get up and running asap and my python (among others) ain't too hot, so I used Copilot. A lot.

## Overview

0. Set up Azure ML, upload and register as a model in Azure ML
1. Set up FLUX.1-Dev NF4 ("Flux NF4" for short) for upload to Azure ML
2. Create Azure ML Environment
3. Write scoring script
4. Create Endpoint and Deployment
5. Test and profit

## 0. Set up Azure ML

Follow instructions from learn.microsoft.com to set up an Azure ML workspace in an Azure Subscription.
Some notes
- GPU compute quota is needed for this model. CPU works in theory, but is in practice way too slow
- Check which GPU SKUs are allowed for Managed Endpoints from [link]. For instance, A10 GPUs are not supported. I requested Standard_NC40ads_H100_v5, which I found to be a good balance of price/performance for my needs.
- Ensure GPU has enough VRAM. I selected Flux.1 dev NF4 which is a quanitized model that's ~9GB (vs regular ~30GB) in size which will fit any valid GPU's VRAM
- Security notes...

## 1. Set up HF model for upload, upload & register model

Flux.1-Dev NF4 consists of multiple components and not just a single model weights file, as many of the tutorials seemed to assume:
- 1 transformer (the Flux model itself)
- 2 CLIP (Contrastive Language-Image Pair) text encoder models
- 2 tokenizer models
- 1 variational autoencoder (vae)
- scheduler

There are a few ways to deal with this but one way is to load the model using FluxPipeline.from_pretrained() with the correct parameters, then save it out to a subdirectory on disk with FluxPipeline.save_pretrained(). Here, I saved it to "./saved_model". Before saving, run a test generation that outputs a PNG file. 

Now, FLUX is a gated model, meaning it is openly accessible, but you need to login to HF first, i.e. no anonymous access. So you need an HF account first of all.
Then, install the Hugging Face CLI:

    # pip install huggingface_hub
    
Then login to it with an Access Token you generate in your account:

    # huggingface-cli login

Another way that is similiar, is to utilize the FluxPipeline cache. When a model is loaded, FluxPipeline caches the model in either a default subdirectory or one specified by the user. That cache directory could also be uploaded to AzureML, but the structure created in the cache is not as intuitive as that of the save_pretrained method, so I prefer the former way out of these two.
   
Next, in python, configure an Azure MLClient with the right credentials, subscription, etc. and from there, create a registered model. 

The only wrinkle here is that by default, the AzureML workspace's storage account was set up with key-based authentication disabled. But I was working from a local machine and could not get a Managed Identity for it (I guess the assumption is that the user would be working from an Azure VM?) so had a specifically enable key auth on the storage account for this to work.

One other learning is that models created with the "v1" API from the azureml library are not compatible with "v2" API using the azure.ai.ml library. Found out the hard way through code suggested by Copilot, when I didn't specify which version it should use.


## 2. Create Azure ML Environment

There are several ways to create an Azure ML Environment, including using either curated or custom environments, etc. and either through the Portal or programmatically. I decided to use a pre-created base image then install needed components. I also assembled the Environment through the Portal.

First, I found in the Microsoft image repo a base image for GPU inferencing: mcr.microsoft.com/azureml/minimal-ubuntu22.04-py39-cuda11.8-gpu-inference:20241216.v1. The presence of the Nvidia CUDA driver is key.

Requirements.txt layers on the required libraries for the Flux NF4 model. Pinning the version numbers is important if you don't want mysterious model crashes.

Once the image is created, it was uploaded to my own Azure Container Registry. 

In the Portal, under Environments, click Create.

Settings: 
- Name: some relevant name
- Select environment source: "use existing docker image with optional conda file"
- Container registry image path: docker pull link to ACR image

Customize:
- Since everything was in the image already, I didn't need conda to do anything more, so left blank

Go through Tags and Review tabs, then Create.


## 3. Write scoring script

To create the endpoint, a scoring script- one that loads the model and handle inferencing requests.p must be supplied. (Side note, Azure ML uses Flask as the default app framework foor this script.) 

The generic scoring script template was modified to include the HF libraries for FluxPipeline and libraries for image manipulation. The now-familiar FluxPipeline.from_pretrained() loads the model from a concatenation of the fixed AZURE_MODEL_DIR location within the container, and "saved_model" which is the subdirectory from step 1.

The generated image was then saved in PNG format and sent as a response, along with the appropriate http mime type. 

## 4. Create Endpoint


## 5. Test 



## Improvements 

This is just a log of my first attempts, and by no means what "good" looks like. 
I intend to make improvements, which will be detailed in this project's backlog (tbd).
Also I will focus on using Python Notebooks going forward instead of this format.

## Lessons learned



