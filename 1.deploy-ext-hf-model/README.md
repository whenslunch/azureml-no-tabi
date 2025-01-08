# Task 1 : Deploy an open source HuggingFace model in AzureML

## Task description

Deploy FLUX.1 Dev NF4 as a real-time Managed Endpoint for inferencing on Azure ML.

## Motivations
- Playing with HuggingFace (HF) models on a local machine has its limits. I needed (a) a more powerful GPU than I currently have, (b) won't pay to use hosted services and (c) just wanted to learn. So I set about deploying this within my Azure test account.
- the HF model I needed was FLUX.1-dev NF4, but that is not available in the Azure ML Model Catalog, which would have been very convenient. So I had to find some way to load that model from HF.
- I wanted to explore Azure ML usage both from scripted and Portal UX approaches, so you will see a mix of Python and Azure ML Studio in the steps below. There's no reason it couldn't be completely one or the other.

## Overview

0. Set up Azure ML
1. Set up FLUX.1-Dev NF4 ("Flux NF4" for short) for upload to Azure ML
2. Create suitable Docker container for deploying the model 
3. Write scoring script
4. Upload and register Flux NF4 as a Model in Azure ML
5. Create Azure ML Environment 
6. Create Endpoint and Deployment
7. Test and profit

## 0. Set up Azure ML

Follow instructions from learn.microsoft.com to set up an Azure ML workspace in an Azure Subscription.
Some notes
- GPU compute quota is needed. CPU works, but is in practice way too slow
- Check which GPU SKUs are allowed for Managed Endpoints from [link]. For instance, A10 GPUs are not supported
- Ensure GPU has enough VRAM. I specifically selected Flux.1 dev NF4 which is a quanitized model that's ~9GB (vs regular ~30GB) in size which will fit any valid GPU's VRAM
- Security notes...

## 1. Set up HF model

## 2. Create Docker container

## Improvements 

This is just a log of my first attempts, and by no means what "good" looks like. 
I intend to make improvements, which will be detailed in this project's backlog.

## Lessons learned



