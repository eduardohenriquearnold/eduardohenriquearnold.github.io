+++
date = "2017-10-09T09:44:42-02:00"
math = false
tags = ["devops", "docker"]
title = "Deploying JupyterHub with Docker and CUDA support"
summary = """
A brief summary of my experience deploying JupyterHub at our lab.
"""
+++

People doing research nowadays can benefit a lot from using [Jupyter notebooks](http://jupyter.org/). It is the case specially with machine learning and data science, where experimentation is frequent and code changes rapidly, as well as new insights are drawn from data and previous experience. In these cases it is useful to have proper documentation of the decision making process and some history of previous attempts to solve a problem.

At [LCS](http://lcs.ufsc.br/) we have a server with fair computing capabilities, including a GTX 1060 GPU. We were looking for a solution that allowed to share this server's resources without a big overhead on the user end, such as having each user installing packages and configuring their environment. The solution we envisioned was to deploy our own [JupyterHub](https://github.com/jupyterhub/jupyterhub) instance on the server, with the additional challenge of sharing the GPU.

Simply put, the JupyterHub service spawns a single-user Jupyter notebook for each user that connects to the server, allowing an easy solution for resource sharing, environment configuration and user isolation. There are many options that must be configured, such as the authentication service and spawner used. Since lately I have been using Docker for service isolation I was glad to know that there was a builtin spawner that created a Docker container for each user. 

The default configuration required the JupyterHub service to be installed on the host machine and provided no isolation from other services/processes. There is the option to create a container with JupyterHub and use the Docker spawner, but it requires a Docker container to launch other containers, quite an inception, huh? To do that we share the Docker socket with the JupyterHub container so that it has access to the host Docker service, enabling it to launch other containers. Although it is not the best solution security-wise, it is the best we could do at the time.

This solution seemed like a lot of hassle to put a service online. Fortunately, the team behind JupyterHub did a great job and provided a [reference deployment](https://github.com/jupyterhub/jupyterhub-deploy-docker) for a single host using Docker. It is important to note that this is not intended to be used in production, since it does not scale well for a very large number of users, in which case it would be recommended to use a kubernetes based deployment. Despite of that, as we are a small research group, this single host solution should be enough.

The reference deployment uses docker-compose to make the service easy to setup and manage. After configuring a Github application for user authentication and the __Let's Encrypt__ SSL keys (using [this script](https://github.com/jupyterhub/jupyterhub-deploy-docker/tree/master/examples/letsencrypt)), I was able to have a working service in a couple of hours. It should have taken less time, but I had to deal with 
a problem using the docker-compose file specific to the __Let's Encrypt__ configuration, check [here](https://github.com/jupyterhub/jupyterhub-deploy-docker/pull/48).

So we now have a working version of JupyterHub, but we still need to make the host GPU available inside the notebook container, which turns out to be the greatest challenge. There is a quite useful wrapper to do so called [nvidia-docker](https://github.com/NVIDIA/nvidia-docker). Although we cannot use it directly this time since the JupyterHub's Docker spawner plugin directly calls Docker through the API. Following Andrea Zonca's [blog post](https://zonca.github.io/2016/10/dockerspawner-cuda.html) we can configure the spawner to share the GPU resources.

First of all, we need to get the correct flags for the specific driver and devices:

```
curl -s localhost:3476/docker/cli
```

Which in our machine outputs

```
--volume-driver=nvidia-docker --volume=nvidia_driver_384.59:/usr/local/nvidia:ro --device=/dev/nvidiactl --device=/dev/nvidia-uvm --device=/dev/nvidia0
```

We then add these options to the spawner configuration in `jupyterhub_config.py`, which will cause the same effect as spawning the containers using the `nvidia-docker` wrapper:

```
c.DockerSpawner.read_only_volumes = {"nvidia_driver_384.59":"/usr/local/nvidia"}
c.DockerSpawner.extra_host_config = { "devices":["/dev/nvidiactl","/dev/nvidia-uvm","/dev/nvidia0"] }
```

Please note that some of this configuration options are overwritten later in the file, so other than adding these lines you should make sure  they remain active. You can also check my own version of the [jupyterhub_config.py](https://github.com/eduardohenriquearnold/jupyterhub-deploy-docker/blob/master/jupyterhub_config.py) file.

Finally, the last step is to create our personalized notebook container image including all the required packages used in our research. Since we want CUDA support, we must use a CUDA enabled image as base, such as the one supported by NVIDIA ``nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04``. A notebook image must have a start script that will be called upon container creation to start the Jupyter notebook server. I have modified the base image to be based on the NVIDIA CUDA image, as seen [here](https://github.com/eduardohenriquearnold/docker-stacks/tree/master/cuda-notebook). I have also created a specific image for our lab, inheriting from this image and containing software libraries such as OpenCV, PyTorch, TensorFlow and Keras, you can check it [here](https://github.com/eduardohenriquearnold/docker-stacks/tree/master/deeptools-notebook).

To check that everything worked as expected and the GPU is accessible from within the container you input the following on a Jupyter notebook command cell: `!nvidia-smi`. It should return the status of GPU. We now have a nice platform to develop our models and share computing resources.


