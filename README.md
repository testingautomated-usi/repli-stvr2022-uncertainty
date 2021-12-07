# Fail-Safe Execution of Deep Learning based Systems through Uncertainty Monitoring (Replication-Package)

This folder contains the source code for the reproduction of our empirical studies, **under review**:

Michael Weiss and Paolo Tonella, **Uncertainty Quantification for Deep Neural Networks:An Empirical Comparison and Usage Guidelines**,
Journal of Software: Testing, Verification and Reliability (STVR).

⚠️ UPDATE on December 7, 2021 ⚠️ This code uses a tensorflow version with a **critical** security vulnerability. We do not upgrade our dependencies in this repo (it's a repo intended for replication), but before you use this code on your machine, please make sure to update the dependencies or to run in a safe container. Find information about the vulnerability [here](https://github.com/advisories/GHSA-57wx-m983-2f88).

#### Access to result databases:
The results, which are shown in the papers' plots and tables in aggregated form, can be found 
in a collection of ``sqlite`` databases. 
After acceptance, we will upload them to zenodo. Until then, please use the following download link:

[https://filesender.switch.ch/filesender2/?s=download&token=f7c00a4b-e7d1-485b-ac6f-785e8a1f92b4](https://filesender.switch.ch/filesender2/?s=download&token=f7c00a4b-e7d1-485b-ac6f-785e8a1f92b4) *updated on december 7, with identical content (link was about to expire)*

The link will expire in february. Should the reviewing still be in progress by then, we'll update the link with a more recent version.

#### Limitations and notes:

- Some of the used datasets (Traffic and ImageNet) need to be downloaded directly from their original sources (copyright reasons). 
- Make sure to mount a drive `root/assets` to save the results and datasets
- For windows, install a venv according to the requirements.txt
- For linux, we recommend building a docker container according to the docker file in `docker/gpu/env/Dockerfile`.
  Optionally, the following script can be used to automatically create the docker container:
  > bash ./scripts/docker_build.sh -g emp_uncertainty.                                                                                                                                                                                       
- Some paths and system config (e.g. gpu selection) are workstation specific and you may have to change them.
- This repository is archived (i.e., we are not planning to modify the replication package in the future).
  For questions about the replication package, do not hesitate to contact us by email.
  For questions regarding `uncertainty-wizard` we refer to the issue tracker in [the library github repo](https://github.com/testingautomated-usi/uncertainty-wizard).
                                                                                                                                                                                       
