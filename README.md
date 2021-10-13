# Fail-Safe Execution of Deep Learning based Systems through Uncertainty Monitoring (Replication-Package)

Traffic Dataset: https://filesender.switch.ch/filesender2/?s=download&token=5b974ad1-8d03-4c66-83db-311c3e77c9d5

This folder contains the source code for the reproduction of our empirical studies, **under review**:

Michael Weiss and Paolo Tonella, **Uncertainty Quantification for Deep Neural Networks:An Empirical Comparison and Usage Guidelines**,
Journal of Software: Testing, Verification and Reliability (STVR).

To get access to `uncertainty_wizard` (the tool presented in the paper),
please refer to [the library github repo](https://github.com/testingautomated-usi/uncertainty-wizard).

#### Access to result databases:
The results, which are shown in the papers' plots and tables in aggregated form, can be found 
in a collection of ``sqlite`` databases. 
After acceptance, we will upload them to zenodo. Until then, please use the following download link:

[https://filesender.switch.ch/filesender2/?s=download&token=2e06d52a-6aec-4300-88bd-490a0f762c89](https://filesender.switch.ch/filesender2/?s=download&token=2e06d52a-6aec-4300-88bd-490a0f762c89)

The link will expire in december. Should the reviewing still be in progress by then, we'll update the link with a more recent version.

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
                                                                                                                                                                                       
