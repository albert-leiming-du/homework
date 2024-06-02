Kubernetes Resource Recommendation System


About 

The system is designed for making recommendatios for a pod's cpu request and mem request, including before deployment, cold start and running phase.
There is a system design report file and a single Demo to show the entire process.

How to use?

1. Install the environment using conda (conda env create -f environment.yaml) or pip (python 3.7.16, pip install -r requirements.txt)
2. Activate the python conda environment by using 'conda activate hsbc'
3. Run ResourceRequestRecommendation.py by 'python ResourceRequestRecommendation.py' or 'python ResourceRequestRecommendation.py --mode 1 --code_filename './ResourceRequestRecommendation.py''