Web Page Structural Similarity Analysis System


About 

Our Web Page Structural Similarity Analysis System is a powerful tool designed to compare the structures of different web pages. 
By analyzing HTML code and extracting key features, it assesses the similarity between pages, aiding in tasks such as detecting duplicate content, identifying plagiarism, and optimizing website design. 
The system is based on DOMTree, simhash, dimension projection, cosine similarity and jaccorb similarity, you may check the design doc.

How to use?

1. Install the environment using conda (conda env create -f environment.yaml) or pip (python 3.7.16, pip install -r requirements.txt)
2. Activate the python conda environment by using 'conda activate hsbc_env'
3. Run main.py by 'python main.py --url1 'xxxx' --url2 'xxxx'', for example: python main.py --url2 'https://www.baidu.com' --url1 'https://www.google.com.hk'