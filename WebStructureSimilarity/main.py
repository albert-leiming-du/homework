# -*- coding: utf-8 -*-
# @Author  : Du Leiming
# @FileName: main.py
# @Date    : 2024/4/16

import argparse

from DOMTree import DOMTree
from numpy import dot
from numpy.linalg import norm


def jaccard_similarity(dom1_vector, dom2_vector):
    """
    Calculate the Jaccard similarity between two vectors.
    Please note: Here, the computatioin equation may be different from traditional equation, so we call it approximate jaccard.

    Args:
        dom1_vector (list of int): The first vector.
        dom2_vector (list of int): The second vector.

    Returns:
        float: The Jaccard similarity between the two vectors.
    """
    a, b = 0, 0
    for i in range(len(dom1_vector)): 
        # we use min value of the two number as the intersection number, and max value of the two number as the union value.
        a += min(dom1_vector[i], dom2_vector[i])
        b += max(dom1_vector[i], dom2_vector[i])
    similarity = a / b if b != 0 else 0  # Avoid division by zero error
    return similarity

def cosine_similarity(dom1_vector, dom2_vector):
    """
    Calculate the cosine similarity between two vectors.

    Args:
        dom1_vector (list of int): The first vector.
        dom2_vector (list of int): The second vector.

    Returns:
        float: The cosine similarity between the two vectors.
    """
    cos_sim = dot(dom1_vector, dom2_vector)/(norm(dom1_vector)*norm(dom2_vector))
    return cos_sim

def get_html_similarity(url1, url2, dimension=500):
    """
    Calculate the similarity between two HTML pages based on their DOM tree representations.

    Args:
        url1 (str): The URL of the first HTML page.
        url2 (str): The URL of the second HTML page.
        dimension (int, optional): The dimensionality of the feature vectors. Default is 1000.

    Returns:
        tuple: A tuple containing three elements:
               - bool: True if the similarity between the pages is above certain thresholds, False otherwise.
               - float: The cosine similarity value between the feature vectors.
               - float: The Jaccard similarity value between the feature vectors.
    """
    # Avoid repeated computation if URLs are same
    if url1 == url2:
        return True, 1., 1.
    dom_tree1 = DOMTree(url=url1, dimension=dimension)
    dom_tree2 = DOMTree(url=url2, dimension=dimension)
    dom_tree1_vector = dom_tree1.get_feature_vector()
    dom_tree2_vector = dom_tree2.get_feature_vector()

    if not dom_tree1_vector:
        print('Some error happed when fetching {}, please check your network or the code.'.format(url1))
    if not dom_tree2_vector:
        print('Some error happed when fetching {}, please check your network or the code.'.format(url2))

    # Do the calculation
    if dom_tree1_vector and dom_tree2_vector:
        cos_value = cosine_similarity(dom_tree1_vector, dom_tree2_vector)
        jaccob_value = jaccard_similarity(dom_tree1_vector, dom_tree2_vector)
        return cos_value > 0.25 or jaccob_value > 0.20, cos_value, jaccob_value
    else:
        print('Some error happed when fetching URLs')
        return None
        
def main():
    parser = argparse.ArgumentParser(description="HTML/CSS Structure Analysis Parse")

    # Add arguments
    parser.add_argument("--url1", type=str, default='https://www.hsbc.com/', help="The first URL address")
    parser.add_argument("--url2", type=str, default='https://www.hsbc.com/', help="The second URL address")
    parser.add_argument("--dimension", type=int, default=500, help="Feature Vector Dimension, default is 500")

    # Parse arguments
    args = parser.parse_args()

    sim_result = get_html_similarity(url1=args.url1, url2=args.url2, dimension=args.dimension)
    if sim_result:
        whether_sim, cos_value, jaccob_value = sim_result
        if whether_sim:
            print('The structure of the two URLs is similar, cos: {}, jaccorb: {}'.format(cos_value, jaccob_value))
        else:
            print('The structure of the two URLs is NOT similar! cos: {}, jaccorb: {}'.format(cos_value, jaccob_value))

if __name__ == "__main__":
    main()