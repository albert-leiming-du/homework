# -*- coding: utf-8 -*-
# @Author  : Du Leiming
# @FileName: DOMTree.py
# @Date    : 2024/4/16

import bs4
import requests

from bs4 import BeautifulSoup
from simhash import Simhash
from treelib import Tree


class DOMTree:
    def __init__(self, url, dimension=500):
        """
        Initializes an instance of the class with a given URL.

        Args:
            url (str): The URL of the webpage to be processed.
            dimension (int): The length of the feature vector.
        """
        self.dom_tree = Tree()
        self.headers = {'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.9 Safari/536.5'}
        self.url = url
        self.dimension = dimension
        self.feature_vector = [0 for i in range(dimension)]

    def fetch_html(self, url):
        """
        Fetch HTML content from the given URL.

        Args:
            url (str): The URL to fetch HTML content from.

        Returns:
            str: The HTML content, or None if fetching fails.
        """
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                return response.content
            else:
                print("Failed to fetch HTML from", url)
                return None
        except Exception as e:
            print("Error:", e)
            return None
    
    def preprocessing(self, bs_tag):
        """
        Data processing for a BeautifulSoup tag, filter unuseful info, replaced specific tokens, and return the features.

        Args:
            bs_tag (str): A node of BeautifulSoup.

        Returns:
            bs_tag_attr_str (str): The str name and attrs in specific form after preprocessing.
        """
        bs_tag_attr_list = [bs_tag.name]
        for attr_key in sorted(bs_tag.attrs.keys()):
            attr_value = str(bs_tag.attrs[attr_key])
            if attr_key == 'src':
                attr_value = ''
            elif attr_key == 'href':
                attr_value = ''
            bs_tag_attr_list.append(attr_key + ':' + attr_value)
        bs_tag_attr_str = ' '.join(bs_tag_attr_list)
        return bs_tag_attr_str

    def build_dom_tree(self, html_content):
        """
        Build tree structure using a non-recursive approach.
        Please note: we don't use recursive approach because it may lead to stack overflow.

        Args:
            html_content (str): The content of a html web page.

        Returns:
            self.dom_tree(treelib): The built dom tree.
        """
        html_bs = BeautifulSoup(html_content, 'lxml')
        bs_tag = None
        for content in html_bs.contents:
            if isinstance(content, bs4.element.Tag):
                bs_tag = content
        if not bs_tag:
            print("Failed to fetch HTML from", self.url)
            return self.dom_tree
                
        # Now let's build the tree
        dom_id = 1
        dom_tree_head = self.dom_tree.create_node(bs_tag.name, dom_id, data=self.preprocessing(bs_tag))
        dom_id += 1
        stack = [(bs_tag, dom_tree_head)]

        while stack:
            current_bs_tag, parent_node = stack.pop()
            # Process the children tag of the current tag
            for child_tag in current_bs_tag.contents:
                if isinstance(child_tag, bs4.element.Tag):
                    # Add the child node to the tree and push it onto the stack
                    child_node_id = self.dom_tree.create_node(child_tag.name, dom_id, parent=parent_node,
                                                    data=self.preprocessing(child_tag))
                    dom_id += 1
                    stack.append((child_tag, child_node_id))
        return self.dom_tree
    
    @staticmethod
    def hash_func(node_feature):
        """
        Compute the hash value of a node feature using the Simhash algorithm.

        Args:
            node_feature (str): The feature of the node.

        Returns:
            int: The hash value.
        """
        return Simhash(node_feature).value

    def get_feature_vector(self):
        """
        Calculate the feature vector for the HTML page.

        Returns:
            list: The feature vector.
        """
        # build the dome tree
        if not self.dom_tree:
            html_content = self.fetch_html(self.url)
            if not html_content:
                return None
            self.build_dom_tree(html_content)
            if not self.dom_tree:
                return None

        # Iterate over each node in the DOM tree
        for node_id in range(1, self.dom_tree.size() + 1):
            node = self.dom_tree.get_node(node_id)

            # Calculate feature hashes for the current node
            calculated_hash_list = self.calculate_feature_hashes_for_one_node(node, node_id)
            for single_hash in calculated_hash_list:
                # Project each segment of hash value to a bucket whose dimensioin is self.dimension
                bin_hash_value = bin(single_hash)[2:]
                bin_len_each_slice = 16
                num_seqment_value = len(bin_hash_value) // bin_len_each_slice
                num_segment = num_seqment_value + 1 if len(bin_hash_value) % bin_len_each_slice != 0 else num_seqment_value
                for hash_index in range(num_segment):
                    segment_value = int(bin_hash_value[hash_index * bin_len_each_slice:(hash_index+1) * bin_len_each_slice], 2)
                    self.feature_vector[segment_value % self.dimension] += 1

        return self.feature_vector

    def calculate_feature_hashes_for_one_node(self, dom_node, dom_node_id):
        """
        Calculate feature hashes for a single DOM node based on its structure and attributes.

        Args:
            dom_node (TreeNode): The DOM node.
            dom_node_id (int): The ID of the DOM node.

        Returns:
            list: A list of feature hashes.
        """
        # Get the data (attributes) of the current DOM node and calculate the feature hash for the node's data
        current_dom_node_data = dom_node.data
        hash_value_list = [self.hash_func(current_dom_node_data)]

        # Calculate feature hashes based on siblings and children structure
        siblings_structure_hash = self.hash_func(current_dom_node_data + ' '.join([brother_node.data for brother_node in self.dom_tree.siblings(dom_node_id)]))
        children_structure_hash = self.hash_func(current_dom_node_data + ' '.join([child_node.data for child_node in self.dom_tree.children(dom_node_id)]))

        # Add the calculated feature hashes to the list
        hash_value_list.extend([siblings_structure_hash, children_structure_hash])
        return hash_value_list
