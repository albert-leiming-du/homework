# -*- coding: utf-8 -*-
# @Author  : Du Leiming
# @FileName: ResourceRequestRecommendation.py
# @Date    : 2024/05/31

import pickle
import logging
import os
import traceback
import argparse
import json
import numpy as np

from itertools import groupby
from simhash import Simhash
from dtw import dtw
from sklearn.cluster import KMeans
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tools.eval_measures import rmspe
from statsmodels.tsa.arima.model import ARIMA

from alibabacloud_cms20190101.client import Client as Cms20190101Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_cms20190101 import models as cms_20190101_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_console.client import Client as ConsoleClient
from alibabacloud_tea_util.client import Client as UtilClient


class ResourceRequestRecommendation(object):

    def __init__(self, mode, before_deployment_db_name='./before_deployment.pickle', cold_start_db_name='./cold_start.pickle',
                 metric_list=['pod.cpu.usage_rate', 'pod.memory.working_set'], start_time=None, end_time=None, code_filename=None):
        """
        Initializes an instance of the class with given parameters.

        Args:
            mode (int): The mode of operation, 1 means before deploymnet, 2 means cold start, 3 means running.
            before_deployment_db_name (str): The name of the database file to be used for before deployment mode.
            cold_start_db_name (str): The name of the database file to be used for cold start mode.
            metric_list (list): A list of metrics to be monitored.
            start_time (str or None): The start time for the data to be processed. Defaults to None.
            end_time (str or None): The end time for the data to be processed. Defaults to None.
            code_filename (str or None): The filename of code to make comparison. Defaults to None.
        """
        self.mode = mode
        self.before_deployment_db_name = before_deployment_db_name
        self.cold_start_db_name = cold_start_db_name
        self.metric_list = metric_list
        self.start_time = start_time
        self.end_time = end_time
        self.code_filename = code_filename
        self.aggregate_window = 60000  # unit in ms to aggregate
        self.endpoint = f'metrics.cn-wulanchabu.aliyuncs.com'
        self.dimensions = '[{"userId":"1008681110180877","cluster":"cb5eb244afbd545c7a6e282adb0b59c0d",\
                            "namespace":"default","app":"linux","type":"Deployment","pod":"linux-5dd5d6c4f9-2rkm9"}]'
        self.ALIBABA_CLOUD_ACCESS_KEY_ID = 'ALIBABA_CLOUD_ACCESS_KEY_ID'  # paste the key_id here
        self.ALIBABA_CLOUD_ACCESS_KEY_SECRET = 'ALIBABA_CLOUD_ACCESS_KEY_SECRET'  # paste the key_secret here

    def before_deployment_recommendation(self):
        """
        Provides resource recommendations for before deployment phase based on code similarity.
        
        Returns:
            tuple: A tuple containing a boolean indicating whether similar code was found,
                   the recommended CPU request (or None if not found), and the recommended memory request (or None if not found).
        """
        # Check if the code file exists or provided code_filename is a file.
        if not os.path.exists(self.code_filename) or not os.path.isfile(self.code_filename):
            logging.warning('Before Deployment Phase: {} file does not exist or not a file, exit.'.format(self.code_filename))
            return 
        
        # Load the database and check if it is loaded successfully.
        database_file = self._load_pickle_file(self.before_deployment_db_name)
        if not database_file:
            logging.warning('Before Deployment Phase: Some errors happend when opening {}, exit.'.format(self.before_deployment_db_name))
            return 
        try:
            with open(self.code_filename, 'r') as fr:
                data = fr.read()
            
            # Compare the code file with the database to find similar entries and get recommendations
            whether_found, cpu_recomm, mem_recomm = self._code_similarity_comparison(data, database_file)

            # If similar code is found, print the recommended CPU and memory requests
            if whether_found:
                logging.info('Before Deployment Phase: The Recommendated Resource is: CPU request: {} U, MEM request: {} MB'.format(cpu_recomm, mem_recomm))
            else:
                logging.info('Before Deployment Phase: Canot do recommendation now, no similar data found.')

            return whether_found, cpu_recomm, mem_recomm
        except Exception as e:
            logging.error('Before Deployment Phase: Some errors happened when making comparisons, exit.')
            logging.error(traceback.format_exc())

    def cold_start_recommendation(self):
        """
        Provides resource recommendations for code start phase based on time series data similarity comparison.

        Returns:
            tuple: A tuple containing a boolean indicating whether similar cold start data was found,
                   the recommended CPU request (or None if not found), and the recommended memory request (or None if not found).
        """
        # Get time series data of the pod given by self.dimensions
        processed_data = self._collect_ts_data_from_a_pod()

        # Load the code start phase database and check whether there existed similar recommendations.
        look_up_result = self._cold_start_cpu_mem_similarity(processed_data)

        # Give the recommendations for cpu and mem based on lookup results and heuristic algorithms.
        final_dict = dict()
        for metric_name, metric_data in processed_data.items():
            if look_up_result.get(metric_name):
                logging.info('Cold Start Phase: using tsa similarity for {}'.format(metric_name))
                metric_value = look_up_result.get(metric_name)
            else:
                logging.info('Cold Start Phase: using heuristic for {}'.format(metric_name))
                metric_value = self._heuristic_method(metric_name=metric_name, single_ts=metric_data)
            final_dict[metric_name] = metric_value

        # using specific config for standardization
        cpu_recomm = final_dict.get('cpu')
        mem_recomm = final_dict.get('mem')
        cpu_standardized, mem_standardized = self._giving_proper_config_size(cpu_recomm, mem_recomm)
        logging.info('Cold Start Phase: The Recommendated Resource is: CPU request: {} U, MEM request: {} MB'.format(cpu_standardized, mem_standardized))
        return True, cpu_standardized, mem_standardized

    def running_recommendation(self):
        """
        Provides resource recommendations for running phase based on a couple of algorithms.

        Returns:
            tuple: A tuple containing a boolean indicating whether we do have recommendations,
                   the recommended CPU request (or None if not found), and the recommended memory request (or None if not found).
        """
        # Get time series data of the pod given by self.dimensions
        processed_data = self._collect_ts_data_from_a_pod()
        final_recomm = dict()

        for metric_name, metric_data in processed_data.items():
            # Check whether the given time series data is stable by using two algorithms
            # If yes, don't predict and get the compute value based on statistics.
            if self._using_rules_to_check_whether_stable(metric_data) or self._using_kmeans_to_check_whether_stable(metric_data):
                logging.info('Running Phase: Using stability check algorithms for {}'.format(metric_name))
                if 'cpu' in metric_name:
                    metric_compute = np.percentile(metric_data, 95)
                else:
                    metric_compute = np.max(metric_data)
            else:
                # Using some time series prediction algorithms to predict the future data and compute the value based on history + predictions.
                whether_predict, predict_data = self._running_predict(metric_data)
                if whether_predict:
                    logging.info('Running Phase: Using tsa prediction algorithms for {}'.format(metric_name))
                    if 'cpu' in metric_name:
                        metric_compute = np.percentile(metric_data + list(predict_data), 95)
                    else:
                        metric_compute = np.max(metric_data + list(predict_data))
                else:
                    # If connot predict, use some heuristic methold to decide the compute value.
                    logging.info('Running Phase: Using heuristic algorithms for {}'.format(metric_name))
                    metric_compute = self._heuristic_method(metric_name=metric_name, single_ts=metric_data)

            # Once we have the compute value, combined compute value with qos models to decide the final recommendations.
            if 'cpu' in metric_name:
                metric_recomm = metric_compute * self._CPU_QOS_model()
            elif 'mem' in metric_name:
                metric_recomm = metric_compute + self._MEM_OQS_model()
            else:
                metric_recomm = metric_compute
            final_recomm[metric_name] = metric_recomm

        # using specific cofig for standardization
        cpu_recomm = final_recomm.get('cpu')
        mem_recomm = final_recomm.get('mem')
        cpu_standardized, mem_standardized = self._giving_proper_config_size(cpu_recomm, mem_recomm)
        logging.info('Running Phase: The Recommendated Resource is: CPU request: {} U, MEM request: {} MB'.format(cpu_standardized, mem_standardized))
        return True, cpu_standardized, mem_standardized
    
    def _giving_proper_config_size(self, cpu, mem):
        """
        Making recommendations based different size config from some specific cloud service providers.
        We don't filter cpu and mem here, just return the original size because different cloud providers may have different limitations.
        """
        return cpu, mem
    
    def _load_pickle_file(self, filename):
        """
        Loads a pickle file.

        Args:
            filename (str): The name of the pickle file to load.

        Returns:
            The loaded data if successful, or None if an error occurs.
        """
        data = None
        try:
            with open(filename, 'rb') as fr:
                data = pickle.load(fr)
        except Exception as e:
            logging.warning('There are some errors when opening {}!'.format(filename))
            logging.error(traceback.format_exc())
        return data

    def _collect_ts_data_from_a_pod(self):
        """
        Get time series data of running from a pod, filter and aggregate it with some preprocessing ways.

        Returns:
            dict: A dictionary containing preprocessed time series data for different metrics.
        """
        # Using aliyun API to get the monitoring data.
        raw_data = self._get_docker_running_data()
        if not raw_data:
            logging.warning('Some errors happened when collecting the usage data of the pod, exit.')
            return
        
        # Using some simple ways to do preprocessing.
        preprocessed_data_dict = dict()
        for k, v in raw_data.items():
            if 'mem' in k:
                preprocessed_data_dict[k] = self._preprocessing_ts_data(v, self.aggregate_window, aggregate_func='max')
            else:
                preprocessed_data_dict[k] = self._preprocessing_ts_data(v, self.aggregate_window, aggregate_func='mean')
        return preprocessed_data_dict

    def _get_docker_running_data(self):
        """
        Retrieves time series data of running from a pod using the official aliyun API.

        Returns:
            dict: A dictionary containing time series data for different metrics.
        """
        config = open_api_models.Config(
            access_key_id=self.ALIBABA_CLOUD_ACCESS_KEY_ID,
            access_key_secret=self.ALIBABA_CLOUD_ACCESS_KEY_SECRET
        )
        config.endpoint = self.endpoint
        client = Cms20190101Client(config)
        
        describe_metric_list_request_list = [cms_20190101_models.DescribeMetricListRequest(
            namespace='acs_k8s',
            metric_name=single_metric,
            period='60',
            start_time=self.start_time,
            end_time=self.end_time,
            dimensions=self.dimensions
        ) for single_metric in self.metric_list]

        runtime = util_models.RuntimeOptions()
        # Get multiple data in just one call.
        try:
            monitor_data = dict()
            for i in range(len(self.metric_list)):
                resp = client.describe_metric_list_with_options(describe_metric_list_request_list[i], runtime)
                returned_json = json.loads(UtilClient.to_jsonstring(resp))
                ts_data_list = json.loads(returned_json.get('body').get('Datapoints'))

                if 'mem' in self.metric_list[i]:
                    # The original unit for memory is Byte, here, convert it to MB.
                    standard_data = [[single_ts.get('timestamp'), single_ts.get('Value') / (1024 * 1024)] for single_ts in ts_data_list]
                else:
                    standard_data = [[single_ts.get('timestamp'), single_ts.get('Value')] for single_ts in ts_data_list]
                if 'cpu' in self.metric_list[i]:
                    monitor_data['cpu'] = standard_data
                elif 'mem' in self.metric_list[i]:
                    monitor_data['mem'] = standard_data
                else:
                    monitor_data[self.metric_list[i]] = standard_data
            return monitor_data
        except Exception as e:
            logging.error('There are some errors when fetching monitoring data.')
            logging.error(traceback.format_exc())

    def _preprocessing_ts_data(self, time_based_ts, aggregate_window=60000, aggregate_func='mean'):
        """
        Preprocesses time series data by aggregating it over a specified window.
        For CPU data, we use 'mean' function, while for Memory data, we use 'max' function here.

        Args:
            time_based_ts (list): The time series data to be processed, where each entry is a [timestamp, value] pair.
            aggregate_window (int): The window size for aggregation. Defaults to 60000 ms.
            aggregate_func (str): The aggregation function to use ('mean' or 'max'). Defaults to 'mean'.

        Returns:
            list: The aggregated time series data without time points.
        """
        # Aggregate the original time series data based on the aggregate window
        aggregate_oroignal_ts = [[x[0]//aggregate_window, x[1]] for x in time_based_ts]
        aggregated_ts = list()    

        # Group the data by the aggregated time window    
        for k, single_group in groupby(aggregate_oroignal_ts, lambda x: x[0]):
            if aggregate_func == 'mean':
                aggregated_ts.append([k, np.mean([x[1] for x in list(single_group)])])
            else:
                aggregated_ts.append([k, np.max([x[1] for x in list(single_group)])])

        # Interpolate the aggregated values(using previous point) to fill in missing time points, avoiding the situation that there may be some missing values.
        time_points_list = [x[0] for x in aggregated_ts]
        time_points_dict = {k: v for k, v in aggregated_ts}
        aggregated_ts_without_time = list()
        start_point = min(time_points_list)
        end_point = max(time_points_list)
        for this_point in range(start_point, end_point+1):
            if this_point in time_points_dict:
                aggregated_ts_without_time.append(time_points_dict.get(this_point))
            else:
                aggregated_ts_without_time.append(time_points_dict.get(this_point - 1))
        return aggregated_ts_without_time
    
    def _CPU_QOS_model(self):
        """
        Returns the Quality of Service (QoS) model value for CPU.

        This method provides a constant value representing the CPU QoS model.
        In a real-world scenario, this could be replaced with a more complex
        calculation based on various factors.

        Returns:
            float: The CPU QoS model value.
        """
        return 1.1
    
    def _MEM_OQS_model(self):
        """
        Returns the Quality of Service (QoS) model value for Memory.

        This method provides a constant value representing the Memory QoS model.
        In a real-world scenario, this could be replaced with a more complex
        calculation based on various factors.

        Returns:
            int: The Memory QoS model value(in MB).
        """
        return 100
    
    def _running_predict(self, single_ts):
        """
        Predicts future values of a time series using ARIMA or Holt-Winters models. 
        More time series prediction algorithms can be implenmented here.

        We first split the original data to train and test part, then using the two parts to 
        veriry the acc of the mode. If the RMSPE of a model's predictions is below a certain 
        threshold, that model is used to predict the entire time series.

        Args:
            single_ts (list): The input time series data.
        
        Returns:
            tuple: A tuple containing a boolean indicating whether a model was used and 
                   the predicted time series (or the original time series if no model was used).
        """
        # we only do single prediction acc check to decide whether to use the model to predict
        single_ts_len = len(single_ts)
        split_point = int(single_ts_len * 0.7)
        single_ts_train = single_ts[:split_point]
        single_ts_veriry = single_ts[split_point:]

        arima_predict = self._arima_to_predict(single_ts=single_ts_train, forecast_steps=len(single_ts_veriry))
        if rmspe(single_ts_veriry, arima_predict) < 10:
            return True, self._arima_to_predict(single_ts=single_ts)
        holt_winters_predict = self._holt_winters_to_predict(single_ts=single_ts_train, forecast_steps=len(single_ts_veriry))
        if rmspe(single_ts_veriry, holt_winters_predict) < 10:
            return True, self._holt_winters_to_predict(single_ts=single_ts)
        return False, single_ts
    
    def _arima_to_predict(self, single_ts, forecast_steps=30):
        """
        Make predictions using an ARIMA (AutoRegressive Integrated Moving Average) model.

        Args:
            single_ts (array-like): Single time series data for which predictions are to be made.
            forecast_steps (int): Number of future time steps for which predictions are to be made.

        Returns:
            array-like: Predicted values for the specified number of future time steps.
        """
        arima_model = ARIMA(single_ts, order=(1, 0, 1))  # using some simple order here, require accurate model in reality.
        fitted_model = arima_model.fit()
        predicted_data = fitted_model.forecast(forecast_steps)
        return predicted_data

    def _holt_winters_to_predict(self, single_ts, forecast_steps=30):
        """
        Make predictions using Holt-Winters method.

        Args:
            single_ts (array-like): Single time series data for which predictions are to be made.
            forecast_steps (int): Number of future time steps for which predictions are to be made.

        Returns:
            array-like: Predicted values for the specified number of future time steps.
        """
        holt_winters_model = ExponentialSmoothing(single_ts, seasonal_periods=4, trend='additive', seasonal='additive')
        fitted_model = holt_winters_model.fit()
        predicted_data = fitted_model.forecast(forecast_steps)
        return predicted_data
    
    def _heuristic_method(self, metric_name, single_ts):
        """
        Heuristic method to compute a metric based on the input time series data.

        Args:
            metric_name (str): The name of the metric to be computed. Used to determine the computation method.
            single_ts (array-like): The input time series data on which the heuristic method will be applied.

        Returns:
            float: The computed metric based on the heuristic method.
        """
        simple_exp_model = SimpleExpSmoothing(single_ts)
        simple_exp_model_fitted = simple_exp_model.fit()
        exp_smoothed_data = simple_exp_model_fitted.fittedvalues
        if 'cpu' in metric_name:
            metric_compute = np.percentile(exp_smoothed_data, 99)
        else:
            metric_compute = np.max(exp_smoothed_data)
        return metric_compute

    def _using_rules_to_check_whether_stable(self, single_ts):
        """
        Checks whether the time series is stable based on coefficient of variation (CV).

        The method calculates the coefficient of variation (CV) for different window sizes
        and checks whether the mean of these CVs is below a threshold. If the mean CV is below
        the threshold, the time series is considered stable.

        Args:
            single_ts (list): The input time series data.
        
        Returns:
            bool: True if the time series is stable, False otherwise.
        """
        stride_list = [5, 10, 15, 20]
        windowed_cv = self._get_windowed_cv_feature(single_ts, stride_list)
        return np.mean(windowed_cv) < 0.1

    def _using_kmeans_to_check_whether_stable(self, single_ts):
        """
        Checks whether the time series is stable using KMeans clustering.

        The method creates feature vectors for the input time series and
        uses KMeans clustering to determine stability. It assigns the input
        time series to one of the clusters based on its similarity to the
        existing feature vectors.

        Args:
            single_ts (list): The input time series data.
        
        Returns:
            bool: True if the time series is stable, False otherwise.
        """
        stride_list = [5, 10, 15, 20]
        single_ts_feature_vector = self._get_windowed_cv_feature(single_ts, stride_list=stride_list)

        # Using some random data to train the model. In reality, there should have some pretrianed model to load.
        existed_feature_vectors = [np.array([0] * len(single_ts_feature_vector))] + [np.random.rand(len(single_ts_feature_vector)) for i in range(20)]
        kmeans_model = KMeans(n_clusters=3, random_state=0).fit(existed_feature_vectors)

        # using a list of 0 as the label the find the target label name. 
        this_ts_feature_samples = [[0] * len(single_ts_feature_vector), single_ts_feature_vector]
        predicted_label = kmeans_model.predict(this_ts_feature_samples)
        return predicted_label[0] == predicted_label[1]      

    def _get_windowed_cv_feature(self, single_ts, stride_list):
        """
        Generates a feature vector based on the coefficient of variation (CV) for the input time series.

        The method calculates the coefficient of variation (CV) for different window sizes
        specified by the stride_list, and generates a feature vector based on these CVs.

        Args:
            single_ts (list): The input time series data.
            stride_list (list): The list of stride values for windowing.

        Returns:
            list: The feature vector representing the time series.
        """
        windowed_cv_for_ts = list()
        single_ts_len = len(single_ts)
        # Iterate over each stride value in the stride_list
        for single_stride in stride_list:
            for i in range(single_ts_len//single_stride):
                std_over_the_window = np.std(single_ts[single_stride * i: single_stride * (i+1)])
                mean_over_the_window = np.mean(single_ts[single_stride * i: single_stride * (i+1)]) + 1e-10  # To avoid division by zero
                # Calculate the coefficient of variation (CV)
                windowed_cv_for_ts.append(std_over_the_window/mean_over_the_window)
        return windowed_cv_for_ts  
    
    def _code_similarity_comparison(self, code_to_check, code_simhash_database):
        """
        Compares a code snippet to existing code snippets in a database using Simhash.

        The method calculates the Simhash value for the input code snippet and compares it
        with the Simhash values of existing code snippets in the database. If a similar code
        snippet is found (based on Hamming distance), it returns the corresponding CPU and
        memory recommendations.

        Args:
            code_to_check (str): The code snippet to check for similarity.
            code_simhash_database (list): The database containing Simhash values and corresponding recommendations.
                The format of the simhash database is like the following case:
                    [   
                        # [simhash_value, cpu_recomm, mem_recomm]
                        [7483809945577191432, 0.5, 200],
                        [1083809945577191432, 1.0, 500],
                        [2183129945577191432, 2.0, 1000],
                    ]
        Returns:
            tuple: A tuple containing a boolean indicating whether a similar code snippet was found,
                   the recommended CPU request (or None if not found), and the recommended memory request (or None if not found).
        """
        hash_of_code = Simhash(code_to_check).value
        recom_cpu = None
        recom_mem = None
        whether_found = False
        for single_hash, single_cpu, single_mem in code_simhash_database:
            # check whether there exist similiar code in our database by using hamming distance 
            hamming_distance = bin(hash_of_code ^ single_hash).count('1')
            if hamming_distance < 30:
                recom_cpu = single_cpu
                recom_mem = single_mem
                whether_found = True
                break
        return whether_found, recom_cpu, recom_mem
    
    def _cold_start_time_series_similarity(self, single_ts, time_series_database, dtw_threshold=10):
        """
        Compares a time series to existing time series in a database using Dynamic Time Warping (DTW).

        The method calculates the DTW distance between the input time series and each time series
        in the database. If the minimum DTW distance is below a specified threshold, it indicates
        similarity, and the corresponding time series from the database is returned.

        Args:
            single_ts (list): The input time series data.
            time_series_database (list): The database containing time series data.
                A single time_series_dabase have the format like:
                    [   #[time_series_data, cpu_recomm or mem_recomm]
                        [[0.57369895, 0.25999088, ..., 0.09616202, 0.823009], 0.5 or 200],
                        [[0.57369895, 0.25999088, ..., 0.09616202, 0.823009], 0.5 or 200],
                        [[0.57369895, 0.25999088, ..., 0.09616202, 0.823009], 0.5 or 200],
                    ]
            dtw_threshold (float): The threshold for DTW distance to consider similarity. Defaults to 10.

        Returns:
            tuple: A tuple containing a boolean indicating whether a similar time series was found,
                   the CPU request (or None if not found) or the memory request (or None if not found).
        """
        manhattan_distance = lambda x, y: np.abs(x - y)
        dtw_distance = [dtw(single_ts, single_exist_ts[0], dist=manhattan_distance)[0] for single_exist_ts in time_series_database]
        min_index = np.argmin(dtw_distance)
        min_dtw_distance = dtw_distance[min_index]
        # We use simple dtw_threshold here.
        if min_dtw_distance < dtw_threshold:
            return True, time_series_database[min_index][1]
        return False, None
    
    def _cold_start_cpu_mem_similarity(self, processed_data):
        """
        Compares processed data to existing time series data in a database to recommend CPU and memory resources.

        The method loads a database containing time series data for different metrics and compares the processed
        data with the database entries. If similarity is found, CPU and memory recommendations are made.

        Args:
            processed_data (dict): Processed data containing metric names and corresponding time series data.

        Returns:
            dict: A dictionary containing recommended CPU and memory resources for metrics where similarity is found.
        """
        # Load time series database from pickle file
        ts_database_dict = self._load_pickle_file(self.cold_start_db_name)
        if not ts_database_dict:
            logging.warning('Cold Start Phase: Some errors happend when opening {}, exit.'.format(self.cold_start_db_name))
            return 
        recomm_result = dict()
        for metric_name, metric_data in processed_data.items():
            if metric_name in ts_database_dict:
                whether_found, resoure_recomm = self._cold_start_time_series_similarity(metric_data, ts_database_dict.get(metric_name))
                if whether_found:
                    recomm_result[metric_name] = resoure_recomm
        return recomm_result

    def main(self):
        """
        Executes the main functionality based on the specified mode.

        The method checks the mode specified by the user and executes the corresponding functionality:
        1. Before Deployment Recommendation
        2. Cold Start Recommendation
        3. Running Recommendation
        If the mode is not 1, 2, or 3, it logs a warning.

        Returns:
            None
        """
        if self.mode == 1:
            self.before_deployment_recommendation()
        elif self.mode == 2:
            self.cold_start_recommendation()
        elif self.mode == 3:
            self.running_recommendation()
        else:
            logging.warning('Wrong Mode, please specify the mode to 1, 2, 3.')


if __name__ == "__main__":
    logging.basicConfig(
            format='%(asctime)s %(levelname)s %(message)s',
            level=logging.INFO
        )
    parser = argparse.ArgumentParser(description="K8S cpu/mem request recommendation system")

    # Add arguments
    parser.add_argument("--mode", type=int, default=3, help="System Mode to choose, 1-before deployment, 2-cold start, 3-running")
    parser.add_argument("--start_time", type=str, default='2024-05-30 00:00:00', help="Start Time to get data")
    parser.add_argument("--end_time", type=str, default=None, help="End Time to get data")
    parser.add_argument("--code_filename", type=str, default='./ResourceRequestRecommendation.py', help="code filename in before deployment phase")
    
    # Parse arguments
    args = parser.parse_args()
    
    resource_recomm_system = ResourceRequestRecommendation(mode=args.mode, start_time=args.start_time, end_time=args.end_time, code_filename=args.code_filename)
    resource_recomm_system.main()
