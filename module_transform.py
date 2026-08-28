"""
module_transform.py
=================
This module provides a Transform class for transforming datasets based on a specified changed_dict.
The class handles both numerical and categorical attributes, applying transformations
as defined in the changed_dict parameter.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer

from config import (
    PARAMS_MAIN_ALPHA_O,

    PARAMS_TRANSFORM,
    PARAMS_TRANSFORM_MULTI,
    PARAMS_TRANSFORM_STREAM,
    PARAMS_TRANSFORM_STREAM_CONFIG,
    PARAMS_TRANSFORM_LOG_EPSILON,
    
    PARAMS_TRANSFORM_N_BINS,
    PARAMS_TRANSFORM_X_MAX

)


class Transform:
    """
    A class for transforming data based on a changed_dict specification.
    
    This class transforms data according to the provided changed_dict,
    which defines transformations for numerical and categorical attributes.
    """
    
    def __init__(self):
        """
        Initialize a new Transform instance.
        
        Automatically generates stream data based on configurations from config.py.
        """
        # Generate stream_data based on stream_name from config.py
        self.stream_data = self.gen_stream(PARAMS_TRANSFORM_STREAM, **PARAMS_TRANSFORM_STREAM_CONFIG)

        # Validate and set alpha_O parameter from config.py
        self.alpha_O = PARAMS_MAIN_ALPHA_O if 0 < PARAMS_MAIN_ALPHA_O < 1 else 0.8


    def _cal_beta_value(self, df_temp_attr, beta_value):
        """
        Calculate the beta transformation for a single attribute using the specified method.
        
        Parameters:
            df_temp_attr: The attribute data to transform
            beta_value: The beta value to use for transformation
            
        Returns:
            Transformed attribute data
        """
        if PARAMS_TRANSFORM == 'poly':
            return (df_temp_attr.astype(object).abs() ** beta_value) * (df_temp_attr.apply(np.sign))
        elif PARAMS_TRANSFORM == 'log':
            return np.abs(np.log(df_temp_attr.abs() + PARAMS_TRANSFORM_LOG_EPSILON) / np.log(beta_value)) * (df_temp_attr.apply(np.sign))
        elif PARAMS_TRANSFORM == 'arcsin':
            max_value = df_temp_attr.abs().max()
            min_value = df_temp_attr.abs().min()
            df_temp_attr = 2 * (df_temp_attr - min_value) / (max_value - min_value) - 1
            return (np.arcsin(df_temp_attr.astype(object).abs() / max_value) ** beta_value) * (df_temp_attr.apply(np.sign))
        else:
            raise ValueError(f"Invalid transform method: {PARAMS_TRANSFORM}")


    def _calculate_beta_transformation(self, df_temp_attr, beta_index):
        """
        Calculate beta transformation for a single beta value based on current PARAMS_TRANSFORM_MULTI.
        
        Parameters:
            df_temp_attr: The attribute data to transform
            beta_index: The index in stream_data (required for t2 and t3 transforms)
            
        Returns:
            Transformed attribute data
        """
        if PARAMS_TRANSFORM_MULTI == 't1':
            beta_value = self.stream_data[beta_index]
            return self._cal_beta_value(df_temp_attr, beta_value)
        elif PARAMS_TRANSFORM_MULTI == 't2':
            if isinstance(beta_index, int) and 0 <= beta_index < len(self.stream_data):
                beta_value = np.prod(self.stream_data[:beta_index+1])
                return self._cal_beta_value(df_temp_attr, beta_value)
            else:
                raise ValueError(f"Invalid beta_index: {beta_index}")
        elif PARAMS_TRANSFORM_MULTI == 't3':
            X_sum = 0
            if isinstance(beta_index, int) and 0 <= beta_index < len(self.stream_data):
                for i in range(beta_index + 1):
                    beta_i = self.stream_data[i]
                    X_sum += self._cal_beta_value(df_temp_attr, beta_i)
                return X_sum
            else:
                raise ValueError(f"Invalid beta_index: {beta_index}")
        else:
            raise ValueError(f"Invalid transform method: {PARAMS_TRANSFORM_MULTI}")


    def _transform_numerical_attribute(self, df_temp_attr, change):
        """
        Transform a single numerical attribute based on the provided change parameters.
        
        Parameters:
            df_temp_attr: The numerical attribute data to transform
            change: Transformation parameters for the attribute
            
        Returns:
            Transformed attribute data
        """
        # Get beta_O and beta_Y from stream_data if available
        if 'beta_O' in change.keys():
            beta_O_index = change['beta_O']
            if beta_O_index < 0:
                X_O = 1
            else:
                X_O = self._calculate_beta_transformation(df_temp_attr, beta_O_index)
        
        if 'beta_Y' in change.keys():
            beta_Y_index = change['beta_Y']
            if beta_Y_index < 0:
                X_Y = 1
            else:
                X_Y = self._calculate_beta_transformation(df_temp_attr, beta_Y_index)
        
        if 'beta_O' in change.keys() and 'beta_Y' in change.keys():
            df_temp_attr =  self.alpha_O * X_O + (1 - self.alpha_O) * X_Y
        elif 'beta_O' in change.keys():
            df_temp_attr =  X_O
        elif 'beta_Y' in change.keys():
            df_temp_attr =  X_Y
        
        return df_temp_attr


    def _transform_categorical_attribute(self, df_temp_attr, re_bin_dict):
        """
        Transform a single categorical attribute based on the provided re-binning dictionary.
        
        Parameters:
            df_temp_attr: The categorical attribute data to transform
            re_bin_dict: Dictionary defining the re-binning rules
            
        Returns:
            Transformed attribute data
        """
        # Create a copy to avoid modifying the original
        transformed_attr = df_temp_attr.copy()
        
        # Apply re-binning
        for old_value, new_value in re_bin_dict.items():
            transformed_attr[transformed_attr == old_value] = new_value
            
        return transformed_attr


    def transform_data(self, X, changed_dict, num_attrs, cate_attrs):
        """
        Transform the dataset according to the specified changed_dict.
        
        Parameters:
            X: DataFrame to be transformed
            changed_dict: Dictionary containing attributes to change and their transformation parameters
                          Format: {attribute_name: transformation_parameters}
            num_attrs: List of numerical attribute names
            cate_attrs: List of categorical attribute names
            
        Returns:
            pandas.DataFrame: Transformed dataset
        """
        # Create a copy of the data to avoid modifying the original
        df_data_temp = X.copy()

        # Process each attribute specified in changed_dict
        for attribute in changed_dict.keys():
            df_temp_attr = df_data_temp[attribute]
            change = changed_dict[attribute]

            if change == 'dropped':
                df_temp_attr = 1
            else:
                if attribute in num_attrs:
                    # Process numerical attributes using the dedicated method
                    df_temp_attr = self._transform_numerical_attribute(df_temp_attr, change)
                elif attribute in cate_attrs:
                    # Process categorical attributes using the dedicated method
                    re_bin_dict = changed_dict[attribute]
                    df_temp_attr = self._transform_categorical_attribute(df_temp_attr, re_bin_dict)

            # Update the attribute in the temporary DataFrame
            df_data_temp[attribute] = df_temp_attr
        return df_data_temp


    def check_transform_validity(self, X, attribute, change, num_attrs, cate_attrs):
        """
        Check if transforming the specified attribute with the given change is valid.
        
        Parameters:
            X: Original DataFrame
            attribute: Attribute name to check
            change: Transformation parameters for the attribute
            num_attrs: List of numerical attribute names
            cate_attrs: List of categorical attribute names
            
        Returns:
            bool: True if transformation is valid, False otherwise
        """
        try:
            # Create a copy of the attribute to avoid modifying the original
            df_temp_attr = X[attribute].copy()
            
            # Apply transformation directly to the single attribute
            if attribute in num_attrs:
                df_transformed_attr = self._transform_numerical_attribute(df_temp_attr, change)
                
                # Check for infinity values
                if df_transformed_attr.isin([PARAMS_TRANSFORM_X_MAX, -PARAMS_TRANSFORM_X_MAX]).any():
                    return False
                      
                # Check for valid data types
                if not np.issubdtype(df_transformed_attr.dtype, np.number):
                    return False
            
            elif attribute in cate_attrs:
                df_transformed_attr = self._transform_categorical_attribute(df_temp_attr, change)
                
                # Check for single unique value
                if df_transformed_attr.nunique() == 1:
                    return False
                
        except Exception as e:
            return False
            
        return True


    def gen_stream(self, mode, p=102, q=173, emin=1/2, emax=2, order=0, length=10):
        """
        Generate stream data based on the specified mode.
        
        Parameters:
            mode: Stream generation mode (d4A, d4B, E1-E9)
            p: Prime number for E modes (default: 102)
            q: Prime number for E modes (default: 173)
            emin: Minimum value for E modes (default: 1/2)
            emax: Maximum value for E modes (default: 2)
            order: Order of the stream (-1: descending, 0: original, 1: ascending)
            length: Length of the stream (default: 10, only for d4A and d4B)
            
        Returns:
            List: Generated stream data
        """
        if mode == 'd4A':
            # Generate 3, 5, 7, 9, ...
            return list(range(3, 3 + 2 * length, 2))
        elif mode == 'd4B':
            # Generate 1/3, 1/5, 1/7, 1/9, ...
            return [1 / x for x in range(3, 3 + 2 * length, 2)]
        elif mode.startswith('E') and mode[1:].isdigit() and 1 <= int(mode[1:]) <= 9:
            # Handle E1-E9 modes
            mode_num = int(mode[1:])
            stream_0 = []
            for num in range(q + 1):
                if (num != 0) and (num != 1):
                    temp_num = (p ** num) % q
                    if (temp_num != 1) and (temp_num != (q - 1)):
                        stream_0.append(temp_num)
            
            stream = []
            if mode_num == 1:
                for i in range(len(stream_0)):
                    temp_num_s = stream_0[i]
                    if emin < temp_num_s < emax:
                        stream.append(temp_num_s)
            elif mode_num == 2:
                for i in range(len(stream_0) - 1):
                    temp_num_s = stream_0[i] / stream_0[i + 1]
                    if emin < temp_num_s < emax:
                        stream.append(temp_num_s)
            elif mode_num == 3:
                for i in range(len(stream_0) - 1):
                    temp_num_s = stream_0[i] / (stream_0[i] + stream_0[i + 1])
                    if emin < temp_num_s < emax:
                        stream.append(temp_num_s)
            elif mode_num == 4:
                for i in range(len(stream_0) - 1):
                    temp_num_s = abs(stream_0[i] - stream_0[i + 1])
                    if emin < temp_num_s < emax:
                        stream.append(temp_num_s)
            elif mode_num == 5:
                for i in range(len(stream_0) - 1):
                    temp_num_s = np.abs((stream_0[i] - stream_0[i + 1]) / (stream_0[i] + stream_0[i + 1]))
                    if emin < temp_num_s < emax:
                        stream.append(temp_num_s)
            elif mode_num == 6:
                for i in range(len(stream_0)):
                    temp_num_s = 1 / stream_0[i]
                    if emin < temp_num_s < emax:
                        stream.append(temp_num_s)
            elif mode_num == 7:
                for i in range(len(stream_0) - 1):
                    temp_num_s = 1 / stream_0[i] + 1 / stream_0[i + 1]
                    if emin < temp_num_s < emax:
                        stream.append(temp_num_s)
            elif mode_num == 8:
                for i in range(len(stream_0) - 1):
                    temp_num_s = stream_0[i] * stream_0[i + 1] / (stream_0[i] + stream_0[i + 1])
                    if emin < temp_num_s < emax:
                        stream.append(temp_num_s)
            elif mode_num == 9:
                for i in range(len(stream_0) - 2):
                    temp_num_s = (stream_0[i] + stream_0[i + 1] + stream_0[i + 2]) / (stream_0[i] * stream_0[i + 1] * stream_0[i + 2])
                    if emin < temp_num_s < emax:
                        stream.append(temp_num_s)
            
            # Order the stream
            if order == -1:
                return list(np.sort(stream)[::-1])
            elif order == 0:
                return stream
            elif order == 1:
                return list(np.sort(stream))

        else:
            raise ValueError(f"Unsupported stream mode: {mode}")


def calculate_x_train_y_train_nmi_dict(X_train, Y_train, n_bins=PARAMS_TRANSFORM_N_BINS, strategy='quantile'):
    """
    Fully encapsulated function: Calculate the column-wise Normalized Mutual Information (NMI) 
    between X_train (multi-column features) and Y_train (single-column label), and store the results in a dictionary.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be of type pandas.DataFrame, please check the input format")
    if not isinstance(Y_train, pd.Series):
        raise TypeError("Y_train must be of type pandas.Series, please check the input format")
    
    X_processed = X_train.copy()
    Y_processed = Y_train.copy()

    # only discretize continuous columns
    continuous_cols = []
    discrete_cols = []
    for col in X_processed.columns:
        if X_processed[col].dtype == 'float64' or (X_processed[col].dtype == 'int64' and X_processed[col].nunique() > 10):
            continuous_cols.append(col)
        else:
            discrete_cols.append(col)
    
    # Discretize continuous columns if any exist
    if continuous_cols:
        discretizer = KBinsDiscretizer(
            n_bins=n_bins,
            encode='ordinal',  # Encode as integers to meet NMI calculation requirements
            strategy=strategy,
            random_state=42  # Fix random seed to ensure result reproducibility
        )
        discrete_continuous_data = discretizer.fit_transform(X_processed[continuous_cols])
        discrete_continuous_df = pd.DataFrame(
            discrete_continuous_data.astype(int),
            columns=continuous_cols,
            index=X_processed.index
        )
        X_processed[continuous_cols] = discrete_continuous_df

    nmi_result_dict = {}
    for feature_col in X_processed.columns:
        current_feature = X_processed[feature_col].values
        target_label = Y_processed.values
        
        # Calculate NMI score, handle extreme cases (return 0 if the feature has identical values)
        try:
            nmi_score = normalized_mutual_info_score(current_feature, target_label)
        except:
            nmi_score = 0.0
        
        # Round to 4 decimal places and store in the dictionary
        nmi_result_dict[feature_col] = round(nmi_score, 4)
    
    return nmi_result_dict

