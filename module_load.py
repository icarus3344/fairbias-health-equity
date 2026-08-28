"""
module_load.py
=================
This module provides a DataLoader class for loading datasets and processing variable types.
The class handles dataset loading, determining variable types, splitting data into feature,
target, and protected attributes, and encoding categorical variables for machine learning tasks.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Optional

# Import configuration
from config import (
    VERBOSE,
    DATASET,
    DATASET_INFO,
    PARAMS_NUM_TO_CAT_CUTS_O,
    PARAMS_NUM_TO_CAT_METHOD_O,
    PARAMS_NUM_TO_CAT_CUTS_Y,
    PARAMS_NUM_TO_CAT_METHOD_Y,
    PARAMS_CAT_FROM_NUM_BINS,
    PARAMS_CAT_FROM_NUM_RATIO
)

class DataLoader:
    """
        Data loader class for loading datasets and processing variable types
    """
    
    def __init__(self):
        """Initialize DataLoader class"""
        self.df = None

        self.label_X = None
        self.df_X = None
        
        self.label_Y = None
        self.df_Y = None
        
        self.label_O = None
        self.df_O = None
        
        self.categorical_columns = []
        self.numerical_columns = []

        self.label_encoders = {}


    def load_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, List[str], List[str]]:
        """
        Load dataset and process according to requirements
        
        Returns:
            tuple: (X feature data, Y target variable, O protected attributes, 
                   list of categorical variables, list of continuous variables)
        """
        # Read dataset
        self._read_dataset()
        
        # Determine variable types
        self._determine_variable_types()
        
        # Split data into Y, O, and X
        self._split_data()
        
        # Process numerical O variables
        self._process_numerical_o()
        
        # Encode categorical variables to integers starting from 0
        self._encode_categorical_variables()
        
        # Output detailed information (if VERBOSE is enabled)
        if VERBOSE:
            self._print_data_info()
        
        return self.X, self.Y, self.O, self.categorical_columns, self.numerical_columns
    

    def _read_dataset(self, df=None) -> None:
        """Read dataset file, choose between name and path""" 
        if df:
            self.df = df

        else:
            self.data_path = DATASET.get('path', '')
            print(self.data_path)

            if not self.data_path:
                self.data_name = DATASET.get('name', '')

                if self.data_name in DATASET_INFO:
                    self.data_path = DATASET_INFO[self.data_name].get('path')
                else:
                    raise ValueError(f"Dataset '{self.data_name}' not found in DATASET_INFO")

            if os.path.exists(self.data_path):
                if self.data_path.endswith('.csv'):
                    self.df = pd.read_csv(self.data_path)
                    print(self.df)
                elif self.data_path.endswith('.xlsx') or self.data_path.endswith('.xls'):
                    self.df = pd.read_excel(self.data_path)
                elif self.data_path.endswith('.json'):
                    self.df = pd.read_json(self.data_path)
                elif self.data_path.endswith('.parquet'):
                    self.df = pd.read_parquet(self.data_path)
                else:
                    raise ValueError(f"Unsupported file format: {self.data_path}")
                
                if VERBOSE:
                    print(f"Loaded dataset from path: {self.data_path}")
            else:
                raise ValueError(f"Dataset file not found: {self.data_path}")


        if not hasattr(self, 'df'):
            raise RuntimeError("Error: self.df is not defined, please read the data first")
        
        if not isinstance(self.df, pd.DataFrame):
            raise TypeError(f"Error: self.df should be of type pd.DataFrame, but got {type(self.df)} instead")
        
        if self.df.empty:
            raise ValueError("Error: self.df is a DataFrame, but it contains no data rows")
    

    def _determine_variable_types(self) -> None:
        """Determine variable types (categorical or continuous)"""
        # Get predefined variable types from DATASET
        self.categorical_columns = DATASET.get('categorical', [])
        self.numerical_columns = DATASET.get('numerical', [])
        
        # If not enough type information provided, auto-detect
        if not self.categorical_columns or not self.numerical_columns:
            if hasattr(self, 'data_name'):
                self.categorical_columns = DATASET_INFO[self.data_name].get('categorical', [])
                self.numerical_columns = DATASET_INFO[self.data_name].get('numerical', [])
        
        all_columns = set(self.df.columns)
        no_type_columns = all_columns - (set(self.categorical_columns) | set(self.numerical_columns))

        auto_categorical, auto_numerical = self._auto_identify_column_types(no_type_columns)
        
        self.categorical_columns = list(set(self.categorical_columns) | set(auto_categorical))
        self.numerical_columns = list(set(self.numerical_columns) | set(auto_numerical))
        
        # Validate column names
        all_columns = set(self.df.columns)
        self.categorical_columns = [col for col in self.categorical_columns if col in all_columns]
        self.numerical_columns = [col for col in self.numerical_columns if col in all_columns]

        cate_set = set(self.categorical_columns)
        num_set = set(self.numerical_columns)
        duplicate_columns = cate_set & num_set
        
        if duplicate_columns:
            raise ValueError(
                f"Columns are marked as both categorical and numerical: {sorted(duplicate_columns)}"
            )
        
        covered_columns = cate_set | num_set
        uncovered_columns = all_columns - covered_columns
        
        if uncovered_columns:
            raise ValueError(
                f"Some columns are not classified as categorical or numerical: {sorted(uncovered_columns)}"
            )
    

    def _auto_identify_column_types(self, no_type_columns) -> Tuple[List[str], List[str]]:
        """
        Auto-identify column types
        
        Returns:
            tuple: (list of categorical variables, list of continuous variables)
        """
        categorical = []
        numerical = []
        
        for col in no_type_columns:
            # Skip columns with all NaN values
            if self.df[col].isna().all():
                continue
            
            # Check data type
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                numerical.append(col)

            elif pd.api.types.is_bool_dtype(self.df[col]):
                categorical.append(col)
                
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                unique_num = self.df[col].nunique()
                if unique_num < PARAMS_CAT_FROM_NUM_BINS:
                    categorical.append(col)
                else:
                    numerical.append(col)

            else:
                try:
                    _ = pd.to_numeric(self.df[col])
                    unique_ratio = self.df[col].nunique() / len(self.df[col])
                    if unique_ratio < PARAMS_CAT_FROM_NUM_RATIO:
                        categorical.append(col)
                    else:
                        numerical.append(col)
                except (ValueError, TypeError):
                    categorical.append(col)
        
        return categorical, numerical


    def _split_data(self) -> None:
        """Split data into Y, O, and X"""
        # Get Y
        self.label_Y = DATASET.get('label_Y')

        if not self.label_Y and self.data_name:
            self.label_Y = DATASET_INFO[self.data_name].get('label_Y')

        if not self.label_Y or self.label_Y not in self.df.columns:
            raise ValueError("Target variable 'lable_Y' is not defined in DATASET or not in the dataset")
        
        self.Y = self.df[self.label_Y].copy()
        

        # Get O
        self.label_O = DATASET.get('label_O', [])

        if not self.label_O and self.data_name:
            self.label_O = DATASET_INFO[self.data_name].get('label_O', [])
        
        if not self.label_O:
            self.label_O_top_K = DATASET.get('label_O_top_K')

            if not self.label_O_top_K:
                if self.data_name:
                    self.label_O_top_K = DATASET_INFO[self.data_name].get('label_O_top_K', 1)
                else:
                    self.label_O_top_K = 1
            
            from eval import Evaluator
            epsilon_dict = {}
            for label_O_temp in self.df.columns:
                if label_O_temp != self.label_Y:
                    evaluator = Evaluator(
                        label_O=label_O_temp,
                        label_Y=self.label_Y,
                        cate_attrs=self.categorical_columns,
                        num_attrs=self.numerical_columns
                    )

                    if label_O_temp in self.categorical_columns:
                        df_O_temp = self.df[label_O_temp]
                    elif label_O_temp in self.numerical_columns:
                        df_O_temp = self._transform_num_to_cate(self.df[label_O_temp])

                    df_epsilon_temp = evaluator.calculate_epsilon(
                        X=self.df.drop(columns=[self.Y, label_O_temp]),
                        O=pd.DataFrame(df_O_temp),
                        cate_attrs=self.categorical_columns,
                        num_attrs=self.numerical_columns
                    )
                    epsilon_dict[label_O_temp] = max(df_epsilon_temp[label_O_temp].values())

            sorted_labels = sorted(epsilon_dict.items(), key=lambda x: x[1], reverse=True)
    
            self.label_O = [item[0] for item in sorted_labels[:self.label_O_top_K]]


        if isinstance(self.label_O, str):
            self.label_O = [self.label_O]
        
        self.label_O = [col for col in self.label_O if col in self.df.columns]
        if not self.label_O:
            raise ValueError(f"Protected attributes {self.label_O} do not exist in the dataset")
        
        self.O = self.df[self.label_O].copy()


        # Get X
        self.label_X = DATASET.get('label_X')
        if not self.label_X and hasattr(self, 'data_name'):
            self.label_X = DATASET_INFO[self.data_name].get('label_X', [])
        
        if not self.label_X:
            self.X = self.df.drop(columns=[self.label_Y] + self.label_O).copy()
            self.label_X = self.X.columns.to_list()
        else:
            self.label_X = [col for col in self.label_X if col not in self.label_O]
            self.X = self.X[self.label_X]


    def _process_numerical_o(self) -> None:
        for col in self.O.columns:
            # Check if column is numerical
            if pd.api.types.is_numeric_dtype(self.O[col]) and col not in self.categorical_columns:
                self.O[col] = self._transform_num_to_cate(self.O[col])


    def _transform_num_to_cate(self, ser:pd.Series, mode='O') -> pd.Series:
        """Process numerical protected attributes O, converting them to categorical variables"""
        if mode == 'O':
            method = PARAMS_NUM_TO_CAT_METHOD_O
            bins = PARAMS_NUM_TO_CAT_CUTS_O
        elif mode == 'Y':
            method = PARAMS_NUM_TO_CAT_METHOD_Y
            bins = PARAMS_NUM_TO_CAT_CUTS_Y

        # If O has multiple columns, process each column
        if method == 'median':
            # Binarize using median
            median_val = ser.median()
            ser = (ser > median_val).astype(str)
        elif method == 'quartile':
            # Quartile binning into 4 categories
            try:
                ser = pd.qcut(ser, q=4, labels=False)
                ser = ser.astype(str)
            except ValueError:
                # If quartile binning fails, use custom binning
                ser = pd.cut(ser, bins=bins, labels=False)
                ser = ser.astype(str)
        else:
            # Custom binning
            ser = pd.cut(ser, bins=bins, labels=False)
            ser = ser.astype(str)
        
        return ser
    

    def _encode_categorical_variables(self) -> None:
        """Encode categorical variables to integers starting from 0 and process target variable Y"""
        # Encode categorical variables in X
        for col in self.categorical_columns:
            if col in self.X.columns:
                encoder = LabelEncoder()
                # Handle NaN values by converting them to a string representation
                X_col = self.X[col].fillna('NaN')
                self.X[col] = encoder.fit_transform(X_col)
                self.label_encoders[col] = encoder
        
        # Encode categorical variables in O
        for col in self.O.columns:
            if not pd.api.types.is_numeric_dtype(self.O[col]):
                encoder = LabelEncoder()
                # Handle NaN values by converting them to a string representation
                O_col = self.O[col].fillna('NaN')
                self.O[col] = encoder.fit_transform(O_col)
                self.label_encoders[col] = encoder
        
        # Encode categorical variables in Y
        if pd.api.types.is_numeric_dtype(self.Y):
            self.Y = self._transform_num_to_cate(self.Y, mode='Y')
        
        Y_col = self.Y.fillna('NaN')
        self.Y = pd.Series(encoder.fit_transform(Y_col), name=self.label_Y)
        self.label_encoders[self.label_Y] = encoder

        

    def _print_data_info(self) -> None:
        """Print detailed information about the dataset"""
        print("\nDataset loaded:")
        print(f"Original data shape: {self.df.shape}")
        print(f"Features X shape: {self.X.shape}")
        print(f"Target Y shape: {self.Y.shape}")
        print(f"Protected attributes O shape: {self.O.shape}")
        print(f"Categorical variables: {self.categorical_columns}")
        print(f"Numerical variables: {self.numerical_columns}")
        print(f"First 5 samples of Y:\n{self.Y.head()}")
        print(f"First 5 samples of O:\n{self.O.head()}")
        print(f"First 5 samples of X:\n{self.X.head()}")
    

    def get_encoding_map(self, column: str) -> Optional[Dict[int, str]]:
        """
        Get encoding map for a categorical column
        
        Parameters:
            column: Name of the categorical column
        
        Returns:
            Dictionary mapping encoded values to original labels, or None if column is not encoded
        """
        if column not in self.label_encoders:
            return None
        
        encoder = self.label_encoders[column]
        return {i: label for i, label in enumerate(encoder.classes_)}
    

    def get_column_types(self) -> Dict[str, List[str]]:
        """Get variable type information for the current dataset"""
        return {
            "categorical": self.categorical_columns,
            "numerical": self.numerical_columns
        }
        

    def inverse_encode(self, column: str, encoded_values) -> np.ndarray:
        """
        Convert encoded values back to original labels
        
        Parameters:
            column: Name of the column to inverse encode
            encoded_values: Encoded values to convert back
        
        Returns:
            Original labels corresponding to the encoded values
        """
        if column not in self.label_encoders:
            raise ValueError(f"Column '{column}' is not encoded or does not exist")
        
        encoder = self.label_encoders[column]
        return encoder.inverse_transform(encoded_values)

