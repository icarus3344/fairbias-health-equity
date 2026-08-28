import copy
import numpy as np
import pandas as pd
from itertools import combinations

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_selection import mutual_info_classif

from eval import Evaluator
from module_transform import Transform

from config import (
    SEED,
    VERBOSE,
    
    PARAMS_MAIN_AE_IMPORTANCE_MEASURE,
    PARAMS_MAIN_AE_REBIN_METHOD,
)


class AccuracyEnhancement:
    def __init__(self, evaluator: Evaluator, transformer: Transform, label_Y, cate_attrs, num_attrs):
        """
            Initialize BiasMitigation class
            
            Parameters:
            evaluator: Evaluator
                Evaluator instance for calculating metrics
            transformer: Transform
                Transform instance for data transformation
            label_O: str or list of str
                Protected attribute(s) to mitigate bias
            label_Y: str
                Target variable
            cate_attrs: list of str
                Categorical attributes in the dataset
            num_attrs: list of str
                Numerical attributes in the dataset
        """
        self.label_Y = label_Y
        self.cate_attrs = cate_attrs
        self.num_attrs = num_attrs
        self.model = evaluator.model
        self.transformer = transformer
        self.skip_attr_list = []
    

    def _find_attribute(self, df_data_temp, Y, changed_dict):
        """
        Find the attribute with maximum epsilon value from df_epsilon
        
        Parameters:
        df_epsilon: dict
            Epsilon values from Evaluator.calculate_epsilon
            
        Returns:
        tuple
            (selected_label_O, selected_attribute) - The attribute pair with highest epsilon
        """
        if PARAMS_MAIN_AE_IMPORTANCE_MEASURE == 'a1':
            importance = mutual_info_classif(df_data_temp, Y, random_state=SEED)
            df_importance = pd.DataFrame(importance)
            df_importance.index = df_data_temp.columns
            df_importance.columns = ['importance']
            df_importance = df_importance.sort_values(by='importance', ascending=True)
        elif PARAMS_MAIN_AE_IMPORTANCE_MEASURE == 'a2':
            clf = LogisticRegressionCV(
                penalty='l1',
                solver='saga',
                multi_class='multinomial',
                random_state=SEED,
                max_iter=1000,
                n_jobs=-1
            ).fit(df_data_temp, Y)
            importance = np.mean(np.abs(clf.coef_), axis=0)
            df_importance = pd.DataFrame(importance)
            df_importance.index = df_data_temp.columns
            df_importance.columns = ['importance']
            df_importance = df_importance.sort_values(by='importance', ascending=True)
        elif PARAMS_MAIN_AE_IMPORTANCE_MEASURE == 'a3':
            importance = permutation_importance(self.model, df_data_temp, Y, random_state=SEED).importances_mean
            df_importance = pd.DataFrame(importance)
            df_importance.index = df_data_temp.columns
            df_importance.columns = ['importance']
            df_importance = df_importance.sort_values(by='importance', ascending=True)
        
        for attr_acc in df_importance.index:
            if attr_acc in changed_dict.keys():
                if changed_dict[attr_acc] != 'dropped':
                    if attr_acc not in self.skip_attr_list:
                        break
            else:
                if attr_acc not in self.skip_attr_list:
                    break
        return attr_acc
        

    def step(self, X, transformed_df, Y, selected_attribute, changed_dict):
        """
        Perform a bias mitigation step
        
        Parameters:
        X: pandas.DataFrame
            Input dataset
        transformed_df: pandas.DataFrame
            Transformed dataset
        O: pandas.Series
            Protected attribute(s) to mitigate bias
        selected_attribute: str
            Attribute to be mitigated
        selected_label_O: str
            Protected attribute to be mitigated for the selected_attribute
        changed_dict: dict
            Dictionary of changes made to the dataset
        df_epsilon: dict, optional
            Precomputed epsilon values from Evaluator.calculate_epsilon
            If not provided, will be computed using Evaluator
        
        Returns:
        tuple
            (selected_label_O, selected_attribute) - The attribute pair with highest epsilon
        """
        df_data_ae_temp = pd.concat([transformed_df.reset_index(drop=True), Y.reset_index(drop=True)], axis=1)
        attr_acc = selected_attribute
        temp_dict = changed_dict.copy()

        # Update changed_dict based on attribute type
        if selected_attribute in self.cate_attrs:
            if PARAMS_MAIN_AE_REBIN_METHOD == 'r1':
                diff_change = []
                for i,j in combinations(df_data_ae_temp[self.label_Y].unique(), 2):
                    df_0 = df_data_ae_temp[[attr_acc, self.label_Y]][df_data_ae_temp[self.label_Y]==i].groupby(attr_acc).count()
                    df_1 = df_data_ae_temp[[attr_acc, self.label_Y]][df_data_ae_temp[self.label_Y]==j].groupby(attr_acc).count()
                    df_c = (df_1 - df_0).fillna(0).abs().sort_values(by=self.label_Y)
                    diff = (df_c.max() + df_c.min()).values[0]
                    change_temp = {df_c.index[-1]:df_c.index[0]}
                    diff_change.append([change_temp, diff])
                diff_change = pd.DataFrame(diff_change, columns=['change', 'diff'])
                change = diff_change[diff_change['diff'].abs() == diff_change['diff'].abs().max()]['change'].values[0]
            
            elif PARAMS_MAIN_AE_REBIN_METHOD == 'r2':
                df_c = df_data_ae_temp[attr_acc].value_counts().sort_values()
                change = {df_c.index[1]:df_c.index[0]}
            
            elif PARAMS_MAIN_AE_REBIN_METHOD == 'r3':
                best_acc_temp = 0
                for i,j in combinations(df_data_ae_temp[attr_acc].unique(), 2):
                    change_temp = {i:j}
                    temp_dict_0 = temp_dict.copy()
                    temp_dict_0[attr_acc].update(change_temp)
                    temp_X_train, temp_Y_train, temp_X_test, temp_Y_test = train_test_split(X, Y, test_size=0.3, random_state=SEED)
                    temp_X_train = self.transformer.transform_data(temp_X_train, temp_dict_0)
                    temp_X_test = self.transformer.transform_data(temp_X_test, temp_dict_0)
                    model_temp = copy.deepcopy(self.model)
                    model_temp.fit(temp_X_train, temp_Y_train)
                    acc_temp = model_temp.score(temp_X_test, temp_Y_test)
                    if acc_temp > best_acc_temp:
                        change = change_temp
                        best_acc_temp = acc_temp
            
            if attr_acc in temp_dict.keys():
                temp_dict[attr_acc].update(change)
            else:
                temp_dict[attr_acc] = change
            
            if self.transformer.check_transform_validity(X, attr_acc, temp_dict[attr_acc], self.num_attrs, self.cate_attrs):
                changed_dict[attr_acc] = temp_dict[attr_acc]
            else:
                self.skip_attr_list.append(attr_acc)

            if attr_acc in temp_dict.keys():
                temp_dict[attr_acc].update(change)
            else:
                temp_dict[attr_acc] = change

            if self.transformer.check_transform_validity(X, attr_acc, temp_dict[attr_acc], self.num_attrs, self.cate_attrs):
                changed_dict[attr_acc] = temp_dict[attr_acc]
            else:
                self.skip_attr_list.append(attr_acc)
        
        else:
            # Numerical attribute - increment beta_O by 1
            attr_acc = selected_attribute
            temp_changed_dict = changed_dict.copy()
            if attr_acc not in temp_changed_dict:
                change = {'beta_Y': 0}
            else:
                change = temp_changed_dict[attr_acc]
                change['beta_Y'] = change.get('beta_Y', 0) + 1
            
            if self.transformer.check_transform_validity(X, attr_acc, change, self.num_attrs, self.cate_attrs):
                changed_dict[attr_acc] = change
            else:
                self.skip_attr_list.append(attr_acc)
            
        return changed_dict
    

    def enhance(self, X, Y, changed_dict=None):
        """
        Continuously perform bias mitigation steps based on PARAMS_STEP
        
        Parameters:
        X: pandas.DataFrame
            Input dataset
        O: pandas.Series
            Protected attribute(s) to mitigate bias
        changed_dict: dict, optional
            Initial dictionary of changes made to the dataset
            If not provided, will be initialized as empty dict
        
        Returns:
        tuple
            (transformed_df, final_changed_dict) - The transformed dataset and final changes
        """
        # Initialize changed_dict if not provided
        if changed_dict is None:
            changed_dict = {}
        
        # Create a copy of the dataframe to modify
        transformed_df = self.transformer.transform_data(X, changed_dict, self.num_attrs, self.cate_attrs)
        
        max_attempts = len(self.transformer.stream_data)  # Maximum number of attempts to decrease epsilon
        attempt = 0
        success = False
        
        # Create a temporary dictionary to track changes for each attempt
        temp_changed_dict = changed_dict.copy()
        temp_transformed_df = transformed_df.copy()
        
        selected_attribute = self._find_attribute(temp_transformed_df, Y, temp_changed_dict)

        X_train, X_test, transformed_X_train, transformed_X_test, Y_train, Y_test = train_test_split(X, transformed_df, Y, test_size=0.3, random_state=SEED)
        initial_acc = self.model.score(transformed_X_test, Y_test)


        while attempt < max_attempts and not success:
            attempt += 1

            # Perform a mitigation step
            temp_changed_dict = self.step(X_train, transformed_X_train, Y_train, selected_attribute, temp_changed_dict)
            if VERBOSE:
                if selected_attribute in temp_changed_dict.keys():
                    print(f"Attempt {attempt}: Selected attribute = {selected_attribute}, Change = {temp_changed_dict[selected_attribute]}")
                elif selected_attribute in self.skip_attr_list:
                    print(f"Attempt {attempt}: Selected attribute = {selected_attribute}, No change, Skip")

            # Apply the changes to the dataframe
            transformed_X_train = self.transformer.transform_data(X_train, temp_changed_dict, self.num_attrs, self.cate_attrs)
            transformed_X_test = self.transformer.transform_data(X_test, temp_changed_dict, self.num_attrs, self.cate_attrs)
            
            # Calculate new epsilon after change
            model_temp = copy.deepcopy(self.model)
            model_temp.fit(transformed_X_train, Y_train)
            new_acc = model_temp.score(transformed_X_test, Y_test)
            
            if VERBOSE:
                print(f"Attempt {attempt}: After change, accuracy = {new_acc:.6f}")
            
            # Check if accuracy increased
            if new_acc > initial_acc:
                if VERBOSE:
                    print(f"Success! Accuracy increased after attempt {attempt}.")
                # Update the main change dict and transformed df
                changed_dict.update(temp_changed_dict)
                transformed_df = temp_transformed_df
                success = True
            else:
                if VERBOSE:
                    print(f"Warning: Accuracy did not increase in attempt {attempt}.")
                self.skip_attr_list.append(selected_attribute)
            
            if selected_attribute in self.skip_attr_list:
                break
        # After all attempts, update the main variables 

        return transformed_df, changed_dict