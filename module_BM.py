import pandas as pd
from itertools import combinations
import copy

from eval import Evaluator
from module_transform import Transform, calculate_x_train_y_train_nmi_dict

from config import (
    VERBOSE,

    PARAMS_MAIN_THRESHOLD_PHI,
    PARAMS_MAIN_THRESHOLD_PHI_ADAPT,
    
    PARAMS_MAIN_BM_REBIN_METHOD
)

class BiasMitigation:
    def __init__(self, evaluator: Evaluator, transformer: Transform, label_O, cate_attrs, num_attrs):
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
        self.label_O = label_O
        self.cate_attrs = cate_attrs
        self.num_attrs = num_attrs
        self.evaluator = evaluator
        self.transformer = transformer
    

    def _find_max_epsilon_attribute(self, df_epsilon, zorder=0):
        """
        Find the attribute with the specified rank (zorder) of epsilon value from df_epsilon
        
        Parameters:
        df_epsilon: dict
            Epsilon values from Evaluator.calculate_epsilon
        zorder: int (non-negative)
            The rank of epsilon to select, 0 for maximum (1st largest), 1 for 2nd largest, etc.
            
        Returns:
        tuple
            (selected_label_O, selected_attribute) - The attribute pair with the specified rank of epsilon
        """
        epsilon_candidates = []
        
        for label_O in df_epsilon:
            epsilon_series = df_epsilon[label_O]
            
            for attribute, epsilon_value in epsilon_series.items():
                epsilon_candidates.append((epsilon_value, label_O, attribute))
        
        epsilon_candidates.sort(reverse=True, key=lambda x: x[0])
        
        if zorder < 0 or zorder >= len(epsilon_candidates):
            raise ValueError(f"zorder must be a non-negative integer less than {len(epsilon_candidates)}. "
                            f"Received zorder={zorder}")
        
        selected_epsilon, selected_label_O, selected_attribute = epsilon_candidates[zorder]
        return selected_label_O, selected_attribute


    def compute_r1_rebin(self, df, attr_bias, label_O, zorder=1):
        diff_change = []
        
        unique_groups = list(df[label_O].unique())
        if len(unique_groups) < 2:
            return {}
        
        for i, j in combinations(unique_groups, 2):
            df_0 = df[df[label_O] == i].groupby(attr_bias).size()
            df_1 = df[df[label_O] == j].groupby(attr_bias).size()
            
            if len(df_0) == 0 or len(df_1) == 0:
                continue
            
            df_0_sum = max(1, len(df_0))
            df_1_sum = max(1, len(df_1))
            
            prop_0 = df_0 / df_0_sum
            prop_1 = df_1 / df_1_sum
            
            diff_matrix = prop_1 - prop_0
            
            group_diffs = []
            for group in diff_matrix.index:
                other_groups = [g for g in diff_matrix.index if g != group]
                for other in other_groups:
                    diff_val = diff_matrix.loc[group] - diff_matrix.loc[other]
                    group_diffs.append({
                        'group1': group,
                        'group2': other,
                        'diff': diff_val
                    })
            
            if not group_diffs:
                continue
            
            df_diffs = pd.DataFrame(group_diffs)
            
            if df_diffs.empty:
                continue
            
            df_diffs_sorted = df_diffs.sort_values(by='diff', ascending=False)
            if zorder <= len(df_diffs_sorted):
                selected = df_diffs_sorted.iloc[zorder-1]
                change_temp = {selected['group1']: selected['group2']}
            else:
                selected = df_diffs_sorted.iloc[0]
                change_temp = {selected['group1']: selected['group2']}
            
            diff_change.append({
                'change': change_temp,
                'diff': selected['diff']
            })
        
        if not diff_change:
            return {}
        
        df_results = pd.DataFrame(diff_change)
        
        if df_results.empty or 'diff' not in df_results.columns:
            return {}
        
        df_results_sorted = df_results.sort_values(by='diff', ascending=False)
        
        if zorder <= len(df_results_sorted):
            return df_results_sorted.iloc[zorder-1]['change']
        else:
            return df_results_sorted.iloc[0]['change']
        

    def step(self, X, transformed_df, O, selected_attribute, selected_label_O, changed_dict, zorder):
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
        zorder: int
            The rank order for selecting alternative changes
            
        Returns:
        dict: Updated changed_dict
        """
        if selected_attribute in changed_dict and changed_dict[selected_attribute] == 'dropped':
            return changed_dict
        
        if selected_attribute in self.cate_attrs:
            df_data_bm_temp = pd.concat([transformed_df.reset_index(drop=True), O.reset_index(drop=True)], axis=1)
            attr_bias = selected_attribute
            label_O = selected_label_O
            
            unique_values = df_data_bm_temp[attr_bias].nunique()
            if unique_values < 2:
                changed_dict[attr_bias] = 'dropped'
                return changed_dict
            
            change = {}
            
            if PARAMS_MAIN_BM_REBIN_METHOD == 'r1':
                change = self.compute_r1_rebin(
                    df_data_bm_temp,
                    attr_bias,
                    label_O,
                    zorder=zorder
                )
                
            elif PARAMS_MAIN_BM_REBIN_METHOD == 'r2':
                df_c = df_data_bm_temp[attr_bias].value_counts().sort_values()
                if len(df_c) >= 2:
                    change = {df_c.index[1]:df_c.index[0]}

            if change:
                if attr_bias not in changed_dict:
                    changed_dict[attr_bias] = {}
                elif isinstance(changed_dict[attr_bias], dict):
                    pass
                else:
                    changed_dict[attr_bias] = {}
                converted_change = {int(k): int(v) for k, v in change.items()}
                changed_dict[attr_bias].update(converted_change)
            else:
                changed_dict[attr_bias] = 'dropped'
        
        else:
            attr_bias = selected_attribute
            if attr_bias not in changed_dict:
                change = {'beta_O': 0}
            else:
                change = changed_dict[attr_bias]
                if isinstance(change, dict):
                    change['beta_O'] = change.get('beta_O', 0) + 1
                else:
                    changed_dict[attr_bias] = 'dropped'
                    return changed_dict
            
            if self.transformer.check_transform_validity(X, attr_bias, change, self.num_attrs, self.cate_attrs):
                changed_dict[attr_bias] = change
            else:
                changed_dict[attr_bias] = 'dropped'
            
        return changed_dict
    

    def adaptive_adjust_threshold(self, threshold, val_list, direction='down'):
        if len(val_list) >= 2:
            delta = val_list[-2] - val_list[-1]
            delta_total = abs(val_list[0] - threshold)

            if direction == 'down':
                if delta / delta_total < 0.1:
                    threshold = threshold * 0.95

        return threshold


    def mitigate(self, X, Y, O, nmi, changed_dict=None):
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
        if changed_dict is None:
            changed_dict = {}
        
        transformed_df = self.transformer.transform_data(X, changed_dict, self.num_attrs, self.cate_attrs)
        
        max_attempts = len(self.transformer.stream_data)
        attempt = 0
        zorder = 0
        success = False
        
        initial_df_epsilon = self.evaluator.calculate_epsilon(transformed_df, O, self.cate_attrs, self.num_attrs)
        selected_label_O, selected_attribute = self._find_max_epsilon_attribute(initial_df_epsilon)
        initial_max_epsilon = initial_df_epsilon[selected_label_O][selected_attribute]
        phi_list = []
        phi_threshold = PARAMS_MAIN_THRESHOLD_PHI
        
        while attempt < max_attempts and not success:
            attempt += 1
            
            temp_changed_dict = copy.deepcopy(changed_dict)
            
            temp_changed_dict = self.step(X, transformed_df, O, selected_attribute, selected_label_O, temp_changed_dict, zorder=zorder)
            
            if selected_attribute not in temp_changed_dict:
                zorder += 1
                if zorder >= len(X.columns):
                    break
                selected_label_O, selected_attribute = self._find_max_epsilon_attribute(initial_df_epsilon, zorder=zorder)
                initial_max_epsilon = initial_df_epsilon[selected_label_O][selected_attribute]
                attempt = 0
                continue
            
            if VERBOSE:
                print(f"Attempt {attempt}: Selected attribute = {selected_attribute}, Selected label_O = {selected_label_O}, Change = {temp_changed_dict[selected_attribute]}")
            
            temp_transformed_df = self.transformer.transform_data(X, temp_changed_dict, self.num_attrs, self.cate_attrs)
            nmi_temp = calculate_x_train_y_train_nmi_dict(temp_transformed_df, Y)
            phi_iter = ((nmi[selected_attribute] - nmi_temp[selected_attribute]) / (nmi[selected_attribute] + 1e-10))
            phi_list.append(phi_iter)

            if PARAMS_MAIN_THRESHOLD_PHI_ADAPT == 'grid':
                phi_threshold = self.adaptive_adjust_threshold(phi_threshold, phi_list)

            if phi_iter > phi_threshold:
                if attempt >= max_attempts:
                    zorder += 1
                    if zorder >= len(X.columns):
                        break
                    selected_label_O, selected_attribute = self._find_max_epsilon_attribute(initial_df_epsilon, zorder=zorder)
                    initial_max_epsilon = initial_df_epsilon[selected_label_O][selected_attribute]
                    attempt = 0
                continue

            new_df_epsilon = self.evaluator.calculate_epsilon(temp_transformed_df, O, self.cate_attrs, self.num_attrs)
            new_max_epsilon = new_df_epsilon[selected_label_O][selected_attribute]
            
            if VERBOSE:
                print(f"Attempt {attempt}: After change, maximum epsilon for {selected_attribute} on {selected_label_O} = {new_max_epsilon:.6f}")
            
            if new_max_epsilon < initial_max_epsilon:
                if VERBOSE:
                    print(f"Success! Epsilon decreased after attempt {attempt}.")
                changed_dict = copy.deepcopy(temp_changed_dict)
                transformed_df = temp_transformed_df
                success = True
            else:
                if VERBOSE:
                    print(f"Warning: Epsilon did not decrease in attempt {attempt}.")
                changed_dict[selected_attribute] = 'dropped'
                break

        return transformed_df, changed_dict