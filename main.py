import pandas as pd
from sklearn.model_selection import train_test_split
from module_transform import calculate_x_train_y_train_nmi_dict
import numpy as np
import json
import os
import copy

from module_load import DataLoader
from eval import Evaluator
from module_BM import BiasMitigation as BM
from module_AE import AccuracyEnhancement as AE
from module_transform import Transform
import config
from config import PARAMS_MAIN_MAX_ITERATION, USE_BIAS_MITIGATION, USE_ACCURACY_ENHANCEMENT, PARAMS_MAIN_THRESHOLD_EPSILON, PARAMS_MAIN_THRESHOLD_ACCURACY, VERBOSE

def run_test():
    # Load data
    if VERBOSE:
        print("Loading data...")
    loader = DataLoader()
    X, Y, O, categorical, numerical = loader.load_data()

    if VERBOSE:
        print("Categorical attributes:", categorical)
        print("Numerical attributes:", numerical)

    # Initialize evaluator
    evaluator = Evaluator(
        label_O=O.columns.tolist(), 
        label_Y=Y.name,
        cate_attrs=categorical, 
        num_attrs=numerical
    )

    # Split dataset
    X_train, X_test, Y_train, Y_test, O_train, O_test = train_test_split(
        X, Y, O, test_size=0.3, random_state=1
    )

    nmi_org = calculate_x_train_y_train_nmi_dict(X_train, Y_train)

    # Initialize transformer and models
    transformer = Transform()

    # Initialize bias mitigation and accuracy enhancement objects
    bm = BM(
        evaluator=evaluator,
        transformer=transformer,
        label_O=O.columns.tolist(),
        cate_attrs=categorical,
        num_attrs=numerical
    )

    ae = AE(
        evaluator=evaluator,
        transformer=transformer,
        label_Y=Y.name,
        cate_attrs=categorical,
        num_attrs=numerical
    )

    # Calculate initial epsilon values
    init_epsilon = evaluator.calculate_epsilon(
        X, O, categorical, numerical
    )
    if VERBOSE:
        print("Initial epsilon values:")
        print(pd.DataFrame(init_epsilon))
    
    # Calculate initial average epsilon
    all_epsilon_values = []
    for o_label in init_epsilon.keys():
        for attr in init_epsilon[o_label].index:
            all_epsilon_values.append(init_epsilon[o_label][attr])
    init_avg_epsilon = np.mean(all_epsilon_values)
    selected_label_O, selected_attribute = bm._find_max_epsilon_attribute(init_epsilon)
    init_max_epsilon = init_epsilon[selected_label_O][selected_attribute]
    if VERBOSE:
        print(f"Initial max epsilon: {init_max_epsilon}")
    
    # Calculate epsilon threshold
    epsilon_threshold = init_avg_epsilon * PARAMS_MAIN_THRESHOLD_EPSILON

    # Initial evaluation
    if VERBOSE:
        print("Initial evaluation:")
    init_metrics = evaluator.evaluate(
        X_train, Y_train, O_train, X_test, Y_test, O_test
    )
    if VERBOSE:
        print(init_metrics)
    
    # Get initial accuracy
    init_acc = init_metrics.get('ACC', 0)
    if VERBOSE:
        print(f"Initial accuracy: {init_acc}")
    
    # Calculate accuracy threshold
    acc_threshold = init_acc * (1 + PARAMS_MAIN_THRESHOLD_ACCURACY)

    # Initialize variables
    changed_dict = {}
    iter_count = 0
    transformed_df = X.copy()
    
    # Initialize results storage
    results_history = {
        'iterations': [],
        'metrics': [],
        'epsilon_values': [],
        'changed_dicts': [],
        'selected_attributes': []
    }

    # Check if both bias mitigation and accuracy enhancement are disabled
    # If both are disabled, return initial values without iteration
    if not USE_BIAS_MITIGATION and not USE_ACCURACY_ENHANCEMENT:
        if VERBOSE:
            print("\nBias mitigation and accuracy enhancement are both disabled.")
            print("Returning initial values without iteration...")
        # Set final results to initial values
        final_metrics = init_metrics
        changed_dict = {}
        
        # Save results with empty iterations
        save_results(results_history, init_metrics, final_metrics, changed_dict, init_epsilon)
        
        if VERBOSE:
            print("\nProcess completed.")
        return final_metrics, changed_dict, results_history

    # Main loop: combined bias mitigation and accuracy enhancement process
    if VERBOSE:
        print("\nStarting combined bias mitigation and accuracy enhancement process...")
    while iter_count < PARAMS_MAIN_MAX_ITERATION:
        iter_count += 1
        if VERBOSE:
            print(f"\n--- Iteration {iter_count} ---")
        
        # Initialize iteration data
        iter_data = {
            'iteration': iter_count,
            'selected_label_O': None,
            'selected_attribute': None
        }
        
        # Step 1: Bias Mitigation
        if USE_BIAS_MITIGATION:
            if VERBOSE:
                print("Performing bias mitigation...")
            # Calculate current epsilon values
            current_epsilon = evaluator.calculate_epsilon(
                transformed_df, O, categorical, numerical
            )
            
            # Find attribute with maximum epsilon
            selected_label_O, selected_attribute = bm._find_max_epsilon_attribute(current_epsilon)
            if VERBOSE:
                print(f"Selected for bias mitigation: {selected_label_O}, {selected_attribute}")
                print(f"Current max epsilon: {current_epsilon[selected_label_O][selected_attribute]}")
            
            # Update iteration data
            iter_data['selected_label_O'] = selected_label_O
            iter_data['selected_attribute'] = selected_attribute
            
            # Perform bias mitigation
            transformed_df, changed_dict = bm.mitigate(
                X, Y, O, nmi_org, changed_dict
            )
            
            # Save intermediate results
            mitigation_epsilon = evaluator.calculate_epsilon(
                transformed_df, O, categorical, numerical
            )
            if VERBOSE:
                print("Epsilon after bias mitigation:")
                print(pd.DataFrame(mitigation_epsilon))
        
        # Step 2: Accuracy Enhancement
        if USE_ACCURACY_ENHANCEMENT:
            if VERBOSE:
                print("Performing accuracy enhancement...")
            # Perform accuracy enhancement
            _, changed_dict = ae.enhance(
                X_train, Y_train, changed_dict
            )
        
        transformed_df = transformer.transform_data(X, changed_dict, numerical, categorical)
        transformed_X_train = transformer.transform_data(X_train, changed_dict, numerical, categorical)
        transformed_X_test = transformer.transform_data(X_test, changed_dict, numerical, categorical)
        
        # Evaluate current results
        if VERBOSE:
            print("Evaluating current results...")
        metrics = evaluator.evaluate(
            transformed_X_train, Y_train, O_train, transformed_X_test, Y_test, O_test
        )
        if VERBOSE:
            print(metrics)
        
        # Calculate current average epsilon
        current_epsilon = evaluator.calculate_epsilon(
            transformed_df, O, categorical, numerical
        )
        selected_label_O, selected_attribute = bm._find_max_epsilon_attribute(current_epsilon)
        current_max_epsilon = current_epsilon[selected_label_O][selected_attribute]

        if VERBOSE:
            print(f"Current max epsilon: {current_max_epsilon}")
        
        # Get current accuracy
        current_acc = metrics.get('ACC', 0)
        if VERBOSE:
            print(f"Current accuracy: {current_acc}")
        
        # Save iteration results
        results_history['iterations'].append(iter_count)
        results_history['metrics'].append(metrics)
        results_history['epsilon_values'].append(current_epsilon)
        results_history['changed_dicts'].append(copy.deepcopy(changed_dict))
        results_history['selected_attributes'].append(iter_data)
        
        # Check termination conditions
        termination_reason = None
        
        # Check epsilon termination condition if bias mitigation is enabled
        if USE_BIAS_MITIGATION and current_max_epsilon <= epsilon_threshold:
            termination_reason = f"Epsilon threshold reached: {current_max_epsilon} <= {epsilon_threshold}"
            
        # Check accuracy termination condition if accuracy enhancement is enabled
        elif USE_ACCURACY_ENHANCEMENT and current_acc >= acc_threshold:
            termination_reason = f"Accuracy threshold reached: {current_acc} >= {acc_threshold}"
        
        if termination_reason:
            if VERBOSE:
                print(f"\nTermination condition met: {termination_reason}")
            break
        
        # Print changed dictionary
        if VERBOSE:
            print("Current changed_dict:")
            print(changed_dict)
    
    # Final evaluation
    if VERBOSE:
        print("\n--- Final Results ---")
        print("Final evaluation:")
    final_metrics = evaluator.evaluate(
        transformed_df, Y, O, X_test, Y_test, O_test
    )
    if VERBOSE:
        print(final_metrics)

    if VERBOSE:
        print("\nFinal changed_dict:")
        print(changed_dict)

    # Save all results to a single JSON file
    save_results(results_history, init_metrics, final_metrics, changed_dict, init_epsilon)

    if VERBOSE:
        print("\nProcess completed.")
    return final_metrics, changed_dict, results_history

def convert_to_serializable(obj):
    """Convert various data types to Python native types for JSON serialization."""
    # Handle pandas Series
    if 'pandas' in str(type(obj)) and hasattr(obj, 'to_dict'):
        return convert_to_serializable(obj.to_dict())
    
    # Handle pandas DataFrame
    elif 'pandas' in str(type(obj)) and hasattr(obj, 'to_dict') and hasattr(obj, 'columns'):
        return convert_to_serializable(obj.to_dict(orient='records'))
    
    # Handle numpy arrays
    elif hasattr(obj, 'tolist'):
        return convert_to_serializable(obj.tolist())
    
    # Handle dictionaries
    elif isinstance(obj, dict):
        # Convert numpy keys and values
        return {
            (str(k) if hasattr(k, 'dtype') and np.issubdtype(k.dtype, np.integer) else str(k) if not isinstance(k, (str, int, float, bool, type(None))) else k): 
            convert_to_serializable(v) 
            for k, v in obj.items()
        }
    
    # Handle lists and tuples
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    
    # Handle numpy floats
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    
    # Handle numpy integers
    elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    
    # Handle other numpy numeric types
    elif hasattr(obj, 'dtype') and np.issubdtype(obj.dtype, np.number):
        return float(obj) if np.issubdtype(obj.dtype, np.floating) else int(obj)
    
    # Handle datetime types
    elif hasattr(obj, 'strftime'):
        return obj.isoformat()
    
    # Handle any other objects by converting to string
    else:
        return str(obj) if not isinstance(obj, (str, int, float, bool, type(None))) else obj

def save_results(results_history, init_metrics, final_metrics, final_changed_dict, init_epsilon):
    """Save all results to a single JSON file for later analysis."""
    # Create results directory if it doesn't exist
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Get all config parameters
    config_params = {}
    for param_name in dir(config):
        if not param_name.startswith('__') and param_name.isupper():
            config_params[param_name] = getattr(config, param_name)
    
    # Prepare full results data with proper conversion
    full_results = {
        'config_parameters': config_params,
        'initial_metrics': convert_to_serializable(init_metrics),
        'initial_epsilon': convert_to_serializable(init_epsilon),
        'iterations': [],
        'final_results': {
            'metrics': convert_to_serializable(final_metrics),
            'changed_dict': convert_to_serializable(final_changed_dict)
        }
    }
    
    # Add iteration data
    for i, (metrics, changed_dict, selected_attr) in enumerate(zip(
        results_history['metrics'], 
        results_history['changed_dicts'],
        results_history['selected_attributes']
    )):
        iteration_data = {
            'iteration': i + 1,
            'metrics': convert_to_serializable(metrics),
            'changed_dict': convert_to_serializable(changed_dict),
            'selected_attributes': convert_to_serializable(selected_attr)
        }
        full_results['iterations'].append(iteration_data)
    
    # Save all results to a single JSON file
    results_file = os.path.join(results_dir, 'all_results.json')
    with open(results_file, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    if VERBOSE:
        print(f"\nAll results saved to {results_file}")

# Run the test if this script is executed directly
if __name__ == "__main__":
    run_test()