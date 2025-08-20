import math
import sys
from typing import List
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class FeatureReduction(object):

    def __init__(self):
        pass

    @staticmethod
    def forward_selection(data: pd.DataFrame, target: pd.Series,
        significance_level: float=0.1) ->dict:
        """		
		Args:
		    data: (pandas data frame) contains the features
		    target: (pandas series) represents target values to search to generate significant features
		    significance_level: (list) threshold to reject the null hypothesis
		Return:
		    best_features: (list) contains significant features. Each feature name is a string.
		Hint:
		    Forward Selection Steps:
		    1. Start with an empty list of selected features (null model)
		    2. For each feature NOT yet included in the selected features:
		        - Fit a simple regression model using the the selected features AND the feature under consideration
		        - Use sm.OLS() with sm.add_constant() to add a bias constant to your features when fitting the model
		        - Record the p-value of the feature under consideration
		        - Use OLSResults.pvalues to access the pvalues of each feature in a fitted model
		    3. Find the feature with the minimum p-value.
		        - If the feature's p-value < significance level, add the feature to the selected features and repeat from Step 2.
		        - Otherwise, stop and return the selected features
		
		    - You can access the feature names using data.columns.tolist().
		    - You can index into a Pandas dataframe using multiple column names at once in a list.
		"""
        remaining_features = data.columns.tolist() # features to iterate through
        selected_features = [] # empty list of selected features 
        
        while remaining_features: # iterate over all features in the list (while not empty)
            p_vals = pd.Series(dtype=float) # 1D column of type float, Series([], dtype:float64). empty initially, and stores p-values
            # for every candidate feature in curr iteration
            for feature in remaining_features:

                # test model with new feature
                X_feature_matrix = data[selected_features + [feature]] # get selected_features + new feature from DF to be used to test model
                model = sm.OLS(target, sm.add_constant(X_feature_matrix)).fit() # as instructed
                
				# get new p
                p_val = model.pvalues[feature] # get p-value from this specific feature out of all curr
                p_vals[feature] = p_val # associate curr p-value with this feature name. remem is a Series

            smallest_pval = p_vals.min() # smallest p-value
            smallest_pval_feature = p_vals.idxmin() # smallest p-value feature

            if smallest_pval < significance_level: # evaluate null-hypothesis
                selected_features.append(smallest_pval_feature) # reject
                remaining_features.remove(smallest_pval_feature) # fail to reject
            else:
                break

        return selected_features

    @staticmethod
    def backward_elimination(data: pd.DataFrame, target: pd.Series,
        significance_level: float=0.1) ->dict:
        """		
		Args:
		    data: (pandas data frame) contains the features
		    target: (pandas series) represents target values to search to generate significant features
		    significance_level: (list) threshold to reject the null hypothesis
		Return:
		    best_features: (float) contains significant features. Each feature name is a string.
		Hint:
		    Backward Elimination Steps:
		    1. Start with a full list of ALL features as selected features.
		    2. Fit a simple regression model using the selected features
		        - Use sm.OLS() with sm.add_constant() to add a bias constant to your features when fitting the model
		    3. Find the feature with the maximum p-value.
		        - Use OLSResults.pvalues to access the pvalues of each feature in a fitted model
		        - If the feature's p-value >= significance level, REMOVE the feature to the selected features and repeat from Step 2.
		        - Otherwise, stop and return the selected features.
		
		    - You can access the feature names using data.columns.tolist().
		    - You can index into a Pandas dataframe using multiple column names at once in a list.
		"""
        remaining_features = data.columns.tolist() # features to iterate through
        # selected_features = [] # empty list of selected features 
        
        while remaining_features: # iterate over all features in the list (while not empty)
            p_vals = pd.Series(dtype=float) # 1D column of type float, Series([], dtype:float64). empty initially, and stores p-values
            # for every candidate feature in curr iteration
            for feature in remaining_features:

                # test model with new feature
                # X_feature_matrix = data[selected_features + [feature]] # get selected_features + new feature from DF to be used to test model
                model = sm.OLS(target, sm.add_constant(data[remaining_features])).fit() # as instructed
                
				# get new p
                p_val = model.pvalues[feature] # get p-value from this specific feature out of all curr
                p_vals[feature] = p_val # associate curr p-value with this feature name. remem is a Series

            largest_pval = p_vals.max() # smallest p-value
            largest_pval_feature = p_vals.idxmax() # smallest p-value feature

            if largest_pval >= significance_level: # evaluate null-hypothesis
                # selected_features.append(largest_pval_feature)
                remaining_features.remove(largest_pval_feature)
            else:
                break

        return remaining_features

    def evaluate_features(data: pd.DataFrame, y: pd.Series, features: list
        ) ->None:
        """
        PROVIDED TO STUDENTS

        Performs linear regression on the dataset only using the features discovered by feature reduction for each significance level.

        Args:
            data: (pandas data frame) contains the feature matrix
            y: (pandas series) output labels
            features: (python list) contains significant features. Each feature name is a string
        """
        print(f'Significant Features: {features}')
        data_curr_features = data[features]
        x_train, x_test, y_train, y_test = train_test_split(data_curr_features,
            y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = math.sqrt(mse)
        print(f'RMSE: {rmse}')
        print()
