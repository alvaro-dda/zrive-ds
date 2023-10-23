import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import json 
import pickle
from datetime import datetime

def load_data(path: str) -> pd.DataFrame:
    """
    Loads and cleans the data
    """
    df = pd.read_csv(path)

    #Filtering for orders with at least 5 purchases
    df_orders = df.groupby(by='order_id')['outcome'].sum()
    valid_orders = df_orders[df_orders>=5].index.to_list() 
    df = df[df['order_id'].isin(valid_orders)]

    #We remove the ID Columns and Categorical
    feature_columns = ['user_order_seq',
       'ordered_before', 'abandoned_before', 'active_snoozed',
       'set_as_regular', 'normalised_price', 'discount_pct', 'global_popularity',
        'count_adults', 'count_children', 'count_babies',
       'count_pets', 'people_ex_baby', 'days_since_purchase_variant_id',
       'avg_days_to_buy_variant_id', 'std_days_to_buy_variant_id',
       'days_since_purchase_product_type', 'avg_days_to_buy_product_type',
       'std_days_to_buy_product_type']
    
    return df['outcome'], df[feature_columns]

def handler_fit(event:dict) -> dict:
    '''
    Trains a Gradient Boosting Decision Tree model based on the parameters specified in event.
    Returns a json with the path of the saved model.
    '''

    if not isinstance(event["model_parametrisation"], dict):
        raise ValueError("Parameter Grid is not a Dictionary.")
   
    model_parametrisation = event["model_parametrisation"]

    train_data_y, train_data_x = load_data('src/module_4/Datasets_Zrive/feature_frame.csv')
    
    pipeline = Pipeline([
        ('clf', XGBClassifier(**model_parametrisation))])
    
    model = pipeline.fit(train_data_x, train_data_y)
    
    today_date = datetime.now().strftime('%Y_%m_%d')
    model_path = f'src/module_4/push_{today_date}.pkl'

    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)
    
    return {
    "statusCode": "200",
    "body": json.dumps(
    {"model_path": [model_path]})}

curr_event = {"model_parametrisation":
              {"n_estimators": 200,
               "learning_rate": 0.01}}

print(handler_fit(curr_event))