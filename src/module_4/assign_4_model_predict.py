import pickle
import pandas as pd
import json

def handler_predict(event,model_path):
    user_predictions_df = pd.DataFrame.from_dict(json.loads(event["users"]),orient='index')
    
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    
    predictions = model.predict(user_predictions_df)

    json_data = {}
    for user, prediction in zip(user_predictions_df.index, predictions):
        json_data[str(user)] = int(prediction)
    
    return {
    "statusCode": "200",
    "body": json.dumps(
        { "prediction": json_data})
        }

event=dict()
event['users'] = json.dumps({"User_ID_1": 
                 {
    "user_order_seq": 3.000000,
    "ordered_before": 0.000000,
    "abandoned_before": 0.000000,
    "active_snoozed": 0.000000,
    "set_as_regular": 0.000000,
    "normalised_price": 0.081052,
    "discount_pct": 0.053512,
    "global_popularity": 0.000000,
    "count_adults": 2.000000,
    "count_children": 0.000000,
    "count_babies": 0.000000,
    "count_pets": 0.000000,
    "people_ex_baby": 2.000000,
    "days_since_purchase_variant_id": 33.000000,
    "avg_days_to_buy_variant_id": 42.000000,
    "std_days_to_buy_variant_id": 31.134053,
    "days_since_purchase_product_type": 30.000000,
    "avg_days_to_buy_product_type": 30.000000,
    "std_days_to_buy_product_type": 24.276180},
    "User_ID_2":
    {"user_order_seq": 3.000000,
    "ordered_before": 0.000000,
    "abandoned_before": 0.000000,
    "active_snoozed": 0.000000,
    "set_as_regular": 0.000000,
    "normalised_price": 0.081052,
    "discount_pct": 0.053512,
    "global_popularity": 0.000000,
    "count_adults": 2.000000,
    "count_children": 0.000000,
    "count_babies": 0.000000,
    "count_pets": 0.000000,
    "people_ex_baby": 2.000000,
    "days_since_purchase_variant_id": 33.000000,
    "avg_days_to_buy_variant_id": 42.000000,
    "std_days_to_buy_variant_id": 31.134053,
    "days_since_purchase_product_type": 30.000000,
    "avg_days_to_buy_product_type": 30.000000,
    "std_days_to_buy_product_type": 24.276180}
})

print(handler_predict(event,'src/module_4/push_2023_10_22.pkl'))