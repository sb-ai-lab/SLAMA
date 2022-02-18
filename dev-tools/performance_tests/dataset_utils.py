from typing import Any, Dict


def datasets() -> Dict[str, Any]:
    all_datastes = {
        "used_cars_dataset": {
            "path": "/opt/spark_data/small_used_cars_data.csv",
            "task_type": "reg",
            "metric_name": "mse",
            "target_col": "price",
            "roles": {
                "target": "price",
                "drop": ["dealer_zip", "description", "listed_date",
                         "year", 'Unnamed: 0', '_c0',
                         'sp_id', 'sp_name', 'trimId',
                         'trim_name', 'major_options', 'main_picture_url',
                         'interior_color', 'exterior_color'],
                # "numeric": ['latitude', 'longitude', 'mileage']
                "numeric": ['longitude', 'mileage']
            },
            "dtype": {
                'fleet': 'str', 'frame_damaged': 'str',
                'has_accidents': 'str', 'isCab': 'str',
                'is_cpo': 'str', 'is_new': 'str',
                'is_oemcpo': 'str', 'salvage': 'str', 'theft_title': 'str', 'franchise_dealer': 'str'
            }
        },

        "lama_test_dataset": {
            "path": "/opt/spark_data/sampled_app_train.csv",
            "task_type": "binary",
            "metric_name": "areaUnderROC",
            "target_col": "TARGET",
            "roles": {"target": "TARGET", "drop": ["SK_ID_CURR"]},
        },

        # https://www.openml.org/d/734
        "ailerons_dataset": {
            "path": "/opt/spark_data/ailerons.csv",
            "task_type": "binary",
            "metric_name": "areaUnderROC",
            "target_col": "binaryClass",
            "roles": {"target": "binaryClass"},
        },

        # https://www.openml.org/d/4534
        "phishing_websites_dataset": {
            "path": "/opt/spark_data/PhishingWebsites.csv",
            "task_type": "binary",
            "metric_name": "areaUnderROC",
            "target_col": "Result",
            "roles": {"target": "Result"},
        },

        # https://www.openml.org/d/981
        "kdd_internet_usage": {
            "path": "/opt/spark_data/kdd_internet_usage.csv",
            "task_type": "binary",
            "metric_name": "areaUnderROC",
            "target_col": "Who_Pays_for_Access_Work",
            "roles": {"target": "Who_Pays_for_Access_Work"},
        },

        # https://www.openml.org/d/42821
        "nasa_dataset": {
            "path": "/opt/spark_data/nasa_phm2008.csv",
            "task_type": "reg",
            "metric_name": "mse",
            "target_col": "class",
            "roles": {"target": "class"},
        },

        # https://www.openml.org/d/4549
        "buzz_dataset": {
            "path": "/opt/spark_data/Buzzinsocialmedia_Twitter_25k.csv",
            "task_type": "reg",
            "metric_name": "mse",
            "target_col": "Annotation",
            "roles": {"target": "Annotation"},
        },

        # https://www.openml.org/d/372
        "internet_usage": {
            "path": "/opt/spark_data/internet_usage.csv",
            "task_type": "multiclass",
            "metric_name": "ova",
            "target_col": "Actual_Time",
            "roles": {"target": "Actual_Time"},
        },

        # https://www.openml.org/d/4538
        "gesture_segmentation": {
            "path": "/opt/spark_data/gesture_segmentation.csv",
            "task_type": "multiclass",
            "metric_name": "ova",
            "target_col": "Phase",
            "roles": {"target": "Phase"},
        },

        # https://www.openml.org/d/382
        "ipums_97": {
            "path": "/opt/spark_data/ipums_97.csv",
            "task_type": "multiclass",
            "metric_name": "ova",
            "target_col": "movedin",
            "roles": {"target": "movedin"},
        }
    }

    return all_datastes
