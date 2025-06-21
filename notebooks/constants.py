TARGET = "price"
TEST_PERCENTAGE = 0.2
SEED = 0
DROP_FEATURES = ["Unnamed: 0","flight"]
STRATIFY_FEATURE = "class"
NOMINAL_FEATURES = ["airline","source_city","destination_city"]
ORDINAL_FEATURES = ["departure_time","stops","arrival_time","class"]
NUMERICAL_FEATURES = ["duration","days_left"]
CUSTOM_ORDER = {
    "departure_time": ["Early_Morning","Morning","Afternoon","Evening","Night","Late_Night"],
    "stops": ["zero","one","two_or_more"],
    "arrival_time": ["Early_Morning","Morning","Afternoon","Evening","Night","Late_Night"],
    "class": ["Economy","Business"]
}

BASE_MODEL_NAME = "Linear Regression"
MODEL_01_NAME = "Decission Tree Regression"
MODEL_02_NAME = "K-Neighbors Regression"
TRAINING_DATA_LABEL = "Training Data"
TESTING_DATA_LABEL = "Testing Data"