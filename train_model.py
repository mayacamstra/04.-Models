from utils import RMSE

def train_and_evaluate_model(model, data_train, data_train_reg, data_validate, data_validate_reg):
    B_matrix, C_matrix, r2_insample, beta_const = model.train(data_train, data_train_reg)
    
    predictions = model.predict(data_validate, data_validate_reg)
    rmse_value = RMSE(data_validate.T, predictions)
    
    return B_matrix, C_matrix, r2_insample, beta_const, rmse_value