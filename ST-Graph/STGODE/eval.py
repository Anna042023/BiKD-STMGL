import numpy as np

def mask_np(array, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~np.isnan(array)
    else:
        mask = np.not_equal(array, null_val)
    mask = mask.astype('float32')
    if mask.mean() == 0:
        # 避免除零
        mask[:] = 1.0
    return mask

def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mape = np.abs((y_pred - y_true) / (y_true + 1e-5))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100

def masked_rmse_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mse = (y_true - y_pred) ** 2
    mse = np.nan_to_num(mask * mse)
    return np.sqrt(np.mean(mse))

def masked_mae_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mae = np.abs(y_true - y_pred)
    mae = np.nan_to_num(mask * mae)
    return np.mean(mae)
