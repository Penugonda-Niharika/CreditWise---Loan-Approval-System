import pandas as pd

def add_engineered_features(frame):
    """Add the squared features used by the CreditWise model."""
    frame = frame.copy()
    frame["DTI_Ratio_sq"] = frame["DTI_Ratio"] ** 2
    frame["Credit_Score_sq"] = frame["Credit_Score"] ** 2
    return frame
