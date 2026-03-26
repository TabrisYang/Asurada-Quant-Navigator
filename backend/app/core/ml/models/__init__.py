"""阿斯拉量化系統 — ML 模型實作

模型在此目錄下各自一個檔案，透過 @model_registry.register 自動註冊。
匯入此 __init__.py 即可觸發所有模型的註冊。
"""

from app.core.ml.models.lightgbm_predictor import *   # noqa: F401,F403
from app.core.ml.models.xgboost_predictor import *     # noqa: F401,F403
from app.core.ml.models.random_forest_predictor import * # noqa: F401,F403
from app.core.ml.models.lstm_predictor import *         # noqa: F401,F403
from app.core.ml.models.transformer_predictor import *  # noqa: F401,F403
from app.core.ml.models.gru_predictor import *          # noqa: F401,F403
from app.core.ml.models.cnn_lstm_predictor import *     # noqa: F401,F403
