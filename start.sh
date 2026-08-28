#!/bin/bash

# 安装必要的依赖
pip install Flask pandas numpy scikit-learn xgboost lightgbm catboost scipy

# 启动Flask应用
export FLASK_APP=app.py
export FLASK_ENV=development
flask run --host=0.0.0.0 --port=5000