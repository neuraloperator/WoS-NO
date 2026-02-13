#!/bin/bash

# Script to run training with different loss configurations
# All training goes through scripts/main.py

echo "=== MCGINO Training with Different Loss Configurations ==="

# 1. Training with MSE only (baseline)
echoRunning training with MSE only loss..."
cd scripts
python main.py loss=mse_only model=gino dataset=linear_poisson2 train.n_epochs=50

# 2. Training with PINO loss
echoRunning training with PINO loss..."
python main.py loss=pino model=gino dataset=linear_poisson2 train.n_epochs=50

# 3. Training with DeepRitz loss
echoRunning training with DeepRitz loss..."
python main.py loss=deepritz model=gino dataset=linear_poisson2 train.n_epochs=50

# 4. Training with combined physics-informed losses (MSE + PINO + DeepRitz)
echoRunning training with physics-informed losses..."
python main.py loss=physics_informed model=gino dataset=linear_poisson2 train.n_epochs=50

# 5. Training with PINO-WOS combined loss
echoRunning training with PINO-WOS combined loss..."
python main.py loss=pino_wos model=gino dataset=linear_poisson2 train.n_epochs=506g the full physics-informed configuration
echoRunning training with full physics-informed configuration..."
python main.py --config-name mcgino_physics_informed train.n_epochs=50

echo "=== Training completed ===" 