conda activate physParamInference

cd ../PhysParamInference

PYTHONPATH=. python training_pendulum_cvdl.py
PYTHONPATH=. python training_sliding_block_cvdl.py
PYTHONPATH=. python training_bouncing_ball_drop_cvdl.py
PYTHONPATH=. python training_ball_throw_cvdl.py
