conda activate physParamInference

cd ../PhysParamInference

for i in $(seq 0 4); do 
    PYTHONPATH=. python training_pendulum_cvdl.py
done

for i in $(seq 0 4); do 
    PYTHONPATH=. python training_sliding_block_cvdl.py
done

for i in $(seq 0 4); do 
    PYTHONPATH=. python training_bouncing_ball_drop_cvdl.py
done

for i in $(seq 0 4); do 
    PYTHONPATH=. python training_ball_throw_cvdl.py
done
