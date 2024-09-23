source ../.venv/bin/activate

cd ../paig

for i in $(seq 0 4); do 
    PYTHONPATH=. python runners/run_physics.py --task=pendulum --model=PhysicsNet --epochs=500 --batch_size=100 --save_dir=out_pendulum_$i --autoencoder_loss=3.0 --base_lr=3e-4 --anneal_lr=true --color=true --eval_every_n_epochs=10 --print_interval=100 --debug=false --use_ckpt=false 
done

for i in $(seq 0 4); do 
    PYTHONPATH=. python runners/run_physics.py --task=bouncing_ball_drop --model=PhysicsNet --epochs=500 --batch_size=100 --save_dir=bouncing_ball_drop_$i --autoencoder_loss=3.0 --base_lr=3e-4 --anneal_lr=true --color=true --eval_every_n_epochs=10 --print_interval=100 --debug=false --use_ckpt=false 
done

for i in $(seq 0 4); do 
    PYTHONPATH=. python runners/run_physics.py --task=ball_throw --model=PhysicsNet --epochs=500 --batch_size=100 --save_dir=ball_throw_$i --autoencoder_loss=3.0 --base_lr=3e-4 --anneal_lr=true --color=true --eval_every_n_epochs=10 --print_interval=100 --debug=false --use_ckpt=false 
done

for i in $(seq 0 4); do 
    PYTHONPATH=. python runners/run_physics.py --task=sliding_block --model=PhysicsNet --epochs=500 --batch_size=100 --save_dir=sliding_block_$i --autoencoder_loss=3.0 --base_lr=3e-4 --anneal_lr=true --color=true --eval_every_n_epochs=10 --print_interval=100 --debug=false --use_ckpt=false 
done