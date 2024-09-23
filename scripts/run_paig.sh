source ../.venv/bin/activate

cd ../paig

# PYTHONPATH=. python runners/run_physics.py --task=pendulum --model=PhysicsNet --epochs=500 --batch_size=100 --save_dir=out_pendulum_perfect --autoencoder_loss=3.0 --base_lr=3e-4 --anneal_lr=true --color=true --eval_every_n_epochs=10 --print_interval=100 --debug=false --use_ckpt=false 
# PYTHONPATH=. python runners/run_physics.py --task=pendulum_real --model=PhysicsNet --epochs=500 --batch_size=100 --save_dir=out_pendulum_real_perfect --autoencoder_loss=3.0 --base_lr=3e-4 --anneal_lr=true --color=true --eval_every_n_epochs=10 --print_interval=100 --debug=false --use_ckpt=false 
# PYTHONPATH=. python runners/run_physics.py --task=bouncing_ball_drop --model=PhysicsNet --epochs=500 --batch_size=100 --save_dir=out_bouncing_ball_drop_perfect --autoencoder_loss=3.0 --base_lr=3e-4 --anneal_lr=true --color=true --eval_every_n_epochs=10 --print_interval=100 --debug=false --use_ckpt=false 
# PYTHONPATH=. python runners/run_physics.py --task=bouncing_ball_drop_real --model=PhysicsNet --epochs=500 --batch_size=100 --save_dir=out_bouncing_ball_drop_real_perfect --autoencoder_loss=3.0 --base_lr=3e-4 --anneal_lr=true --color=true --eval_every_n_epochs=10 --print_interval=100 --debug=false --use_ckpt=false 
# PYTHONPATH=. python runners/run_physics.py --task=ball_throw --model=PhysicsNet --epochs=500 --batch_size=100 --save_dir=out_ball_throw_perfect --autoencoder_loss=3.0 --base_lr=3e-4 --anneal_lr=true --color=true --eval_every_n_epochs=10 --print_interval=100 --debug=false --use_ckpt=false 
PYTHONPATH=. python runners/run_physics.py --task=sliding_block --model=PhysicsNet --epochs=500 --batch_size=100 --save_dir=out_sliding_block_perfect --autoencoder_loss=3.0 --base_lr=3e-4 --anneal_lr=true --color=true --eval_every_n_epochs=10 --print_interval=100 --debug=false --use_ckpt=false 
# PYTHONPATH=. python runners/run_physics.py --task=sliding_block_real --model=PhysicsNet --epochs=500 --batch_size=100 --save_dir=out_sliding_block_real_perfect --autoencoder_loss=3.0 --base_lr=3e-4 --anneal_lr=true --color=true --eval_every_n_epochs=10 --print_interval=100 --debug=false --use_ckpt=false 


