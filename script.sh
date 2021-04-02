# python run.py -c configs/train_vae.yaml

python run_influence.py --start_test_idx=0 --end_test_idx=99 --config=configs/test_vae.yaml --influence_metric="cosine" --model_ckpt_dir=logs/VanillaVAE/version_0/checkpoints/ --output_dir=vanilla_vae_cosine_IF/

python run_influence.py --start_test_idx=0 --end_test_idx=99 --config=configs/test_vae.yaml --influence_metric="dotprod" --model_ckpt_dir=logs/VanillaVAE/version_0/checkpoints/ --output_dir=vanilla_vae_dotprod_IF/
