To produce the train model-
1. Ensure all files are available:
1.1 model.py
1.2 test.py
1.3 train.py
1.4 structured_iterative_prune.py

2. Ensure the best baseline model is available i.e. best_mobilenetv2_ema.pth

3. Run the following command to train-
python structured_iterative_prune.py --checkpoint best_mobilenetv2_ema.pth --iterations 8 --prune-per-iter 0.05 --epochs-per-iter 10 --batch-size 32 --lr 0.0005 --save-dir .\prune_runs_structure
