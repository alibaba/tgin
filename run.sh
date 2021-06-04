mkdir dnn_save_path
mkdir dnn_best_model
nohup python script/train.py train & 
nohup python script/train.py test & 