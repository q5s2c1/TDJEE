# Doc2TDJEE

**训练模型命令**
```
./train_multi.sh 1 --task_name DCFEE_4 --num_train_epochs 20 --model_type DCFEE --cpt_file_name DCFEE --train_batch_size 32 --gradient_accumulation_steps 8 --loss_lambda 0.1 --schedule_epoch_start 20 --schedule_epoch_length 20 --summary_dir_name ./logs/Summary
```

**预测命令**
```
./test_multi.sh 1 --task_name DCFEE_4
```
