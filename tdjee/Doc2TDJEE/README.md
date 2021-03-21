# Doc2TDJEE

**训练模型命令**
```
./train_multi.sh 1 --task_name DCFEE_4 --num_train_epochs 20 --model_type DCFEE --cpt_file_name DCFEE --train_batch_size 32 --gradient_accumulation_steps 8 --loss_lambda 0.1 --schedule_epoch_start 20 --schedule_epoch_length 20 --summary_dir_name ./logs/Summary
```

**训练结果**
```
ModelType                EquityFreeze            EquityRepurchase        EquityUnderweight       EquityOverweight        EquityPledge            Average               Total (micro)
DCFEE-O                  & 57.4 & 52.1 & 54.6    & 88.3 & 85.1 & 86.7    & 63.1 & 45.5 & 52.8    & 54.9 & 51.9 & 53.4    & 70.2 & 66.2 & 68.1    & 66.8 & 60.1 & 63.1  & 72.4 & 67.6 & 69.9
2020-04-02 20:35:16 - INFO - dee.dee_helper -   Resume decoded results from ./Exps/DCFEE_4/Output/dee_eval.test.pred_span.DCFEE-O.16.pkl
=============== Single vs. Multi (%) (avg_type=micro) ===============
ModelType                EquityFreeze            EquityRepurchase        EquityUnderweight       EquityOverweight        EquityPledge            Total (micro)         Average
DCFEE-O                  & 61.3 & 48.0           & 89.7 & 57.9           & 59.8 & 42.8           & 57.4 & 48.1           & 78.1 & 61.9           & 79.2 & 58.9         & 69.3 & 51.7
```

**预测命令**
```
./test_multi.sh 1 --task_name DCFEE_4
```