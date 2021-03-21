import argparse
import os
import torch.distributed as dist

from dee.utils import set_basic_log_config, strtobool, pre_process_data, save_predict_result
from dee.dee_task import DEETask, DEETaskSetting

set_basic_log_config()


def parse_args(in_args=None):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--task_name', type=str, required=True,
                            help='Take Name')
    arg_parser.add_argument('--data_dir', type=str, default='./Data',
                            help='Data directory')
    arg_parser.add_argument('--exp_dir', type=str, default='./Exps',
                            help='Experiment directory')
    arg_parser.add_argument('--save_cpt_flag', type=strtobool, default=True,
                            help='Whether to save cpt for each epoch')
    arg_parser.add_argument('--skip_train', type=strtobool, default=True,
                            help='Whether to skip training')
    arg_parser.add_argument('--eval_model_names', type=str, default='DCFEE-O',
                            help="Models to be evaluated, seperated by ','")
    arg_parser.add_argument('--re_eval_flag', type=strtobool, default=False,
                            help='Whether to re-evaluate previous predictions')

    # add task setting arguments
    for key, val in DEETaskSetting.base_attr_default_pairs:
        if isinstance(val, bool):
            arg_parser.add_argument('--' + key, type=strtobool, default=val)
        else:
            arg_parser.add_argument('--' + key, type=type(val), default=val)

    arg_info = arg_parser.parse_args(args=in_args)

    return arg_info


if __name__ == '__main__':
    in_argv = parse_args()

    task_dir = os.path.join(in_argv.exp_dir, in_argv.task_name)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir, exist_ok=True)

    in_argv.model_dir = os.path.join(task_dir, "Model")
    in_argv.output_dir = os.path.join(task_dir, "Output")

    # in_argv must contain 'data_dir', 'model_dir', 'output_dir'
    dee_setting = DEETaskSetting(
        **in_argv.__dict__
    )

    # build task
    dee_task = DEETask(dee_setting, load_train=not in_argv.skip_train)
    dee_task.logging('Skip training')

    latest_epoch = '16'
    dee_task.load_model_from_cpt(latest_epoch)
    dee_task.logging('loading model success...')

    # 将测试语料中模型预测的结果和实际的结果使用 json 格式的数据保存起来
    file_load_path = './Data/test.json'
    file_save_path = './Exps/DCFEE_4/Result'
    file_save_name = 'results.txt'

    data = pre_process_data(file_load_path)

    result = dee_task.predict(data)

    save_predict_result(file_save_path, file_save_name, result)

    if dist.is_initialized():
        dist.barrier()
    dee_task.logging('done...')
