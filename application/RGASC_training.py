import sys, os

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# from framework.feature import calculate_features
from framework.models_pytorch import *

from framework.data_generator import *
from framework.processing import *
import numpy as np
import argparse

cuda = config.cuda


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main(argv):
    # calculate_features()

    # 创建一个参数解析实例
    parser = argparse.ArgumentParser()
    # 添加参数解析
    parser.add_argument('-alpha', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('-shared_layers', type=int, required=True)
    parser.add_argument('-lr_decay', action="store_true")
    parser.add_argument('-times', type=int, required=True)
    parser.add_argument('-lr_rate', type=float, required=True)
    parser.add_argument('-batch', type=int, required=True)
    args = parser.parse_args()

    lr_init = args.lr_rate  # 1e-3
    config.shared_layers = args.shared_layers
    config.cuda_seed = 9
    times = args.times
    config.batch_size = args.batch

    accumulation_steps = int(config.batch_size / config.batch_size)
    epochs = 100

    sys_name = 'sys_lambda_' + str(times) + '_'

    lr_decay = args.lr_decay

    itera_step = 200

    models = [Cnn14_asc_aec]
    trainings = [training_flow]

    Model = models[config.shared_layers]
    training = trainings[config.shared_layers]

    if config.cuda_seed:
        np.random.seed(config.cuda_seed)
        torch.manual_seed(config.cuda_seed)
        if config.cuda:
            torch.cuda.manual_seed(config.cuda_seed)
            torch.cuda.manual_seed_all(config.cuda_seed)

    if not lr_decay:
        basic_name = sys_name + str(lr_init).replace('-', '') + '_nolrdecay'
    else:
        basic_name = sys_name + str(lr_init).replace('-', '')

    suffix, system_name = define_system_name(basic_name=basic_name, alpha=args.alpha, epochs=epochs)
    system_path = os.path.join(os.getcwd(), system_name)

    holdout_fold = 1
    dev_train_csv = os.path.join(config.subdir, 'evaluation_setup', 'fold{}_train.txt'.format(holdout_fold))
    dev_validate_csv = os.path.join(config.subdir, 'evaluation_setup', 'fold{}_evaluate.txt'.format(holdout_fold))

    batchnormal = False
    if batchnormal:
        models_dir = os.path.join(system_path, 'mdbn')
    else:
        models_dir = os.path.join(system_path, 'md')

    label_type = '_hard' + '_com' + str(config.shared_layers) + '_' + suffix
    models_dir += label_type

    log_path = models_dir + '_log'

    create_folder(log_path)
    filename = 'com' + str(args.shared_layers) + os.path.basename(__file__).split('.py')[0].split('RGASC')[1]
    print_log_file = os.path.join(log_path, filename + '_print.log')
    sys.stdout = Logger(print_log_file, sys.stdout)
    console_log_file = os.path.join(log_path, filename + '_console.log')
    sys.stderr = Logger(console_log_file, sys.stderr)

    classes_num = len(config.labels)
    event_class = len(config.event_labels)
    model = Model(classes_num, event_class)

    print(model)
    if cuda:
        model.cuda()

    # Data generator
    data_type = 'development'
    hdf5_path = os.path.join(config.dataset_dir, 'acoustic_feature')
    hdf5_file = os.path.join(hdf5_path, '{}.h5'.format(data_type))
    generator = DataGenerator_scene_event(hdf5_path=hdf5_file,
                                          label_type=label_type,
                                          batch_size=config.batch_size,
                                          batchnormal=batchnormal,
                                          dev_train_csv=dev_train_csv,
                                          dev_validate_csv=dev_validate_csv)

    cnn_lr_init = 1e-3
    weight_decay = 0.
    training(generator, model, cuda, models_dir, fc_lr_init=lr_init,  log_path=log_path,
             cnn_lr_init=cnn_lr_init, weight_decay=weight_decay, itera_step=itera_step,
             lr_decay=lr_decay, alpha=args.alpha, accumulation_steps=accumulation_steps,
             epochs=epochs)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)















