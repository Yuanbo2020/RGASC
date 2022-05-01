import os
from time import sleep

gpu_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

alphas = ["1 0 0 0",]

shared_layers = [0]

file = 'RGASC_training.py'
filepath = os.path.join(os.getcwd(), file)

times = 10

lr_rates = [1e-3]

batchs = [64]


for sub_time in range(times):
    each_time = sub_time
    for alpha in alphas:
        for s_layer in shared_layers:
            for lr in lr_rates:
                for batch in batchs:
                    command = 'python {0} -shared_layer {1} ' \
                              '-alpha {2}  -times {3} ' \
                              '-lr_rate {4} -batch {5}'.format(filepath,
                                                               s_layer,
                                                               alpha,
                                                               each_time,
                                                               lr,
                                                               batch)
                    print(command)
                    if os.system(command):  
                        print('\nFailed: ', command, '\n')
                        sleep(100)
            sleep(100)



