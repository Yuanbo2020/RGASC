import time, os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from framework.utilities import create_folder, calculate_accuracy, calculate_confusion_matrix, print_accuracy, \
    plot_confusion_matrix, plot_confusion_matrix_each_file
from framework.models_pytorch import move_data_to_gpu
import framework.config as config
from sklearn import metrics


def inference(models_dir, model, generator, filename, cuda=1, test_type='asc'):

    labels = config.labels

    model_file = os.path.join(models_dir, filename)
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['state_dict'])

    if cuda:
        model.cuda()

    # Predict & evaluate
    for device in config.devices:
        print('Device: {}'.format(device))

        generate_func = generator.generate_validate(data_type='validate',
                                                    devices=device,
                                                    shuffle=False)

        # Inference
        dict = forward_asc_aec(model=model,
                       generate_func=generate_func,
                       cuda=cuda,
                       return_target=True)

        if test_type=='asc':
            outputs = dict['output']  # (audios_num, classes_num)
            targets = dict['target']  # (audios_num, classes_num)
            predictions = np.argmax(outputs, axis=-1)  # (audios_num,)
            classes_num = outputs.shape[-1]
            confusion_matrix = calculate_confusion_matrix(targets, predictions, classes_num)
            class_wise_accuracy = calculate_accuracy(targets, predictions, classes_num)
            plot_confusion_matrix_each_file(models_dir,
                                            confusion_matrix,
                                            title='Device {}'.format(device.upper()),
                                            labels=labels,
                                        values=class_wise_accuracy, filename=filename)
            return np.mean(class_wise_accuracy)
        elif test_type=='aec':
            outputs_event = dict['outputs_event']  # (audios_num, classes_num)
            targets_event = dict['targets_event']  # (audios_num, classes_num)
            aucs = []
            for i in range(targets_event.shape[0]):
                test_y_auc, pred_auc = targets_event[i, :], outputs_event[i, :]
                if np.sum(test_y_auc):
                    test_auc = metrics.roc_auc_score(test_y_auc, pred_auc)
                    aucs.append(test_auc)
            final_auc = sum(aucs) / len(aucs)
            return final_auc



def forward_asc_aec(model, generate_func, cuda, return_target):
    """Forward data to a model.

    Args:
      generate_func: generate function
      cuda: bool
      return_target: bool

    Returns:
      dict, keys: 'audio_name', 'output'; optional keys: 'target'
    """

    outputs = []
    outputs_event = []
    audio_names = []

    if return_target:
        targets = []
        targets_event = []

    no_event = True
    for data in generate_func:
        if return_target:
            if len(data) == 4:
                (batch_x, batch_y, batch_y_event, batch_audio_names) = data
                no_event = False
            elif len(data) == 3:
                (batch_x, batch_y, batch_audio_names) = data

        else:
            if len(data) == 3:
                (batch_x, batch_y_event, batch_audio_names) = data
                no_event = False
            elif len(data) == 2:
                (batch_x, batch_audio_names) = data

        batch_x = move_data_to_gpu(batch_x, cuda)

        with torch.no_grad():
            batch_output, batch_output_event= model(batch_x)
            batch_output = F.softmax(batch_output, dim=-1)
        outputs.append(batch_output.data.cpu().numpy())
        outputs_event.append(batch_output_event.data.cpu().numpy())

        audio_names.append(batch_audio_names)

        if return_target:
            targets.append(batch_y)
            if not no_event:
                targets_event.append(batch_y_event)

    dict = {}

    if len(outputs):
        outputs = np.concatenate(outputs, axis=0)
    dict['output'] = outputs

    if len(outputs_event):
        outputs_event = np.concatenate(outputs_event, axis=0)
    dict['outputs_event'] = outputs_event

    audio_names = np.concatenate(audio_names, axis=0)
    dict['audio_name'] = audio_names

    if return_target:
        targets = np.concatenate(targets, axis=0)
        dict['target'] = targets
        if len(targets_event):
            targets_event = np.concatenate(targets_event, axis=0)
            dict['targets_event'] = targets_event

    return dict


def evaluate_asc_aec(model, generator, data_type, devices, max_iteration, cuda, crossentropy=False, criterion=None):
    """Evaluate

    Args:
      model: object.
      generator: object.
      data_type: 'train' | 'validate'.
      devices: list of devices, e.g. ['a'] | ['a', 'b', 'c']
      max_iteration: int, maximum iteration for validation
      cuda: bool.

    Returns:
      accuracy: float
    """

    # Generate function
    generate_func = generator.generate_validate(data_type=data_type,
                                                devices=devices,
                                                shuffle=True,
                                                max_iteration=max_iteration)

    # # Forward
    dict = forward_asc_aec(model=model,
                          generate_func=generate_func,
                          cuda=cuda,
                          return_target=True)

    outputs = dict['output']  # (audios_num, classes_num)
    targets = dict['target']  # (audios_num, classes_num)
    predictions = np.argmax(outputs, axis=-1)  # (audios_num,)
    classes_num = outputs.shape[-1]
    accuracy = calculate_accuracy(targets, predictions, classes_num, average='macro')

    outputs_event = dict['outputs_event']  # (audios_num, classes_num)
    targets_event = dict['targets_event']  # (audios_num, classes_num)
    # print('targets_event.shape: ', targets_event.shape)

    aucs = []
    for i in range(targets_event.shape[0]):
        test_y_auc, pred_auc = targets_event[i, :], outputs_event[i, :]
        if np.sum(test_y_auc):
            test_auc = metrics.roc_auc_score(test_y_auc, pred_auc)
            aucs.append(test_auc)
    final_auc = sum(aucs) / len(aucs)
    # print('Auc:', final_auc)

    return accuracy, final_auc


def define_system_name(alpha=None, basic_name='system', att_dim=None, n_heads=None, epochs=10000000):
    suffix = ''
    if alpha:
        suffix = suffix.join([str(each) for each in alpha]).replace('.', '')

    sys_name = basic_name
    sys_suffix = '_b' + str(config.batch_size) + '_e' + str(epochs) \
                 + '_attd' + str(att_dim) + '_h' + str(n_heads) if att_dim is not None and n_heads is not None\
        else '_b' + str(config.batch_size) + '_e' + str(epochs)

    sys_suffix = sys_suffix + '_cuda' + str(config.cuda_seed) if config.cuda_seed is not None else sys_suffix
    system_name = sys_name + sys_suffix if sys_suffix is not None else sys_name

    return suffix, system_name



def testing(models_dir, model, generator, cuda, test_type):
    model_list = []
    epoch_list = []
    for model_file in os.listdir(models_dir):
        if model_file.endswith(config.endswith):
            model_list.append(model_file)
            epoch_list.append(int(model_file.replace(config.endswith, '').split('_')[-1]))
    index = np.argsort(np.array(epoch_list))
    model_list = [model_list[i] for i in index]

    acc_list = []
    for filename in model_list:
        scene_acc = inference(models_dir, model, generator,
                              filename, cuda=cuda, test_type=test_type)
        acc_list.append(scene_acc)

    filename = os.path.basename(__file__).split('.py')[0]
    output_dir = models_dir + '_test_png'
    create_folder(output_dir)
    if 'aec' in test_type:
        log_file = os.path.join(output_dir, filename + '_aucs.txt')
        with open(log_file, 'w') as f:
            for auc in acc_list:
                f.write(str(auc) + '\n')

    if 'asc' in test_type:
        log_file = os.path.join(output_dir, filename + '_accs.txt')
        with open(log_file, 'w') as f:
            for auc in acc_list:
                f.write(str(auc) + '\n')



def training_flow(generator, model, cuda, models_dir, lr_decay, fc_lr_init = 1e-3,
    cnn_lr_init = 1e-3, weight_decay=1e-6, itera_step=500, log_path=None, alpha=None,
                                 accumulation_steps=0, only_save_best=config.only_save_best,
                                 epochs=0):
    create_folder(models_dir)

    # Optimizer
    conv_params = list(map(id, model.conv_block1.parameters()))
    conv_params += list(map(id, model.conv_block2.parameters()))
    conv_params += list(map(id, model.conv_block3.parameters()))
    conv_params += list(map(id, model.conv_block4.parameters()))
    conv_params += list(map(id, model.conv_block5.parameters()))
    conv_params += list(map(id, model.conv_block6.parameters()))

    conv_params += list(map(id, model.conv_block3_event.parameters()))
    conv_params += list(map(id, model.conv_block4_event.parameters()))
    conv_params += list(map(id, model.conv_block5_event.parameters()))
    conv_params += list(map(id, model.conv_block6_event.parameters()))

    rest_params = filter(lambda x: id(x) not in conv_params, model.parameters())

    optimizer = optim.Adam([
        {'params': rest_params},
        {'params': model.conv_block1.parameters(), 'lr': cnn_lr_init},
        {'params': model.conv_block2.parameters(), 'lr': cnn_lr_init},
        {'params': model.conv_block3.parameters(), 'lr': cnn_lr_init},
        {'params': model.conv_block4.parameters(), 'lr': cnn_lr_init},
        {'params': model.conv_block5.parameters(), 'lr': cnn_lr_init},
        {'params': model.conv_block6.parameters(), 'lr': cnn_lr_init},

        {'params': model.conv_block3_event.parameters(), 'lr': cnn_lr_init},
        {'params': model.conv_block4_event.parameters(), 'lr': cnn_lr_init},
        {'params': model.conv_block5_event.parameters(), 'lr': cnn_lr_init},
        {'params': model.conv_block6_event.parameters(), 'lr': cnn_lr_init},],
        fc_lr_init, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)

    max_val_scene_acc = 0.000001
    max_val_event_auc = 0.000001
    save_best_model = 0

    validate = 1
    check_iter = int(6122 / config.batch_size)
    print('check_iter: ', check_iter)
    print('really batch size: ', config.batch_size)

    tra_acc_scene = []
    val_acc_scene = []
    tra_auc_event = []
    val_auc_event = []

    tra_loss_list = []
    val_loss_list = []

    tra_acc_scene_file = os.path.join(log_path, 'training_scene_acc.txt')
    val_acc_scene_file = os.path.join(log_path, 'validation_scene_acc.txt')

    tra_auc_event_file = os.path.join(log_path, 'training_event_auc.txt')
    val_auc_event_file = os.path.join(log_path, 'validation_event_auc.txt')

    tra_loss_file = os.path.join(log_path, 'training_losses.txt')
    val_loss_file = os.path.join(log_path, 'validation_losses.txt')

    event_loss = torch.nn.BCELoss()
    mse_loss = torch.nn.MSELoss()

    # Train on mini batches
    for iteration, all_data in enumerate(generator.generate_train()):
        # print(len(all_data))

        if len(all_data) == 2:
            (batch_x, batch_y) = all_data
        elif len(all_data) == 3:
            (batch_x, batch_y, batch_y_event) = all_data

        train_bgn_time = time.time()

        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)
        batch_y_event = move_data_to_gpu(batch_y_event, cuda)

        relation_scene_event = move_data_to_gpu(config.relation_scene_event, cuda)
        relation_event_scene = move_data_to_gpu(config.relation_scene_event.T, cuda)

        model.train()
        optimizer.zero_grad()

        x_scene, x_event = model(batch_x)
        x_scene = F.log_softmax(x_scene, dim=-1)
        inferred_event = torch.matmul(x_scene, relation_scene_event)
        inferred_scene = torch.matmul(x_event, relation_event_scene)
        loss_scene = F.nll_loss(x_scene, batch_y)
        loss_event = event_loss(x_event, batch_y_event)
        loss_infer_scene = mse_loss(inferred_scene, x_scene)
        loss_infer_event = mse_loss(inferred_event, x_event)

        if alpha is not None:
            if type(alpha[0]) == str:
                alpha = [float(each) for each in alpha]
                loss_common = alpha[0] * loss_scene + alpha[1] * loss_infer_scene \
                              + alpha[2] * loss_event + alpha[3] * loss_infer_event
            else:
                loss_common = alpha[0] * loss_scene + alpha[1] * loss_infer_scene \
                              + alpha[2] * loss_event + alpha[3] * loss_infer_event
        else:
            loss_common = loss_scene + loss_event

        loss_common.backward()
        optimizer.step()

        print('iter: ', iteration, 'loss: %.5f' % float(loss_common),
              'l_s: %.5f' % float(loss_scene),
              'l_s_by_e: %.5f' % float(loss_infer_scene),
              'l_e: %.5f' % float(loss_event),
              'l_e_by_s: %.5f' % float(loss_infer_event),
              )

        tra_loss_list.append(float(loss_common))

        print('final loss: %.5f' % float(loss_common),
              'l_s: %.5f' % float(alpha[0] * loss_scene),
              'l_s_by_e: %.5f' % float(alpha[1] * loss_infer_scene),
              'l_e: %.5f' % float(alpha[2] * loss_event),
              'l_e_by_s: %.5f' % float(alpha[3] * loss_infer_event), )

        # 6122 / 64 = 95.656
        if iteration % check_iter == 0 and iteration > 0:
            train_fin_time = time.time()
            tr_acc, event_auc = evaluate_asc_aec(model=model,
                                                 generator=generator,
                                                 data_type='train',
                                                 devices=config.devices,
                                                 max_iteration=None,
                                                 cuda=cuda)

            tra_acc_scene.append(tr_acc)
            tra_auc_event.append(event_auc)

            if validate:
                va_acc, va_event_auc = evaluate_asc_aec(model=model,
                                                 generator=generator,
                                                 data_type='validate',
                                                 devices=config.devices,
                                                 max_iteration=None,
                                                 cuda=cuda)
                val_acc_scene.append(va_acc)
                val_auc_event.append(va_event_auc)

                if va_event_auc > max_val_event_auc:
                    max_val_event_auc = va_event_auc
                    max_val_event_auc_itera = iteration

                if va_acc > max_val_scene_acc:
                    max_val_scene_acc = va_acc
                    save_best_model = 1
                    max_val_scene_acc_itera = iteration

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            print('validating iter: ', iteration, 'scene_acc: %.5f' % tr_acc, ' val_scene_acc: %.5f' % va_acc,
                  'event_auc: %.5f' % event_auc, ' val_event_auc: %.5f' % va_event_auc)

            print('validating iter: {}, T_train: {:.5f} s, T_val: {:.5f} s, max_val_scene_acc: {:.5f} , '
                  'max_val_scene_acc_itera: {}, max_val_event_auc: {:.5f} , max_val_event_auc_itera: {}'
                  .format(iteration, train_time, validate_time, max_val_scene_acc, max_val_scene_acc_itera,
                          max_val_event_auc, max_val_event_auc_itera))

            np.savetxt(tra_acc_scene_file, tra_acc_scene, fmt='%.5f')
            np.savetxt(val_acc_scene_file, val_acc_scene, fmt='%.5f')
            np.savetxt(tra_auc_event_file, tra_auc_event, fmt='%.5f')
            np.savetxt(val_auc_event_file, val_auc_event, fmt='%.5f')

        if only_save_best:
            if save_best_model:
                save_best_model = 0
                save_out_dict = {'iteration': iteration, 'state_dict': model.state_dict(),
                                 'optimizer': optimizer.state_dict(),
                                 'scene_valacc': va_acc,
                                 'event_valauc': va_event_auc}
                save_out_path = os.path.join(models_dir, 'scene_best' + config.endswith)
                torch.save(save_out_dict, save_out_path)
                print('Best model saved to {}'.format(save_out_path))

        else:
            check_epoch = int(1000 * accumulation_steps)  # 1000
            if (save_best_model or iteration % check_epoch == 0) and iteration > 0:
                save_best_model = 0
                save_out_dict = {'iteration': iteration, 'state_dict': model.state_dict(),
                                 'optimizer': optimizer.state_dict(),
                                 'scene_valacc': va_acc,
                                 'event_valauc': va_event_auc}
                save_out_path = os.path.join(models_dir,
                                             'scene_{}_event_{}_md_{}'.format('%.4f' % va_acc, va_event_auc,
                                                                              iteration) + config.endswith)
                torch.save(save_out_dict, save_out_path)
                print('Model saved to {}'.format(save_out_path))

        # Reduce learning rate
        check_itera_step = int(itera_step * accumulation_steps)
        if lr_decay and (iteration % check_itera_step == 0 > 0):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9

        # Stop learning
        if iteration > (epochs*(6122 /config.batch_size)):
            if not only_save_best:
                save_out_dict = {'iteration': iteration, 'state_dict': model.state_dict(),
                                 'optimizer': optimizer.state_dict(),
                                 'scene_valacc': va_acc,
                                 'event_valauc': va_event_auc}
                save_out_path = os.path.join(models_dir,
                                             'scene_{}_event_{}_md_{}'.format('%.4f' % va_acc, va_event_auc,
                                                                              iteration) + config.endswith)
                torch.save(save_out_dict, save_out_path)
                print('Model saved to {}'.format(save_out_path))

            final_test = 1
            if final_test:

                tr_acc, event_auc = evaluate_asc_aec(model=model,
                                                 generator=generator,
                                                 data_type='train',
                                                 devices=config.devices,
                                                 max_iteration=None,
                                                 cuda=cuda)
                tra_acc_scene.append(tr_acc)
                tra_auc_event.append(event_auc)

                if validate:
                    va_acc, va_event_auc = evaluate_asc_aec(model=model,
                                                 generator=generator,
                                                 data_type='validate',
                                                 devices=config.devices,
                                                 max_iteration=None,
                                                 cuda=cuda)
                    val_acc_scene.append(va_acc)
                    val_auc_event.append(va_event_auc)

                    if va_event_auc > max_val_event_auc:
                        max_val_event_auc = va_event_auc

                    if va_acc > max_val_scene_acc:
                        max_val_scene_acc = va_acc

                save_out_dict = {'state_dict': model.state_dict()}
                save_out_path = os.path.join(models_dir,
                                             'final_scene_{}_event_{}_md_{}'.format('%.4f' % va_acc, va_event_auc,
                                                                                    iteration) + config.endswith)
                torch.save(save_out_dict, save_out_path)
                print('Fianl Model saved to {}'.format(save_out_path))

                print('iter: ', iteration, 'scene_acc: %.6f' % tr_acc, ' val_scene_acc: %.6f' % va_acc,
                      'event_auc: %.6f' % event_auc, ' val_event_auc: %.6f' % va_event_auc)

                np.savetxt(tra_acc_scene_file, tra_acc_scene, fmt='%.5f')
                np.savetxt(val_acc_scene_file, val_acc_scene, fmt='%.5f')
                np.savetxt(tra_auc_event_file, tra_auc_event, fmt='%.5f')
                np.savetxt(val_auc_event_file, val_auc_event, fmt='%.5f')
                np.savetxt(tra_loss_file, tra_loss_list, fmt='%.5f')

            print('iteration: ', iteration, 'max_val_scene_acc: ', max_val_scene_acc,
                  'max_val_scene_acc_itera: ', max_val_scene_acc_itera,
                  'max_val_event_auc: ', max_val_event_auc,
                  'max_val_event_auc_itera: ', max_val_event_auc_itera)
            print('Training is done!!!')

            print('Fianl: ', iteration, 'scene_acc: %.6f' % tr_acc, ' val_scene_acc: %.6f' % va_acc,
                  'event_auc: %.6f' % event_auc, ' val_event_auc: %.6f' % va_event_auc)

            break


