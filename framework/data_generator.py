import numpy as np
import h5py, os, pickle
import csv
import time
from framework.utilities import calculate_scalar, scale
import framework.config as config


class DataGenerator_scene_event(object):

    def __init__(self, hdf5_path, label_type, batch_size, batchnormal, dev_train_csv=None,
                 dev_validate_csv=None, seed=1234):
        """
        Inputs:
          hdf5_path: str
          batch_size: int
          dev_train_csv: str | None, if None then use all data for training
          dev_validate_csv: str | None, if None then use all data for training
          seed: int, random seed
        """

        self.batch_size = batch_size

        self.random_state = np.random.RandomState(seed)
        self.validate_random_state = np.random.RandomState(0)
        lb_to_ix = config.lb_to_ix

        # Load data
        load_time = time.time()
        hf = h5py.File(hdf5_path, 'r')

        self.audio_names = np.array([s.decode() for s in hf['filename'][:]])
        # print(self.audio_names)

        self.pseudo = False
        file = os.path.join(os.getcwd(), 'pseudo_labels.pkl')
        self.pseudo = True

        if self.pseudo:
            data = pickle.load(open(file, "rb"))
            self.pseudo_tagging_labels = [data[name.split('.wav')[0]] for name in self.audio_names]
            self.pseudo_tagging_labels = np.array(self.pseudo_tagging_labels)
            self.pseudo_tagging_labels = self.pseudo_tagging_labels[:, config.events_index]

        self.x = hf['feature'][:]
        self.scene_labels = [s.decode() for s in hf['scene_label'][:]]
        self.identifiers = [s.decode() for s in hf['identifier'][:]]
        self.source_labels = [s.decode() for s in hf['source_label']]
        self.y = np.array([lb_to_ix[lb] for lb in self.scene_labels])

        hf.close()
        print('Loading data time: {:.3f} s'.format(time.time() - load_time))

        # Use all data for training
        if dev_train_csv is None and dev_validate_csv is None:

            self.train_audio_indexes = np.arange(len(self.audio_names))
            print('Use all development data for training. ')

        # Split data to training and validation
        else:

            self.train_audio_indexes = self.get_audio_indexes_from_csv(
                dev_train_csv)

            self.validate_audio_indexes = self.get_audio_indexes_from_csv(
                dev_validate_csv)

            print('Split development data to {} training and {} '
                         'validation data. '.format(len(self.train_audio_indexes),
                                                    len(self.validate_audio_indexes)))

        self.batchnormal = batchnormal
        if not batchnormal:
            # Calculate scalar
            (self.mean, self.std) = calculate_scalar(
                self.x[self.train_audio_indexes])

    def get_audio_indexes_from_csv(self, csv_file):
        """Calculate indexes from a csv file.

        Args:
          csv_file: str, path of csv file
        """

        with open(csv_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            lis = list(reader)

        audio_indexes = []

        for li in lis:
            audio_name = li[0].split('/')[1]

            if audio_name in self.audio_names:
                audio_index = np.where(self.audio_names == audio_name)[0][0]
                audio_indexes.append(audio_index)

        return audio_indexes

    def generate_train(self):
        """Generate mini-batch data for training.

        Returns:
          batch_x: (batch_size, seq_len, freq_bins)
          batch_y: (batch_size,)
        """

        batch_size = self.batch_size
        audio_indexes = np.array(self.train_audio_indexes)
        audios_num = len(audio_indexes)

        self.random_state.shuffle(audio_indexes)

        iteration = 0
        pointer = 0

        while True:
            # Reset pointer
            if pointer >= audios_num:
                pointer = 0
                self.random_state.shuffle(audio_indexes)

            # Get batch indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + batch_size]
            pointer += batch_size

            iteration += 1

            batch_x = self.x[batch_audio_indexes]
            batch_y = self.y[batch_audio_indexes]

            if self.pseudo:
                batch_y_event = self.pseudo_tagging_labels[batch_audio_indexes]

            if not self.batchnormal:
                batch_x = self.transform(batch_x)

            if self.pseudo:
                yield batch_x, batch_y, batch_y_event
            else:
                yield batch_x, batch_y

    def generate_validate(self, data_type, devices, shuffle,
                          max_iteration=None):
        """Generate mini-batch data for evaluation.

        Args:
          data_type: 'train' | 'validate'
          devices: list of devices, e.g. ['a'] | ['a', 'b', 'c']
          max_iteration: int, maximum iteration for validation
          shuffle: bool

        Returns:
          batch_x: (batch_size, seq_len, freq_bins)
          batch_y: (batch_size,)
          batch_audio_names: (batch_size,)
        """

        batch_size = self.batch_size

        if data_type == 'train':
            audio_indexes = np.array(self.train_audio_indexes)

        elif data_type == 'validate':
            audio_indexes = np.array(self.validate_audio_indexes)

        else:
            raise Exception('Invalid data_type!')

        if shuffle:
            self.validate_random_state.shuffle(audio_indexes)

        # Get indexes of specific devices
        devices_specific_indexes = []

        for n in range(len(audio_indexes)):
            if self.source_labels[audio_indexes[n]] in devices:
                devices_specific_indexes.append(audio_indexes[n])

        print('Number of {} audios in specific devices {}: {}'.format(
            data_type, devices, len(devices_specific_indexes)))

        audios_num = len(devices_specific_indexes)

        iteration = 0
        pointer = 0

        while True:
            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= audios_num:
                break

            # Get batch indexes
            batch_audio_indexes = devices_specific_indexes[
                                  pointer: pointer + batch_size]
            # print(batch_audio_indexes)
            pointer += batch_size

            iteration += 1

            batch_x = self.x[batch_audio_indexes]
            batch_y = self.y[batch_audio_indexes]
            batch_audio_names = self.audio_names[batch_audio_indexes]

            if self.pseudo:
                batch_y_event = self.pseudo_tagging_labels[batch_audio_indexes]

            if not self.batchnormal:
                # Transform data
                batch_x = self.transform(batch_x)

            if self.pseudo:
                yield batch_x, batch_y, batch_y_event, batch_audio_names
            else:
                yield batch_x, batch_y, batch_audio_names

    def transform(self, x):
        """Transform data.

        Args:
          x: (batch_x, seq_len, freq_bins) | (seq_len, freq_bins)

        Returns:
          Transformed data.
        """

        return scale(x, self.mean, self.std)





