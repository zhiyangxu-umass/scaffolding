"""
The ``train`` subcommand can be used to train a model.
It requires a configuration file and a directory in
which to write the results.

.. code-block:: bash

   $ python -m allennlp.run train --help
   usage: run [command] train [-h] -s SERIALIZATION_DIR param_path

   Train the specified model on the specified dataset.

   positional arguments:
   param_path            path to parameter file describing the model to be trained

   optional arguments:
    -h, --help            show this help message and exit
    -s SERIALIZATION_DIR, --serialization-dir SERIALIZATION_DIR
                            directory in which to save the model and its logs
"""
from typing import Dict
import argparse
import json
import logging
import os
import random
import sys
from copy import deepcopy

from allennlp.commands.evaluate import evaluate
from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.tee_logger import TeeLogger
from allennlp.common.util import prepare_environment
from allennlp.data import Dataset, Vocabulary
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.iterators import BasicIterator
from allennlp.models.archival import archive_model
from allennlp.models.model import Model
from allennlp.training.consist_trainer import ConsistTrainer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TrainConsist(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Train the specified model on the specified dataset.'''
        subparser = parser.add_parser(
                name, description=description, help='Train a model')
        subparser.add_argument('param_path',
                               type=str,
                               help='path to parameter file describing the model to be trained')

        # This is necessary to preserve backward compatibility
        serialization = subparser.add_mutually_exclusive_group(required=True)
        serialization.add_argument('-s', '--serialization-dir',
                                   type=str,
                                   help='directory in which to save the model and its logs')
        serialization.add_argument('--serialization_dir',
                                   type=str,
                                   help=argparse.SUPPRESS)

        subparser.set_defaults(func=train_model_from_args)

        return subparser


def train_model_from_args(args: argparse.Namespace):
    """
    Just converts from an ``argparse.Namespace`` object to string paths.
    """
    train_model_from_file(args.param_path, args.serialization_dir)


def train_model_from_file(parameter_filename: str, serialization_dir: str) -> Model:
    """
    A wrapper around :func:`train_model` which loads the params from a file.

    Parameters
    ----------
    param_path: str, required.
        A json parameter file specifying an AllenNLP experiment.
    serialization_dir: str, required
        The directory in which to save results and logs.
    """
    # Load the experiment config from a file and pass it to ``train_model``.
    params = Params.from_file(parameter_filename)
    return train_model(params, serialization_dir)


def train_model(params: Params, serialization_dir: str) -> Model:
    """
    This function can be used as an entry point to running models in AllenNLP
    directly from a JSON specification using a :class:`Driver`. Note that if
    you care about reproducibility, you should avoid running code using Pytorch
    or numpy which affect the reproducibility of your experiment before you
    import and use this function, these libraries rely on random seeds which
    can be set in this function via a JSON specification file. Note that this
    function performs training and will also evaluate the trained model on
    development and test sets if provided in the parameter json.

    Parameters
    ----------
    params: Params, required.
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir: str, required
        The directory in which to save results and logs.
    """
    prepare_environment(params)

    os.makedirs(serialization_dir, exist_ok=True)
    sys.stdout = TeeLogger(os.path.join(serialization_dir, "stdout.log"), sys.stdout)  # type: ignore
    sys.stderr = TeeLogger(os.path.join(serialization_dir, "stderr.log"), sys.stderr)  # type: ignore
    handler = logging.FileHandler(os.path.join(serialization_dir, "python_logging.log"))
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logging.getLogger().addHandler(handler)
    serialization_params = deepcopy(params).as_dict(quiet=True)
    with open(os.path.join(serialization_dir, "model_params.json"), "w") as param_file:
        json.dump(serialization_params, param_file, indent=4)

    # Now we begin assembling the required parts for the Trainer.

    # 1. get train dataset reader
    srl_train_dataset_reader = DatasetReader.from_params(params.pop('srl_dataset_reader'))
    constit_train_dataset_reader = DatasetReader.from_params(params.pop('constit_dataset_reader'))

    # 2. get test dataset reader 
    srl_test_dataset_reader = DatasetReader.from_params(params.pop('srl_test_dataset_reader'))
    constit_test_dataset_reader = DatasetReader.from_params(params.pop('constit_test_dataset_reader'))

    # 3. srl train data
    srl_train_data_path = params.pop('srl_train_data_path')
    logger.info("Reading srl training data from %s", srl_train_data_path)
    srl_train_data = srl_train_dataset_reader.read(srl_train_data_path)
    all_srl_datasets: Dict[str, Dataset] = {"srl_train": srl_train_data}

    # 4. constit train data
    constit_train_data_path = params.pop('constit_train_data_path', None)
    if constit_train_data_path is not None:
        logger.info("Reading constit training data from %s", constit_train_data_path)
        constit_train_data = constit_train_dataset_reader.read(constit_train_data_path)
        all_constit_datasets: Dict[str, Dataset] = {"constit_train": constit_train_data}

        constit_train_size = len(constit_train_data.instances)
        if constit_train_size < len(srl_train_data.instances):
            difference = len(srl_train_data.instances) - constit_train_size
            aux_sample = [random.choice(constit_train_data.instances) for _ in range(difference)]
            constit_train_data = Dataset(constit_train_data.instances + aux_sample)
            logger.info("Inflating constit train data from %d to %d samples",
                        constit_train_size, len(constit_train_data.instances))
    else:
        logger.info("No constit training data provided")
        constit_train_data = None
        all_constit_datasets: Dict[str, Dataset] = {}


    # 5. srl Validation data
    srl_validation_data_path = params.pop('srl_validation_data_path', None)
    if srl_validation_data_path is not None:
        logger.info("Reading validation data from %s", srl_validation_data_path)
        srl_validation_data = srl_test_dataset_reader.read(srl_validation_data_path)
        all_srl_datasets["srl_validation"] = srl_validation_data
    else:
        srl_validation_data = None
    # # 6. constit validation data
    # constit_validation_data_path = params.pop('constit_validation_data_path', None)
    # if constit_validation_data_path is not None:
    #     logger.info("Reading validation data from %s", constit_validation_data_path)
    #     constit_validation_data = constit_test_dataset_reader.read(constit_validation_data_path)
    #     all_constit_datasets["constit_validation"] = constit_validation_data
    # else:
    #     constit_validation_data = None

    # 7. srl Test data
    srl_test_data_path = params.pop("srl_test_data_path", None)
    if srl_test_data_path is not None:
        logger.info("Reading test data from %s", srl_test_data_path)
        srl_test_data = srl_test_dataset_reader.read(srl_test_data_path)
        all_srl_datasets["srl_test"] = srl_test_data
    else:
        srl_test_data = None

    # # 8. constit Test data
    # constit_test_data_path = params.pop("constit_test_data_path", None)
    # if constit_test_data_path is not None:
    #     logger.info("Reading test data from %s", constit_test_data_path)
    #     constit_test_data = constit_test_dataset_reader.read(constit_test_data_path)
    #     all_constit_datasets["constit_test"] = constit_test_data
    # else:
    #     constit_test_data = None

    # Create vocab using the primary and auxiliary datasets.
    srl_datasets_for_vocab_creation = set(params.pop("srl_datasets_for_vocab_creation",
                                                 all_srl_datasets))
    constit_datasets_for_vocab_creation = set(params.pop("constit_datasets_for_vocab_creation",
                                                 all_constit_datasets))

    for dataset in srl_datasets_for_vocab_creation:
        if dataset not in all_srl_datasets:
            raise ConfigurationError(f"invalid 'dataset_for_vocab_creation' {dataset}")

    logger.info("Creating a vocabulary using %s data.", ", ".join(srl_datasets_for_vocab_creation))
    logger.info("Creating a vocabulary using %s data.", ", ".join(constit_datasets_for_vocab_creation))
    srl_vocab_dataset = Dataset([instance for key, dataset in all_srl_datasets.items()
                                     for instance in dataset.instances
                                     if key in srl_datasets_for_vocab_creation])
    constit_vocab_dataset = Dataset([instance for key, dataset in all_constit_datasets.items()
                           for instance in dataset.instances
                           if key in constit_datasets_for_vocab_creation])
    vocab = Vocabulary.from_params(params.pop("vocabulary", {}),
                                   dataset=srl_vocab_dataset,
                                   aux_dataset=constit_vocab_dataset)

    vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))

    # for k, v in vocab._token_to_index.items():
    #     print(k,len(v))

    model = Model.from_params(vocab, params.pop('model'))
    srl_iterator = DataIterator.from_params(params.pop("srl_iterator"))
    constit_iterator = DataIterator.from_params(params.pop("constit_iterator"))

    srl_train_data.index_instances(vocab)
    if constit_train_data is not None:
        constit_train_data.index_instances(vocab)

    if srl_validation_data:
        # srl_validation_data.truncate(1000)
        srl_validation_data.index_instances(vocab)
    # if constit_validation_data:
    #     constit_validation_data.index_instances(vocab)

    trainer_params = params.pop("trainer")
    trainer = ConsistTrainer.from_params(model=model,
                                          serialization_dir=serialization_dir,
                                          iterator=srl_iterator,
                                          aux_iterator=constit_iterator,
                                          train_dataset=srl_train_data,
                                          aux_train_dataset=constit_train_data,
                                          validation_dataset=srl_validation_data,
                                          params=trainer_params,
                                          files_to_archive=params.files_to_archive)

    evaluate_on_test = params.pop("evaluate_on_test", True)
    # params.assert_empty('base train command')
    trainer.train()

    # # Now tar up results
    archive_model(serialization_dir, files_to_archive=params.files_to_archive)

    if srl_test_data and evaluate_on_test:
        srl_test_data.index_instances(vocab)
        test_iterator = BasicIterator(batch_size=32)
        evaluate(model, srl_test_data, test_iterator, cuda_device=trainer._cuda_device, serialization_directory=serialization_dir)  # pylint: disable=protected-access

    elif srl_test_data:
        logger.info("To evaluate on the test set after training, pass the "
                    "'evaluate_on_test' flag, or use the 'allennlp evaluate' command.")

    return model
