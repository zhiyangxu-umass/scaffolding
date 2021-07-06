"""
A :class:`~allennlp.training.trainer.Trainer` is responsible for training a
:class:`~allennlp.models.model.Model`.

Typically you might create a configuration file specifying the model and
training parameters and then use :mod:`~allennlp.commands.train`
rather than instantiating a ``Trainer`` yourself.
"""

import logging
import os
import shutil
import time
from typing import Dict, Optional, List, Tuple

import torch
import torch.optim.lr_scheduler
from torch.nn.utils.clip_grad import clip_grad_norm
from torch.optim.lr_scheduler import _LRScheduler as PytorchLRScheduler  # pylint: disable=protected-access
import tqdm
from tensorboard import SummaryWriter

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Dataset
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.models.archival import archive_model
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.optimizers import Optimizer
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TensorboardWriter:
    """
    Wraps a pair of ``SummaryWriter`` instances but is a no-op if they're ``None``.
    Allows Tensorboard logging without always checking for Nones first.
    """
    def __init__(self, train_log: SummaryWriter = None, validation_log: SummaryWriter = None) -> None:
        self._train_log = train_log
        self._validation_log = validation_log

    def add_train_scalar(self, name: str, value: float, global_step: int) -> None:
        if self._train_log is not None:
            self._train_log.add_scalar(name, value, global_step)

    def add_validation_scalar(self, name: str, value: float, global_step: int) -> None:
        if self._validation_log is not None:
            self._validation_log.add_scalar(name, value, global_step)


class ConsistInjectionTrainer:
    def __init__(self,
                 model: Model,
                 optimizer: torch.optim.Optimizer,
                 iterator: DataIterator,
                 train_dataset: Dataset,
                 aux_iterator: DataIterator,
                 aux_train_dataset: Dataset,
                 mixing_ratio: float = 1.0,
                 warmup_epoch: int = -1,
                 validation_dataset: Optional[Dataset] = None,
                 patience: int = 2,
                 validation_metric: str = "-loss",
                 num_epochs: int = 20,
                 serialization_dir: Optional[str] = None,
                 files_to_archive: Dict[str, str] = None,
                 cuda_device: int = -1,
                 grad_norm: Optional[float] = None,
                 grad_clipping: Optional[float] = None,
                 learning_rate_scheduler: Optional[PytorchLRScheduler] = None,
                 no_tqdm: bool = False) -> None:
        """
        Parameters
        ----------
        model : ``Model``, required.
            An AllenNLP model to be optimized. Pytorch Modules can also be optimized if
            their ``forward`` method returns a dictionary with a "loss" key, containing a
            scalar tensor representing the loss function to be optimized.
        optimizer : ``torch.nn.Optimizer``, required.
            An instance of a Pytorch Optimizer, instantiated with the parameters of the
            model to be optimized.
        iterator : ``DataIterator``, required.
            A method for iterating over a ``Dataset``, yielding padded indexed batches.
        train_dataset : ``Dataset``, required.
            A ``Dataset`` to train on. The dataset should have already been indexed.
        validation_dataset : ``Dataset``, optional, (default = None).
            A ``Dataset`` to evaluate on. The dataset should have already been indexed.
        patience : int, optional (default=2)
            Number of epochs to be patient before early stopping.
        validation_metric : str, optional (default="loss")
            Validation metric to measure for whether to stop training using patience
            and whether to serialize an ``is_best`` model each epoch. The metric name
            must be prepended with either "+" or "-", which specifies whether the metric
            is an increasing or decreasing function.
        num_epochs : int, optional (default = 20)
            Number of training epochs.
        serialization_dir : str, optional (default=None)
            Path to directory for saving and loading model files. Models will not be saved if
            this parameter is not passed.
        cuda_device : int, optional (default = -1)
            An integer specifying the CUDA device to use. If -1, the CPU is used.
            Multi-gpu training is not currently supported, but will be once the
            Pytorch DataParallel API stabilises.
        grad_norm : float, optional, (default = None).
            If provided, gradient norms will be rescaled to have a maximum of this value.
        grad_clipping : ``float``, optional (default = ``None``).
            If provided, gradients will be clipped `during the backward pass` to have an (absolute)
            maximum of this value.  If you are getting ``NaNs`` in your gradients during training
            that are not solved by using ``grad_norm``, you may need this.
        learning_rate_scheduler : PytorchLRScheduler, optional, (default = None)
            A Pytorch learning rate scheduler. The learning rate will be decayed with respect to
            this schedule at the end of each epoch. If you use
            :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`, this will use the ``validation_metric``
            provided to determine if learning has plateaued.
        no_tqdm : ``bool``, optional (default=False)
            We use ``tqdm`` for logging, which will print a nice progress bar that updates in place
            after every batch.  This is nice if you're running training on a local shell, but can
            cause problems with log files from, e.g., a docker image running on kubernetes.  If
            ``no_tqdm`` is ``True``, we will not use tqdm, and instead log batch statistics using
            ``logger.info``, outputting a line at most every 10 seconds.
        """
        self._model = model
        self._iterator = iterator
        self._optimizer = optimizer
        self._train_dataset = train_dataset
        self._validation_dataset = validation_dataset

        self._aux_iterator = aux_iterator
        self._aux_train_dataset = aux_train_dataset
        self._mixing_ratio = mixing_ratio
        self._warmup_epoch = warmup_epoch

        self._patience = patience
        self._num_epochs = num_epochs
        self._serialization_dir = serialization_dir
        self._files_to_archive = files_to_archive
        self._cuda_device = cuda_device
        self._grad_norm = grad_norm
        self._grad_clipping = grad_clipping
        self._learning_rate_scheduler = learning_rate_scheduler

        increase_or_decrease = validation_metric[0]
        if increase_or_decrease not in ["+", "-"]:
            raise ConfigurationError("Validation metrics must specify whether they should increase "
                                     "or decrease by pre-pending the metric name with a +/-.")
        self._validation_metric = validation_metric[1:]
        self._validation_metric_decreases = increase_or_decrease == "-"
        self._no_tqdm = no_tqdm

        if self._cuda_device >= 0:
            self._model = self._model.cuda(self._cuda_device)
            torch.cuda.set_device(self._cuda_device)

        self._log_interval = 10  # seconds
        self._summary_interval = 100  # num batches between logging to tensorboard

        self._last_log = 0.0  # time of last logging

        if serialization_dir is not None:
            train_log = SummaryWriter(os.path.join(serialization_dir, "log", "train"))
            validation_log = SummaryWriter(os.path.join(serialization_dir, "log", "validation"))
            self._tensorboard = TensorboardWriter(train_log, validation_log)
        else:
            self._tensorboard = TensorboardWriter()

    def _enable_gradient_clipping(self) -> None:
        if self._grad_clipping is not None:
            # Pylint is unable to tell that we're in the case that _grad_clipping is not None...
            # pylint: disable=invalid-unary-operand-type
            clip_function = lambda grad: grad.clamp(-self._grad_clipping, self._grad_clipping)
            for parameter in self._model.parameters():
                if parameter.requires_grad:
                    parameter.register_hook(clip_function)

    def _rescale_gradients(self) -> None:
        """
        Performs gradient rescaling. Is a no-op if gradient rescaling is not enabled.
        """
        if self._grad_norm:
            clip_grad_norm(self._model.parameters(), self._grad_norm)

    def _batch_loss(self, batch: torch.Tensor = None, for_training: bool = True, aux_batch: torch.Tensor = None) -> torch.Tensor:
        """
        Does a forward pass on the given batch and returns the ``loss`` value in the result.
        If ``for_training`` is `True` also applies regularization penalty.
        """
        total_loss = 0.0
        if batch is not None:
            srl_output_dict = self._model(**batch,srl_batch=True)

            try:
                srl_loss = srl_output_dict["loss"]
                if for_training:
                    srl_loss += self._model.get_regularization_penalty()
            except KeyError:
                raise ConfigurationError("The model you are trying to optimize does not contain a"
                                        " 'loss' key in the output of model.forward(inputs).")
            total_loss += srl_loss    
        if aux_batch is not None:
            constit_output_dict = self._model(**aux_batch,srl_batch=False)
            try:
                constit_loss = constit_output_dict["loss"]
                if for_training:
                    constit_loss += self._model.get_regularization_penalty()
            except KeyError:
                raise ConfigurationError("No `loss` key in output_dict for auxiliary batch.")
            total_loss += self._mixing_ratio * constit_loss

        return total_loss

    def _get_metrics(self, total_loss: float, batch_num: int, reset: bool = False) -> dict:
        """
        Gets the metrics but sets ``"loss"`` to
        the total loss divided by the ``batch_num`` so that
        the ``"loss"`` metric is "average loss per batch".
        """
        metrics = self._model.get_metrics(reset=reset)
        metrics["loss"] = float(total_loss / batch_num)
        return metrics

    def _train_epoch(self, epoch: int) -> dict:
        """
        Trains one epoch and returns metrics.
        """
        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        train_loss = 0.0
        # Set the model to "train" mode.
        self._model.train()

        # Get tqdm for the training batches
        train_generator = self._iterator(self._train_dataset,
                                         num_epochs=1,
                                         cuda_device=self._cuda_device)
        num_training_batches = self._iterator.get_num_batches(self._train_dataset)
        train_generator_tqdm = tqdm.tqdm(train_generator,
                                         disable=self._no_tqdm,
                                         total=num_training_batches)
        if self._aux_train_dataset is not None:
            aux_train_generator = self._aux_iterator(self._aux_train_dataset,
                                                    num_epochs=1,
                                                    cuda_device=self._cuda_device)

        self._last_log = time.time()
        batch_num = 0

        consist_training = False
        if epoch > self._warmup_epoch:
            consist_training = True
            logger.info("consist Training")
        else:
            logger.info("Training")
        if self._aux_train_dataset is not None:
            for batch, aux_batch in zip(train_generator_tqdm, aux_train_generator):
                batch_num += 1
                self._optimizer.zero_grad()

                if consist_training:
                    self._model.constit2srl_consist = True
                    loss = self._batch_loss(aux_batch=aux_batch, for_training=True)
                    loss.backward()
                    self._rescale_gradients()
                    self._optimizer.step()
                    self._model.constit2srl_consist = False
                    loss = self._batch_loss(batch=batch, for_training=True, aux_batch=aux_batch)
                    # loss.backward()
                    # self._rescale_gradients()
                    # self._optimizer.step()
                    # loss = self._batch_loss(batch=batch, for_training=True)

                else:
                    self._model.constit2srl_consist = False
                    loss = self._batch_loss(batch=batch, for_training=True, aux_batch=aux_batch)

                loss.backward()

                # Make sure Variable is on the cpu before converting to numpy.
                # .cpu() is a no-op if you aren't using GPUs.
                train_loss += loss.data.cpu().numpy()

                self._rescale_gradients()

                self._optimizer.step()

                # Update the description with the latest metrics
                metrics = self._get_metrics(train_loss, batch_num)
                description = self._description_from_metrics(metrics)
                train_generator_tqdm.set_description(description)

                # Log parameter values to Tensorboard
                batch_num_total = num_training_batches * epoch + batch_num
                if batch_num_total % self._summary_interval == 0:
                    for name, param in self._model.named_parameters():
                        self._tensorboard.add_train_scalar("parameter_mean/" + name,
                                                        param.data.mean(),
                                                        batch_num_total)
                        self._tensorboard.add_train_scalar("parameter_std/" + name, param.data.std(), batch_num_total)
                        if param.grad is not None:
                            self._tensorboard.add_train_scalar("gradient_mean/" + name,
                                                            param.grad.data.mean(),
                                                            batch_num_total)
                            self._tensorboard.add_train_scalar("gradient_std/" + name,
                                                            param.grad.data.std(),
                                                            batch_num_total)
                    self._tensorboard.add_train_scalar("loss/loss_train", metrics["loss"], batch_num_total)
                    self._metrics_to_tensorboard(batch_num_total,
                                                {"epoch_metrics/" + k: v for k, v in metrics.items()})

                # Log progress in no-tqdm case
                if self._no_tqdm and time.time() - self._last_log > self._log_interval:
                    logger.info("Batch %d/%d: %s", batch_num, num_training_batches, description)
                    self._last_log = time.time()
        else:
            for batch in train_generator_tqdm:
                batch_num += 1
                self._optimizer.zero_grad()

                loss = self._batch_loss(batch, for_training=True)
                loss.backward()

                # Make sure Variable is on the cpu before converting to numpy.
                # .cpu() is a no-op if you aren't using GPUs.
                train_loss += loss.data.cpu().numpy()

                self._rescale_gradients()

                self._optimizer.step()

                # Update the description with the latest metrics
                metrics = self._get_metrics(train_loss, batch_num)
                description = self._description_from_metrics(metrics)
                train_generator_tqdm.set_description(description)

                # Log parameter values to Tensorboard
                batch_num_total = num_training_batches * epoch + batch_num
                if batch_num_total % self._summary_interval == 0:
                    for name, param in self._model.named_parameters():
                        self._tensorboard.add_train_scalar("parameter_mean/" + name,
                                                        param.data.mean(),
                                                        batch_num_total)
                        self._tensorboard.add_train_scalar("parameter_std/" + name, param.data.std(), batch_num_total)
                        if param.grad is not None:
                            self._tensorboard.add_train_scalar("gradient_mean/" + name,
                                                            param.grad.data.mean(),
                                                            batch_num_total)
                            self._tensorboard.add_train_scalar("gradient_std/" + name,
                                                            param.grad.data.std(),
                                                            batch_num_total)
                    self._tensorboard.add_train_scalar("loss/loss_train", metrics["loss"], batch_num_total)
                    self._metrics_to_tensorboard(batch_num_total,
                                                {"epoch_metrics/" + k: v for k, v in metrics.items()})

                # Log progress in no-tqdm case
                if self._no_tqdm and time.time() - self._last_log > self._log_interval:
                    logger.info("Batch %d/%d: %s", batch_num, num_training_batches, description)
                    self._last_log = time.time()

        return self._get_metrics(train_loss, batch_num, reset=True)

    def _should_stop_early(self, metric_history: List[float]) -> bool:
        """
        uses patience and the validation metric to determine if training should stop early
        """
        if len(metric_history) > self._patience:
            # Is the best score in the past N epochs worse than the best score overall?
            if self._validation_metric_decreases:
                return min(metric_history[-self._patience:]) > min(metric_history)
            else:
                return max(metric_history[-self._patience:]) < max(metric_history)

        return False

    def _metrics_to_tensorboard(self,
                                epoch: int,
                                train_metrics: dict,
                                val_metrics: dict = None) -> None:
        """
        Sends all of the train metrics (and validation metrics, if provided) to tensorboard.
        """
        for name, value in train_metrics.items():
            # print(name)
            if "overall" in name or "loss" in name:
                self._tensorboard.add_train_scalar(name, value, epoch)
                if val_metrics:
                    self._tensorboard.add_validation_scalar(name, val_metrics[name], epoch)

    def _metrics_to_console(self,  # pylint: disable=no-self-use
                            train_metrics: dict,
                            val_metrics: dict = None) -> None:
        """
        Logs all of the train metrics (and validation metrics, if provided) to the console.
        """
        if val_metrics:
            message_template = "Training %s : %3f    Validation %s : %3f "
        else:
            message_template = "Training %s : %3f "

        for name, value in train_metrics.items():
            if "overall" not in name and "loss" not in name:
                continue
            if val_metrics:
                logger.info(message_template, name, value, name, val_metrics[name])
            else:
                logger.info(message_template, name, value)

    def _update_learning_rate(self, epoch: int, val_metric: float = None) -> None:
        if not self._learning_rate_scheduler:
            return

        # Grim hack to determine whether the validation metric we are recording
        # needs to be passed to the scheduler. This is required because the
        # step() function of the different schedulers are (understandably)
        # different to ReduceLROnPlateau.
        reduce_on_plateau = isinstance(self._learning_rate_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

        if reduce_on_plateau and val_metric is None:
            raise ConfigurationError("The reduce_on_plateau learning rate scheduler requires "
                                     "a validation metric to compute the schedule and therefore "
                                     "must be used with a validation dataset.")
        elif reduce_on_plateau:
            self._learning_rate_scheduler.step(val_metric, epoch)
        else:
            self._learning_rate_scheduler.step(epoch)

    def _validation_loss(self) -> Tuple[float, int]:
        """
        Computes the validation loss. Returns it and the number of batches.
        """
        logger.info("Validating")

        self._model.eval()

        val_generator = self._iterator(self._validation_dataset,
                                       num_epochs=1,
                                       cuda_device=self._cuda_device,
                                       for_training=False)
        num_validation_batches = self._iterator.get_num_batches(self._validation_dataset)
        val_generator_tqdm = tqdm.tqdm(val_generator,
                                       disable=self._no_tqdm,
                                       total=num_validation_batches)
        batch_num = 0
        val_loss = 0
        for batch in val_generator_tqdm:
            batch_num += 1

            loss = self._batch_loss(batch, for_training=False)
            val_loss += loss.data.cpu().numpy()

            # Update the description with the latest metrics
            val_metrics = self._get_metrics(val_loss, batch_num)
            description = self._description_from_metrics(val_metrics)
            val_generator_tqdm.set_description(description)

            # Log progress in the no-tqdm case
            if self._no_tqdm and time.time() - self._last_log > self._log_interval:
                logger.info("Batch %d/%d: %s", batch_num, num_validation_batches, description)
                self._last_log = time.time()

        return val_loss, batch_num

    def train(self) -> None:
        """
        Trains the supplied model with the supplied parameters.
        """
        epoch_counter, validation_metric_per_epoch = self._restore_checkpoint()
        self._enable_gradient_clipping()

        logger.info("Beginning training.")

        training_start_time = time.time()
        for epoch in range(epoch_counter, self._num_epochs):
            epoch_start_time = time.time()
            train_metrics = self._train_epoch(epoch)

            if self._validation_dataset is not None:
                # We have a validation set, so compute all the metrics on it.
                val_loss, num_batches = self._validation_loss()
                val_metrics = self._get_metrics(val_loss, num_batches, reset=True)

                # Check validation metric for early stopping
                this_epoch_val_metric = val_metrics[self._validation_metric]
                validation_metric_per_epoch.append(this_epoch_val_metric)
                if self._should_stop_early(validation_metric_per_epoch):
                    logger.info("Ran out of patience.  Stopping training.")
                    break

                # Check validation metric to see if it's the best so far
                if self._validation_metric_decreases:
                    is_best_so_far = this_epoch_val_metric == min(validation_metric_per_epoch)
                else:
                    is_best_so_far = this_epoch_val_metric == max(validation_metric_per_epoch)
            else:
                # No validation set, so just assume it's the best so far.
                is_best_so_far = True
                val_metrics = this_epoch_val_metric = None

            if epoch % 10 == 0 or is_best_so_far:
                self._save_checkpoint(epoch, validation_metric_per_epoch, is_best=is_best_so_far)
            self._metrics_to_tensorboard(epoch, train_metrics, val_metrics=val_metrics)
            self._metrics_to_console(train_metrics, val_metrics)
            self._update_learning_rate(epoch, val_metric=this_epoch_val_metric)

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", time.strftime("%H:%M:%S", time.gmtime(epoch_elapsed_time)))

            if epoch < self._num_epochs - 1:
                training_elapsed_time = time.time() - training_start_time
                estimated_time_remaining = training_elapsed_time * \
                    ((self._num_epochs - epoch_counter) / float(epoch - epoch_counter + 1) - 1)
                formatted_time = time.strftime("%H:%M:%S", time.gmtime(estimated_time_remaining))
                logger.info("Estimated training time remaining: %s", formatted_time)

    def _description_from_metrics(self, metrics: Dict[str, float]) -> str:
        # pylint: disable=no-self-use
        return ', '.join(["%s: %.3f" % (name, value) for name, value in metrics.items() if "overall" in name or "loss" in name]) + " ||"

    def _save_checkpoint(self,
                         epoch: int,
                         val_metric_per_epoch: List[float],
                         is_best: Optional[bool] = None) -> None:
        """
        Saves a checkpoint of the model to self._serialization_dir.
        Is a no-op if self._serialization_dir is None.

        Parameters
        ----------
        epoch : int, required.
            The epoch of training.
        is_best: bool, optional (default = None)
            A flag which causes the model weights at the given epoch to
            be copied to a "best.th" file. The value of this flag should
            be based on some validation metric computed by your model.
        """
        if self._serialization_dir is not None:
            model_path = os.path.join(self._serialization_dir, "model_state_epoch_{}.th".format(epoch))
            model_state = self._model.state_dict()
            torch.save(model_state, model_path)

            training_state = {'epoch': epoch,
                              'val_metric_per_epoch': val_metric_per_epoch,
                              'optimizer': self._optimizer.state_dict()}
            torch.save(training_state, os.path.join(self._serialization_dir,
                                                    "training_state_epoch_{}.th".format(epoch)))
            if is_best:
                logger.info("Best validation performance so far. "
                            "Copying weights to '%s/best.th'.", self._serialization_dir)
                shutil.copyfile(model_path, os.path.join(self._serialization_dir, "best.th"))
                archive_model(self._serialization_dir, files_to_archive=self._files_to_archive)

    def _restore_checkpoint(self) -> Tuple[int, List[float]]:
        """
        Restores a model from a serialization_dir to the last saved checkpoint.
        This includes an epoch count and optimizer state, which is serialized separately
        from  model parameters. This function should only be used to continue training -
        if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        `` model.load_state_dict(torch.load("/path/to/model/weights.th"))``

        If ``self._serialization_dir`` does not exist or does not contain any checkpointed weights,
        this function will do nothing and return 0.

        Returns
        -------
        epoch: int
            The epoch at which to resume training, which should be one after the epoch
            in the saved training state.
        """
        have_checkpoint = (self._serialization_dir is not None and
                           any("model_state_epoch_" in x for x in os.listdir(self._serialization_dir)))

        if not have_checkpoint:
            # No checkpoint to restore, start at 0
            return 0, []

        serialization_files = os.listdir(self._serialization_dir)
        model_checkpoints = [x for x in serialization_files if "model_state_epoch" in x]
        epoch_to_load = max([int(x.split("model_state_epoch_")[-1].strip(".th")) for x in model_checkpoints])

        model_path = os.path.join(self._serialization_dir,
                                  "model_state_epoch_{}.th".format(epoch_to_load))
        training_state_path = os.path.join(self._serialization_dir,
                                           "training_state_epoch_{}.th".format(epoch_to_load))

        model_state = torch.load(model_path, map_location=util.device_mapping(self._cuda_device))
        training_state = torch.load(training_state_path, map_location=util.device_mapping(self._cuda_device))
        self._model.load_state_dict(model_state)
        self._optimizer.load_state_dict(training_state["optimizer"])

        # We didn't used to save `validation_metric_per_epoch`, so we can't assume
        # that it's part of the trainer state. If it's not there, an empty list is all
        # we can do.
        if "val_metric_per_epoch" not in training_state:
            logger.warning("trainer state `val_metric_per_epoch` not found, using empty list")
            val_metric_per_epoch: List[float] = []
        else:
            val_metric_per_epoch = training_state["val_metric_per_epoch"]

        return training_state["epoch"] + 1, val_metric_per_epoch

    @classmethod
    def from_params(cls,
                    model: Model,
                    serialization_dir: str,
                    iterator: DataIterator,
                    aux_iterator: DataIterator,
                    train_dataset: Dataset,
                    aux_train_dataset: Dataset,
                    validation_dataset: Optional[Dataset],
                    params: Params,
                    files_to_archive: Dict[str, str]) -> 'ConsistInjectionTrainer':

        patience = params.pop("patience", 2)
        validation_metric = params.pop("validation_metric", "-loss")
        num_epochs = params.pop("num_epochs", 20)
        cuda_device = params.pop("cuda_device", -1)
        grad_norm = params.pop("grad_norm", None)
        grad_clipping = params.pop("grad_clipping", None)
        lr_scheduler_params = params.pop("learning_rate_scheduler", None)

        if cuda_device >= 0:
            model = model.cuda(cuda_device)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(parameters, params.pop("optimizer"))

        if lr_scheduler_params:
            scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params)
        else:
            scheduler = None
        no_tqdm = params.pop("no_tqdm", False)

        mixing_ratio = params.pop("mixing_ratio", 1.0)
        warmup_epoch = params.pop("warmup_epoch", -1)

        params.assert_empty(cls.__name__)
        return ConsistInjectionTrainer(model=model,
                               optimizer=optimizer,
                               iterator=iterator,
                               train_dataset=train_dataset,
                               aux_iterator=aux_iterator,
                               aux_train_dataset=aux_train_dataset,
                               mixing_ratio=mixing_ratio,
                               warmup_epoch=warmup_epoch,
                               validation_dataset=validation_dataset,
                               patience=patience,
                               validation_metric=validation_metric,
                               num_epochs=num_epochs,
                               serialization_dir=serialization_dir,
                               files_to_archive=files_to_archive,
                               cuda_device=cuda_device,
                               grad_norm=grad_norm,
                               grad_clipping=grad_clipping,
                               learning_rate_scheduler=scheduler,
                               no_tqdm=no_tqdm)