import os
import sys
import numpy as np
import logging
from ctcdecode import CTCBeamDecoder
import jiwer
import tqdm


import torch
from torch import nn
import torch.nn.functional as F

from read_emg import EMGDataset, SizeAwareSampler
from architecture import Model
from data_utils import combine_fixed_length, decollate_tensor

from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_boolean("debug", False, "debug")
flags.DEFINE_string("output_directory", "output", "where to save models and outputs")
flags.DEFINE_float("learning_rate", 3e-4, "learning rate")
flags.DEFINE_integer("learning_rate_warmup", 1000, "steps of linear warmup")
flags.DEFINE_integer("learning_rate_patience", 5, "learning rate decay patience")
flags.DEFINE_boolean(
    "start_training_from_model", False, "start training from this model"
)
flags.DEFINE_float("l2", 0, "weight decay")
flags.DEFINE_string("evaluate_saved", None, "run evaluation on given model file")


def test(model, testset, device):
    model.eval()

    blank_id = len(testset.text_transform.chars)
    decoder = CTCBeamDecoder(
        testset.text_transform.chars + "_",
        blank_id=blank_id,
        log_probs_input=True,
        model_path="lm.binary",
        alpha=1.5,
        beta=1.85,
    )

    dataloader = torch.utils.data.DataLoader(testset, batch_size=1)
    references = []
    predictions = []
    with torch.no_grad():
        for example in tqdm.tqdm(dataloader, "Evaluate", disable=None):
            X_raw = example["raw_emg"].to(device)

            pred = F.log_softmax(model(X_raw), -1)

            beam_results, _, _, out_lens = decoder.decode(pred)
            pred_int = beam_results[0, 0, : out_lens[0, 0]].tolist()

            pred_text = testset.text_transform.int_to_text(pred_int)
            target_text = testset.text_transform.clean_text(example["text"][0])

            references.append(target_text)
            predictions.append(pred_text)

    model.train()

    for i in range(len(references)):
        print(references[i], " : ", predictions[i], "\n\n")

    return jiwer.wer(references, predictions)


def train_model(trainset, devset, device, n_epochs=200):
    # INFO:
    dataloader = torch.utils.data.DataLoader(
        trainset,
        num_workers=0,
        collate_fn=EMGDataset.collate_raw,
        batch_sampler=SizeAwareSampler(trainset, 128000),
    )

    n_chars = len(devset.text_transform.chars)
    model = Model(n_chars + 1).to(device)

    if FLAGS.start_training_from_model:
        model.load_state_dict(
            torch.load(FLAGS.output_directory, map_location=torch.device(device))
        )

    # INFO: configuring optimizer
    optim = torch.optim.AdamW(
        model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.l2
    )
    # INFO: configuring learning rate schedule
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(
        optim, milestones=[125, 150, 175], gamma=0.5
    )

    target_lr = FLAGS.learning_rate

    def schedule_lr(iteration):
        """change lerning rate"""

        iteration = iteration + 1
        if iteration <= FLAGS.learning_rate_warmup:
            for param_group in optim.param_groups:
                param_group["lr"] = iteration * target_lr / FLAGS.learning_rate_warmup

    batch_idx = 0
    # INFO: zero the gradient explicitly at each iteration
    optim.zero_grad()

    # INFO: learning cycle
    for epoch_idx in range(n_epochs):
        losses = []
        # INFO: display progress for learning
        for example in tqdm.tqdm(dataloader, "Train step", disable=None):

            # INFO: set learning rate for curent step
            schedule_lr(batch_idx)

            # INFO: load data to memory
            X_raw = combine_fixed_length(example["raw_emg"], 200 * 8).to(device)

            pred = model(X_raw)
            pred = F.log_softmax(pred, 2)

            # INFO: seq first, as required by ctc
            pred = nn.utils.rnn.pad_sequence(
                decollate_tensor(pred, example["lengths"]), batch_first=False
            )
            y = nn.utils.rnn.pad_sequence(example["text_int"], batch_first=True).to(
                device
            )
            loss = F.ctc_loss(
                pred, y, example["lengths"], example["text_int_lengths"], blank=n_chars
            )
            losses.append(loss.item())

            loss.backward()
            if (batch_idx + 1) % 2 == 0:
                optim.step()
                optim.zero_grad()

            batch_idx += 1
        train_loss = np.mean(losses)
        val = test(model, devset, device)
        lr_sched.step()
        logging.info(
            f"finished epoch {epoch_idx+1} - training loss: {train_loss:.4f} validation WER: {val*100:.2f}"
        )
        torch.save(model.state_dict(), os.path.join(FLAGS.output_directory, "model.pt"))

    model.load_state_dict(
        torch.load(
            os.path.join(FLAGS.output_directory, "model.pt"),
            map_location=torch.device(device),
        )
    )  # re-load best parameters
    return model


def evaluate_saved():
    device = "cuda" if torch.cuda.is_available() and not FLAGS.debug else "cpu"
    testset = EMGDataset(test=True)
    n_chars = len(testset.text_transform.chars)
    model = Model(n_chars + 1).to(device)
    model.load_state_dict(
        torch.load(FLAGS.evaluate_saved, map_location=torch.device(device))
    )
    print("WER:", test(model, testset, device))


def main():
    # INFO: make output directory
    os.makedirs(FLAGS.output_directory, exist_ok=True)
    # INFO: configuring logging
    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join(FLAGS.output_directory, "log.txt"), "w"),
            logging.StreamHandler(),
        ],
        level=logging.INFO,
        format="%(message)s",
    )
    logging.info(sys.argv)
    # INFO: load data
    trainset = EMGDataset(dev=False, test=False)
    devset = EMGDataset(dev=True)
    logging.info("output example: %s", devset.example_indices[0])
    logging.info("train / dev split: %d %d", len(trainset), len(devset))
    # INFO: configuring device for learning
    device = "cuda" if torch.cuda.is_available() and not FLAGS.debug else "cpu"
    # INFO: learning process
    model = train_model(trainset, devset, device)


if __name__ == "__main__":
    FLAGS(sys.argv)
    if FLAGS.evaluate_saved is not None:
        evaluate_saved()
    else:
        main()
