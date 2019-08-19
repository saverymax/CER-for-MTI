#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Copyright 2018 The Google AI Language Team Authors.
BASED ON Google_BERT.
@Author:zhoukaiyin
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import copy
import collections
import os
import pickle
import time
import glob

from bert import modeling
from bert import optimization
from bert import tokenization
import tensorflow as tf
from tensorflow.python.ops import math_ops
import tf_metrics

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir", None,
    "The input datadir.",
)

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string(
    "task_name", None, "The name of the task to train."
)

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written."
)

flags.DEFINE_string(
    "output_preds", None,
    "The file to where the predictions will be written"
)

flags.DEFINE_string(
    "training_file", None,
    "The file name of the training data"
)
## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_string(
    "citation_dir", None,
    "The dataset to load to make predictions on")


flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length", 256,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_integer(
    "n_labels", 8,
    "The label structure to use: Either multiclass or binary"
)

flags.DEFINE_bool(
    "do_train", False,
    "Whether to run training."
)

flags.DEFINE_bool("use_crf", False, "Flag to include CRF as a final layer instead of softmax")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict", False, "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool("do_CDI", False, "Whether to run the model on the CDI evaluation data.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0, "Total number of training epochs to perform.")



flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids,):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids

        self.wordpiece_tokens = []


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads BIO data."""
        with open(input_file) as f:
            # Lines is a list of lists of lines
            lines = []
            # Words is joined sentence
            words = []
            # Labels are the joined labels
            labels = []
            line_cnt = 0
            for line in f:
                line_cnt += 1
                text = line.strip().split("\t")
                current_doc_id = text[0]
                # pmid tracker
                if line_cnt == 1:
                    previous_doc_id = text[0]
                elif current_doc_id != previous_doc_id:
                    l = ' '.join([label for label in labels])
                    w = ' '.join([word for word in words])
                    lines.append([l, w, previous_doc_id])
                    words = []
                    labels = []
                    previous_doc_id = text[0]

                # And then get the word + label
                word = text[1]
                label = text[-1]
                words.append(word)
                labels.append(label)
            # Add the last document
            l = ' '.join([label for label in labels])
            w = ' '.join([word for word in words])
            lines.append([l, w, current_doc_id])

            return lines

    @classmethod
    def _read_text_data(cls, citation_dir):
        """
        For inference without labels
        Read text files in a directory
        and concatentate all the lines in the
        files together
        """

        citations = []
        for citation in glob.iglob(citation_dir):
            pmid = citation.split("/")[-1].split(".")[0]
            with open(citation, "r", encoding="utf8") as f:
              citation_text = f.read()
              citations.append(["O", citation_text.strip(), pmid])
        print("Citation count:")
        print(len(citations))
        return citations
    
    @classmethod
    def _read_CDI_data(cls, citation_dir):
        """
        For chemdner CDI task
        Read text files in a directory
        and concatentate all the lines in the
        files together
        """

        citations = []
        with open(citation_dir, "r", encoding="utf8") as f:
            for citation in f.readlines():
                pmid = citation.split("\t")[0]
                title = citation.split("\t")[1]
                abstract = citation.split("\t")[2]
                citation_concat = title + " " + abstract
                citations.append(["O", citation_concat, pmid])

        print("Citation count:")
        print(len(citations))
        return citations


class NerProcessor(DataProcessor):
    def get_train_examples(self, data_dir, training_file):
        return self._create_example(
            self._read_data(os.path.join(data_dir, training_file)), "train"
        )

    def get_dev_examples(self, data_dir):
        """Don't currently have a dev implementation set up"""
        return self._create_example(
            self._read_data(os.path.join(data_dir, ".txt")), "dev"
        )

    def get_test_examples(self, citation_dir):
        return self._create_example(
            self._read_text_data(citation_dir), "test")

    def get_CDI_examples(self, citation_dir):
        return self._create_example(
            self._read_CDI_data(citation_dir), "CDI")

    def get_multi_labels(self):
        """
        Not used for final experiment, as individual models are training on individual types
        """

        labels = [
                "B-aapp",
                "I-aapp",
                "B-inch",
                "I-inch",
                "B-orch",
                "I-orch",
                "B-nnon",
                "I-nnon",
                "O",
                "X", 
                "[CLS]", 
                "[SEP]"]

        return labels

    def get_binary_labels(self):
        labels = [
                "B-chem", 
                "I-chem", 
                "O",
                "X",
                "[CLS]",
                "[SEP]"]

        return labels

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = str(line[2])
            label = tokenization.convert_to_unicode(line[0])
            text = tokenization.convert_to_unicode(line[1])
            examples.append(InputExample(guid=guid, text=text, label=label))

        return examples


def write_tokens(tokens,mode, guid):
    if mode=="test": 
        path = os.path.join(FLAGS.output_dir, "results/token_"+mode+".txt")
        wf = open(path,'a')
        for token in tokens:
            if token!="**NULL**":
                wf.write("{0}\t{1}\n".format(guid, token))
        wf.close()


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, label_map, mode):
    """
    This receives a single citation,
    and processes it, generating the embeddings
    bert requires
    """

    # Break up the text into words
    textlist = example.text.split(' ')
    # Break up the labels
    # Don't need this for new data
    if FLAGS.do_train:
        labellist = example.label.split(' ')
    labels = []
    tokens = []
    # Go through the tokens and break into
    # subwords. Add these and adjusted labels
    # to tokens and labels (the [])
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        # Configuring labels for training:
        if FLAGS.do_train:
            label_1 = labellist[i]
            # For the subwords in the tokens,
            # label the first bit with the label_1
            # and then X after
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                else:
                    # Label the word bits with X
                    labels.append("X")
    # If the input is too long after sub word
    # tokenization, break it up
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        if FLAGS.do_train:
            labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    # Segment ids is for next sentence prediction:
    # Indicates if the sentence is from the first or
    # second bit
    segment_ids = []
    label_ids = []
    # Add CLS to the beginning, as bert was trained with this
    # n_tokens is going to be the same as tokens, but
    # just with cls and sep at the beginning and end,
    # as well as ***NULL*** in n_tokens where the length
    # needs to filled up to max_seq_length
    ntokens.append("[CLS]")
    segment_ids.append(0)
    if FLAGS.do_train:
        label_ids.append(label_map["[CLS]"])
    # For each label in the text, add its int mapping
    # to label_ids.
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        if FLAGS.do_train:
            label_ids.append(label_map[labels[i]])
    # Add sep to the end, indicating phrase separator
    # Try not adding sep with CRF
    if not FLAGS.use_crf:
        ntokens.append("[SEP]")
        segment_ids.append(0)
        if FLAGS.do_train:
            label_ids.append(label_map["[SEP]"])
    # using bert method that maps token to id,
    # get the embedding for the token, in the form
    # bert will see it.
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    # Fill up the rest of the empty embedding with 0
    # This is equivalent to keras zero padding
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        ntokens.append("**NULL**")
        if FLAGS.do_train:
            label_ids.append(0)
    
    if ex_index < 5:
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))

    if not FLAGS.do_train:
        # Hack to not have to deal with labels
        # for inference
        label_len = len(segment_ids)
        for i in range(label_len):
            label_ids.append(9)

    assert len(label_ids) == max_seq_length, len(label_ids)
    assert len(segment_ids) == max_seq_length
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
    )
    guid = example.guid
    example.wordpiece_tokens = ntokens
    write_tokens(ntokens, mode, guid)

    return feature


def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, label_map, mode=None
):
    writer = tf.python_io.TFRecordWriter(output_file)
    tf.logging.info("\nTraining data size: {}\n".format(len(examples)))
    for (ex_index, example) in enumerate(examples):
        if ex_index % 50000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        # convert_single_example is where the meat of the data processing is done
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, label_map, mode)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        # Example is tensorflow data format with features
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d
    return input_fn


def crf_loss(logits,labels,mask,num_labels,mask2len):
    """
    :param logits:
    :param labels:
    :param mask2len:each sample's length
    :return:
    """

    with tf.variable_scope("crf_loss"):
        trans = tf.get_variable(
                "transition",
                shape=[num_labels, num_labels],
                initializer=tf.contrib.layers.xavier_initializer()
        )
    
    log_likelihood,transition = tf.contrib.crf.crf_log_likelihood(logits, labels, transition_params=trans, sequence_lengths=mask2len)
    loss = tf.math.reduce_mean(-log_likelihood)
   
    return loss, transition


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings):
    sys.stdout.write("Getting model\n")
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    # The model returns the final layer:
    # Tensor of shape [batch_size, seq_length, hidden_size] corresponding
    # to the final hidden of the transformer encoder.
    output_layer = model.get_sequence_output()
    hidden_size = output_layer.shape[-1].value
    print(hidden_size)
    output_weight = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer()
    )
    with tf.variable_scope("loss"):
        if is_training:
            # Keep prob is the probability that each element is kept
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        output_layer = tf.reshape(output_layer, [-1, hidden_size])
        # Matrix multiplication:
        # These are "non normalized" predictions
        # that will be the input for the softmax
        logits = tf.matmul(output_layer, output_weight, transpose_b=True)
        # Include bias in 1D:
        logits = tf.nn.bias_add(logits, output_bias)
        #logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, 7])
        logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, num_labels])
        # Compute the class probablities
        if FLAGS.use_crf:
            mask2len = tf.reduce_sum(input_mask, axis=1)
            loss, transition = crf_loss(logits, labels, input_mask, num_labels, mask2len)	
            predictions, viterbi_score = tf.contrib.crf.crf_decode(logits, transition, mask2len)
            return (loss, logits, predictions)
        else:
            # Use softmax layer
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            # Convert probs to one hot vectors for each token
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
            # Take per exmple loss and reduce it so as to be able to do NER
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_sum(per_example_loss)
            probabilities = tf.nn.softmax(logits, axis=-1)
            predictions = tf.argmax(probabilities,axis=-1)
            return (loss, per_example_loss, logits, predictions)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        #label_mask = features["label_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # TODO: Get per example loss from CRF
        if FLAGS.use_crf:
            (total_loss, logits, predicts) = create_model(
                bert_config, is_training, input_ids, input_mask,segment_ids, label_ids,
                num_labels, use_one_hot_embeddings)
        else:
            (total_loss,  per_example_loss, logits, predicts) = create_model(
                bert_config, is_training, input_ids, input_mask,segment_ids, label_ids,
                num_labels, use_one_hot_embeddings)
        tvars = tf.trainable_variables()
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                precision = tf_metrics.precision(label_ids,predictions, label_length+1 ,[1,2],average="micro")
                recall = tf_metrics.recall(label_ids,predictions,label_length+1 ,[1,2],average="micro")
                f = tf_metrics.f1(label_ids,predictions,label_length+1 ,[1,2],average="micro")
                return {
                    "eval_precision":precision,
                    "eval_recall":recall,
                    "eval_f": f,
                    "predictions": predictions,
                }
            eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode = mode, predictions = predicts, scaffold_fn=scaffold_fn
            )
        return output_spec
    return model_fn


def process_predictions(result, pmids, predict_examples, id2label):
    """
    Function to process BERT tokens, IOB predictions
    and create the full entities
    """

    entity_pmids = []
    entity_labels = []
    whole_tokens = []
    output_token_file = os.path.join(FLAGS.output_dir, "results/BERT_tokens_labels.txt")
    # Reconstitute bert tokens
    with open(output_token_file, "w", encoding="utf*") as f:
        for pmid, predictions, example in zip(pmids, result, predict_examples):
            sub_token = False
            entity_label = ""
            entity_pmid = ""
            prev_label = ""
            token_main = ""
            token_cnt = 0
            for token, p in zip(example.wordpiece_tokens, predictions):
                if p == 0:
                    continue
                label = id2label[p]
                if label == "[CLS]" or label == "[SEP]":
                    continue
                elif label == "X":
                    sub_token = True
                    if token.startswith("#"):
                        token_sub = token.split("#")[2]
                    else:
                        token_sub = token
                    assert token_sub != ""
                    token_main += token_sub
                else:
                    # Some tokens will have no sub tokens, some will, so it is necessary to keep track
                    # of both cases.
                    if sub_token == True or (sub_token == False and token_cnt > 0):
                        whole_tokens.append(token_main)
                        entity_pmids.append(entity_pmid)
                        entity_labels.append(entity_label)
                    entity_label = label
                    entity_pmid = pmid
                    token_main = token
                    sub_token = False
                # Write out tokenization results for all labels
                f.write("{0}\t{1}\t{2}\n".format(pmid, token, label))
                token_cnt += 1
                prev_label = label

    # Combine the B and I entities 
    combined_labels = []
    combined_pmids = []
    combined_tokens = []
    i_token_state = False
    b_token_state = False
    o_label_state = False
    b_token = ""
    prev_label = ""
    token_label = ""
    entity_pmid = ""
    i_cnt = 0
    b_cnt = 0
    cnt = 0
    for pmid, token, label in zip(entity_pmids, whole_tokens, entity_labels):
        # Handle the first line.
        if label == "O":
            prev_label = "O"
            o_label_state = True
            continue
        elif label.startswith("B"):
            # Account for entities that have B- and I- labels and those that have just B-
            # Check if the loop previously visited the I condition.
            if i_token_state == True or (b_token_state == True and i_token_state == False):
                if "-" in b_token:
                    # Account for word piece adding space
                    b_token = "-".join([t.strip() for t in b_token.split("-")])
                if "/" in b_token:
                    b_token = "/".join([t.strip() for t in b_token.split("/")])
                if "(" in b_token:
                    b_token = "(".join([t.strip() for t in b_token.split("(")])
                if ")" in b_token:
                    b_token = ")".join([t.strip() for t in b_token.split(")")])
                combined_labels.append(token_label)
                combined_pmids.append(entity_pmid)
                combined_tokens.append(b_token)
            i_token_state = False
            b_token_state = True
            o_label_state = False
            entity_pmid = pmid
            b_token = token
            token_label = label
            b_cnt += 1
        # Check to see if there are any I- mispredicted. 
        # It is optional to add these to the predictions
        elif label.startswith("I"):
            # Append an inner entity to the previous entity
            i_cnt += 1
            i_token_state = True
            b_token_state = False
            b_token += " " + token
        else:
            print("Unexpected behavior")
            print(pmid, token, label, b_token)
        prev_label = label
        cnt += 1        

    output_predict_file = os.path.join(FLAGS.output_dir, FLAGS.output_preds)
    with open(output_predict_file,'w') as writer:
        for pmid, token, label in zip(combined_pmids, combined_tokens, combined_labels):
            writer.write("{0}\t{1}\t{2}\n".format(pmid, token, label))


def main(_):
    """
    Main function to initiate
    model and train/eval/predict
    """

    start_time = time.time()

    tf.logging.set_verbosity(tf.logging.INFO)
    processors = {
        "ner": NerProcessor
    }

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()

    # Get Multiclass labels or binary class
    if FLAGS.n_labels == 8:
        label_list = processor.get_multi_labels()
    elif FLAGS.n_labels == 2:
        label_list = processor.get_binary_labels()
    label_length = len(label_list)

    # Map the labels to ints
    label_map = {}
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    with open('{}results/label2id.pkl'.format(FLAGS.output_dir),'wb') as w:
        pickle.dump(label_map,w)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir, FLAGS.training_file)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=label_length+1,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        tf.logging.info("Fine-tuning!\n")
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        filed_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file, label_map)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        sys.stdout.write("Calling estimator\n")
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        sys.stdout.write("Finished training\n")

    # For the CER indexing for MEDLINE paper, no evaluation is needed.
    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        filed_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file, label_map)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        eval_steps = None
        if FLAGS.use_tpu:
            eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)
        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)
        # Get metrics from 
        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict or FLAGS.do_CDI:
        token_path = os.path.join(FLAGS.output_dir, "results/token_test.txt")
        with open('{}results/label2id.pkl'.format(FLAGS.output_dir), 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value:key for key,value in label2id.items()}
        if os.path.exists(token_path):
            os.remove(token_path)
        if FLAGS.do_predict:
            predict_examples = processor.get_test_examples(FLAGS.citation_dir)
        elif FLAGS.do_CDI:
            predict_examples = processor.get_CDI_examples(FLAGS.citation_dir)
        pmids = [e.guid for e in predict_examples]

        predict_file = os.path.join(FLAGS.output_dir, "results/predict.tf_record")
        filed_based_convert_examples_to_features(predict_examples, label_list,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file, label_map, mode="test")

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
        if FLAGS.use_tpu:
            # Warning: According to tpu_estimator.py Prediction on TPU is an
            # experimental feature and hence not supported here
            raise ValueError("Prediction in TPU not supported")
        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)
        process_predictions(result, pmids, predict_examples, id2label)
    sys.stdout.write("\n\nRuntime: {0}\n\n".format(time.time() - start_time))

if __name__ == "__main__":
    global label_length
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
