from utils import load_json, split_train_dev_test, dump_to_json
import argparse
import os
from config import Config
from models import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow warnings


def main():
    # 数据的路径
    data_folder = os.path.join('.', 'data')
    # set tasks
    source_dir = os.path.join(data_folder, vec_method, task)
    # create config（创建数据的参数）
    config = Config(task, vec_method, model_name)
    four_datasets = ["20news", "ling_spam", "imdb", "cornell"]
    if task in four_datasets:
        data_set = load_json(os.path.join(source_dir, "data_sets.json"))
        trainset, devset, testset = split_train_dev_test(data_set, dev_ratio=0.1, test_ratio=0.2)
        dataset_name = 0
    else:
        # load datasets
        trainset = load_json(os.path.join(source_dir, 'train.json'))
        devset = load_json(os.path.join(source_dir, 'dev.json'))
        testset = load_json(os.path.join(source_dir, 'test.json'))
        dataset_name = 1
    print(len(trainset))
    print(len(devset))
    print(len(testset))
    # (类别数，句子长度)
    task_classes_seq = {"sst1": (5, 40),
                        "sst2": (2, 40),
                        "cr": (2, 40),
                        "mpqa": (2, 10),
                        "trec": (6, 25),
                        "mr": (2, 50),
                        "subj": (2, 50),
                        "20news": (20, 300),
                        "cornell": (2, 1000),
                        "ling_spam": (2, 1000),
                        "imdb": (2, 500)
                        }

    # training
    batch_size = 50
    epochs = 50
    model = None

    # build model
    if model_name == "TextCNN":
        model = TextCNN(config,
                        num_classes=task_classes_seq[task][0],
                        seq_length=task_classes_seq[task][1],
                        resume_training=resume_training)
    elif model_name == "LSTM":
        model = LSTM(config,
                     num_classes=task_classes_seq[task][0],
                     batch_size=batch_size,
                     seq_length=task_classes_seq[task][1],
                     resume_training=resume_training)
    elif model_name == "Bi_LSTM":
        model = Bi_LSTM(config,
                        num_classes=task_classes_seq[task][0],
                        batch_size=batch_size,
                        seq_length=task_classes_seq[task][1],
                        resume_training=resume_training)
    elif model_name == "RCNN":
        model = RCNN(config,
                     num_classes=task_classes_seq[task][0],
                     batch_size=batch_size,
                     seq_length=task_classes_seq[task][1],
                     resume_training=resume_training)
    elif model_name == "AM_RCNN":
        model = AM_RCNN(config,
                        num_classes=task_classes_seq[task][0],
                        batch_size=batch_size,
                        seq_length=task_classes_seq[task][1],
                        resume_training=resume_training)
    elif model_name == "ARCNN":
        model = ARCNN(config,
                      num_classes=task_classes_seq[task][0],
                      seq_length=task_classes_seq[task][1],
                      batch_size=batch_size,
                      resume_training=resume_training)
    elif model_name == "AM_LSTM":
        model = AM_LSTM(config,
                      num_classes=task_classes_seq[task][0],
                      seq_length=task_classes_seq[task][1],
                      batch_size=batch_size,
                      resume_training=resume_training)
    elif model_name == "AM_Bi_LSTM":
        model = AM_Bi_LSTM(config,
                      num_classes=task_classes_seq[task][0],
                      seq_length=task_classes_seq[task][1],
                      batch_size=batch_size,
                      resume_training=resume_training)
    elif model_name == "ABCNN":
        model = ABCNN(config,
                      num_classes=task_classes_seq[task][0],
                      seq_length=task_classes_seq[task][1],
                      batch_size=batch_size,
                      resume_training=resume_training)
    elif model_name == "ABCNN2":
        model = ABCNN2(config,
                      num_classes=task_classes_seq[task][0],
                      seq_length=task_classes_seq[task][1],
                      batch_size=batch_size,
                      resume_training=resume_training)


    if has_devset:
        loss_dict = model.train(trainset, devset, testset, dataset_name=dataset_name, batch_size=batch_size,
                                epochs=epochs, shuffle=True)
    else:
        trainset = trainset + devset
        loss_dict = model.train(trainset, None, testset, dataset_name=dataset_name, batch_size=batch_size,
                                epochs=epochs, shuffle=True)
    loss_path = "./loss/{}/{}/".format(model_name, vec_method)
    if not os.path.exists(loss_path):
        os.makedirs(loss_path)
    dump_to_json(loss_dict, os.path.join(loss_path, "{}.json".format(task)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='set train task (cr|mpqa|mr|sst1|sst2|subj|trec).')
    parser.add_argument('--vector_method', type=str, required=True, help='glove or word2vec')
    parser.add_argument('--resume_training', type=str, required=True, help='resume previous trained model.')
    parser.add_argument('--has_devset', type=str, required=True, help='indicates if the task has development dataset.')
    parser.add_argument('--model_name', type=str, required=True,
                        help='set model(textcnn, arcnn, lstm, rnn, multi_lstm).')
    args, _ = parser.parse_known_args()
    task = args.task
    vec_method = args.vector_method
    model_name = args.model_name
    resume_training = True if args.resume_training == 'True' else False
    has_devset = True if args.has_devset == 'True' else False
    main()
