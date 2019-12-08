import os
from utils import load_json

vector_method = ['glove', 'word2vec']
datasets = ['mr', 'sst1', 'sst2', 'cr', 'mpqa', 'trec', 'subj', 'cornell', 'ling_spam', 'imdb', '20news']
# models = ['LSTM', 'Bi_LSTM', 'TextCNN', 'RCNN', 'AM_RCNN']
models = ['AM_LSTM', 'AM_Bi_LSTM']
path = "./loss/"
path2 = path + "result/"
for model in models:
    p = path2 + "{}/".format(model)
    if not os.path.exists(p):
        os.makedirs(p)
    for vec in vector_method:
        with open(os.path.join(path2, model, "{}.txt".format(vec)), "w", encoding="utf-8") as f:
            for dp in datasets:
                print(model, " ", vec, " ", dp)
                data_path = os.path.join(path, model, vec, "{}.json".format(dp))
                data = load_json(data_path)
                f.write("{}\tprecision\trecall\tf1-score\tsupport\t{}\n".format(dp, data["best_score"]))
                if model != "RCNN":
                    report = data["report"]
                    for i, rep in enumerate(report):
                        f.write("{}\t{}\t{}\t{}\t{}\n".format(rep[0], rep[1], rep[2], rep[3],rep[4]))
                accuracy = data["accuracy_list"]
                losses = data["losses"]

                f.write("accuracy\t")
                for i, acc in enumerate(accuracy):
                    if i != len(accuracy)-1:
                        f.write("{}\t".format(acc))
                    else:
                        f.write("{}\n".format(acc))
                f.write("loss\t")
                for i, ls in enumerate(losses):
                    if i != len(losses) - 1:
                        f.write("{}\t".format(ls))
                    else:
                        f.write("{}\n".format(ls))
                f.write("\n\n")


