from Models.CustomRNN import CustomRNN
from datasets.datasets import CopyInputDataset, PermutedMnist, DenoisingDataset, RememberLinePMnist
from cells.cells import *
import multiprocessing
from multiprocessing import Process
import tensorflow as tf
import multiprocessing

# Uncomment to run on CPU
tf.config.experimental.set_visible_devices([], 'GPU')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

test_name = "default_test3"

# Note that the datasets are smaller than in the paper for execution time,
# and that not as many experiments are carried here.
def run(test_name, cell_type, benchmark):
    if benchmark == 0:
        dataset = CopyInputDataset
        datasets = [dataset(40000, 10000, 100, 5), dataset(40000, 50000, 100, 600)]
    if benchmark == 1:
        dataset = DenoisingDataset
        datasets = [dataset(60000, 10000, 100, 5, 400, 0), dataset(60000, 10000, 100, 5, 400, 200)]
    if benchmark == 2:
        dataset = PermutedMnist
        datasets = [dataset(100)]
    if benchmark == 3:
        dataset = RememberLinePMnist
        datasets = [dataset(100, 500-28, True), dataset(100, 100-28, False), dataset(100, 500-28, False)]
    for dataset in datasets:
        print("Tackeling : " + dataset.name + "/" + test_name)
        model = CustomRNN(dataset, [128, 128], [], cell_type, "paper_models/" + dataset.name + "/" + test_name + "/")
        model.train(100)
        del(model)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    for cell_type in get_cells_list():
        print(cell_type.__name__)
        p = Process(target = run, args=(test_name, cell_type, 3))
        p.start()
