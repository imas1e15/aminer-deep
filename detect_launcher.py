import argparse
import sys
import json
import time

import zmq
from auxiliaries.detector import Detector
from auxiliaries.utils import *
from models.deeplog import deeplog
from auxiliaries.trainer import Trainer
from config import *

import sys
sys.path.append('../')

seed_everything(seed=1234)

def train():
    Model = deeplog(input_size=options['input_size'],
                    hidden_size=options['hidden_size'],
                    num_layers=options['num_layers'],
                    num_keys=options['num_classes'])
    trainer = Trainer(Model, options)
    trainer.start_train()

def buffer_sequence(seq):
    seq = list(map(lambda n: n, map(int, line.strip().split())))

def write_sequence(data, filename):
    with open(options["data_save_dir"] + filename,'a') as seq_file:
        seq_file.write("{}: {}: {}: {}\n".format(data[0], data[1], data[2], data[3]))
    
def start_detector(options, io_mode, filename=None):
    if not options["learn_mode"]:
        Model = deeplog(
            input_size=options["input_size"],
            hidden_size=options["hidden_size"],
            num_layers=options["num_layers"],
            num_keys=options["num_classes"],
        )
        dectector = Detector(Model, options)
        l_model = dectector.load_model()
    context = zmq.Context()
    zmq_pub_socket = context.socket(zmq.PUB)
    zmq_pub_socket.connect(options["zmq_sub_endpoint"])
    zmq_sub_socket = context.socket(zmq.SUB)
    zmq_sub_socket.connect(options["zmq_pub_endpoint"])
    zmq_sub_socket.setsockopt_string(zmq.SUBSCRIBE, options["zmq_aminer_top"])
    
    while True:
        #time.sleep(2)
        print("Wating for the aminer outpute .......")
        msg = zmq_sub_socket.recv_string()
        top, group, seq = msg.split(":")
        seq = seq.replace("[", "").replace("]", "").replace(",", "").replace("\"", "").split(" ")
        seq = [int(x) for x in seq]
        if not options["learn_mode"]:
            result = dectector.detect_anomaly(l_model, seq[:-1], seq[-1])
        else:
            result = False
        result = [group, seq[:-1],seq[-1] ,result]
        print("Sending the detector result: {}".format(result))
        zmq_pub_socket.send_string("{}:{}:{}".format(options["zmq_detector_top"], group, json.dumps(result)))
        if io_mode:
            print("Writing aminer & detector output in {}".format(filename))
            write_sequence(result,filename)

if __name__ == "__main__":
    filename = time.strftime('%H%M%S')
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['withio', 'without', 'train'])
    args = parser.parse_args()
    if args.mode == 'withio':
        start_detector(options,True, filename)
    elif args.mode == 'without':
        start_detector(options,False)
    elif args.mode == 'train':
        train()
    else:
        print('Invalid input')