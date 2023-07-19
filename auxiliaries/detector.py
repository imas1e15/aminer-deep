import time
from collections import Counter

import torch
import torch.nn as nn
from torch.autograd import Variable


class Detector:
    def __init__(self, model, options):
        self.device = options["device"]
        self.model = model
        self.model_path = options["model_path"]
        self.num_candidates = options["num_candidates"]
        self.num_classes = options["num_classes"]
        self.input_size = options["input_size"]

    def load_model(self):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)["state_dict"])
        model.eval()
        print("model_path: {}".format(self.model_path))
        return model

    def detect_anomaly(self, model, seq, label):
        with torch.no_grad():
            seq1 = [0] * self.num_classes  #
            log_conuter = Counter(seq)  #
            for key in log_conuter:
                seq1[key] = log_conuter[key]

            seq = (
                torch.tensor(seq, dtype=torch.float)
                .view(-1, len(seq), self.input_size)
                .to(self.device)
            )
            seq1 = (
                torch.tensor(seq1, dtype=torch.float)
                .view(-1, self.num_classes, self.input_size)
                .to(self.device)
            )
            label = torch.tensor(label).view(-1).to(self.device)
            output = model(features=[seq, seq1], device=self.device)
            predicted = torch.argsort(output, 1)[0][-self.num_candidates :]
            if label not in predicted:
                return True
            else:
                return False