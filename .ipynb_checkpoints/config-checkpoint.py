options = dict()
options["data_save_dir"] = "/home/ubuntu/aminer-deep/data/"
options['data_file_name'] = "144431"
options["device"] = "cpu"

options['sequentials'] = True

# Model
options["input_size"] = 1
options["hidden_size"] = 64
options["num_layers"] = 2
options["num_classes"] = 269

# Train
options["batch_size"] = 2048
options["accumulation_step"] = 1
options["optimizer"] = "adam"
options["lr"] = 0.001
options["max_epoch"] = 10
options["lr_step"] = (300, 350)
options["lr_decay_ratio"] = 0.1
options["resume_path"] = None
options["model_name"] = "audit"
options["save_dir"] = "/home/ubuntu/aminer-deep/result/aminer-deep/"

# Predict
options[
    "model_path"
] = "/home/ubuntu/aminer-deep/result/deeplog/audit_last.pth"
options["num_candidates"] = 9

# ZMQ configration, the endpoint is presented from proxy point of view
options["zmq_pub_endpoint"] = "tcp://127.0.0.1:5559"
options["zmq_sub_endpoint"] = "tcp://127.0.0.1:5560"
options["zmq_aminer_top"] = "AminerOut"
options["zmq_detector_top"] = "DetectorOut"
options["learn_mode"] = True
