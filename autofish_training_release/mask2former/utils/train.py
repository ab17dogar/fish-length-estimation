import os
import argparse
import yaml
import uuid

# detectron2
import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.engine import hooks
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog

# Mask2Former
import sys
sys.path.append("/workspace/Mask2Former")
from mask2former import add_maskformer2_config

# shared_utils
from register_autofish_dataset import Autofish
from tester import Tester
import train_utils
import augmentations
import plot_metrics


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Define the argument for the YAML file path
    parser.add_argument('-y', '--yaml', required=True, type=str, help='Path to the YAML file.')
    args = parser.parse_args()
    return args


def get_train_config(yaml_cfg):
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    config_file = yaml_cfg["model"]
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = yaml_cfg["weights"]
    cfg.DATALOADER.NUM_WORKERS = yaml_cfg["dataloader_workers"]
    cfg.DATASETS.TRAIN = yaml_cfg["train_dataset"]
    cfg.DATASETS.TEST = yaml_cfg["val_dataset"]
    cfg.SOLVER.IMS_PER_BATCH = yaml_cfg["imgs_per_batch"]
    cfg.SOLVER.MAX_ITER = yaml_cfg["training_iters"]
    cfg.SOLVER.STEPS = yaml_cfg["solver_steps"]
    cfg.SOLVER.CHECKPOINT_PERIOD = yaml_cfg["checkpoint_period"]
    cfg.SOLVER.BASE_LR = yaml_cfg["learning_rate"]
    cfg.TEST.EVAL_PERIOD = -1
    cfg.OUTPUT_DIR = yaml_cfg["output_dir"]
    #To change the number of classes -> https://github.com/facebookresearch/Mask2Former/issues/42
    #Changing number of classes raises warning, this is expected behaviour -> https://github.com/facebookresearch/detectron2/issues/196
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes)
    return cfg


if __name__=="__main__":
    args = parse_arguments()
    with open(args.yaml, "r") as f:
        training_yaml = yaml.safe_load(f)

    # To prevent accidental override
    if os.path.exists(training_yaml["output_dir"]):
        training_yaml["output_dir"] += "_{}".format(str(uuid.uuid4())[:4])

    # LOAD AND REGISTER DATASET
    dataset = Autofish.instance_from_yaml(training_yaml)
    dataset.register_all_autofish_splits()

    # COMPILE AUGMENTATIONS
    train_augmentations, test_augmentations =augmentations.get_augmentations(training_yaml)

    # DETECTRON2 CONFIG SETUP
    cfg = get_train_config(training_yaml)
    cfg.freeze()

    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")

    # TRAINER
    train_utils.Mask2FormerAutofishTrainer.data_augmentations = train_augmentations
    trainer = train_utils.Mask2FormerAutofishTrainer(cfg)
    trainer.resume_or_load(resume=False)

    # CUSTOM HOOKS
    if training_yaml['val_dataset'] != []:
        val_loss_hook = train_utils.ValidationLoss(cfg, data_augmentations=test_augmentations)
        trainer.register_hooks([val_loss_hook])
        trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]

    if training_yaml['save_best_hook']['register']:
        save_best_performing_model_hook = hooks.BestCheckpointer(
            eval_period=1,
            checkpointer=trainer.checkpointer,
            val_metric=training_yaml['save_best_hook']['metric'],
            mode=training_yaml['save_best_hook']['mode']
            )
        trainer.register_hooks([save_best_performing_model_hook])
        trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]

    if training_yaml['image_saver_hook']['register']:
        saver_hook = train_utils.ImageSaver(cfg, data_augmentations=train_augmentations, dataset_classes=dataset.classes, log_period=training_yaml['image_saver_hook']['log_period'])
        trainer.register_hooks([saver_hook])

    registered_hooks_named = [x.__class__.__name__ for x in trainer._hooks]
    trainer._hooks.pop(registered_hooks_named.index("EvalHook"))

    print("=================================================")
    print("================REGISTERED HOOKS=================")
    print("=================================================")
    for idx, hook in enumerate(trainer._hooks):
        print(f"{idx} {hook.__class__.__name__}")
    print("=================================================")

    # Save configuration files to the output folder
    with open(os.path.join(cfg.OUTPUT_DIR, "configured_model.yaml"), 'w') as file:
        file.write(str(cfg))

    with open(os.path.join(cfg.OUTPUT_DIR, args.yaml.split("/")[-1]), 'w') as f:
        yaml.safe_dump(training_yaml, f, default_flow_style=False, sort_keys=False)

    # TRAIN
    trainer.train()

    plot_metrics.plot_metrics_json(
        input=os.path.join(cfg.OUTPUT_DIR, "metrics.json"),
        output=os.path.join(cfg.OUTPUT_DIR, "plot.png"),
    )

    # POST-TRAINING EVALUATION
    model_tester = Tester(yaml=training_yaml, train_cfg=get_train_config(training_yaml), data_augmentations=test_augmentations)
    if training_yaml["compute_AP"]:
        model_tester.compute_ap(data_loader_num_worker=training_yaml["dataloader_workers"])
