# import importlib
# from argparse import ArgumentParser
# from omegaconf import OmegaConf
# from os.path import join as pjoin
# import os
# import glob


# def get_module_config(cfg, filepath="./configs"):
#     """
#     Load yaml config files from subfolders
#     """
#     print("filepath: ",filepath)
#     yamls = glob.glob(pjoin(filepath, '*', '*.yaml'))
#     yamls = [y.replace(filepath, '') for y in yamls]
#     for yaml in yamls:
#         nodes = yaml.replace('.yaml', '').replace(os.sep, '.')
#         nodes = nodes[1:] if nodes[0] == '.' else nodes
#         OmegaConf.update(cfg, nodes, OmegaConf.load('./configs' + yaml))

#     return cfg


# def get_obj_from_str(string, reload=False):
#     """
#     Get object from string
#     """
#     print("RELOAD: ",reload)
#     print("String: ",string)

#     module, cls = string.rsplit(".", 1)
#     if reload:
#         module_imp = importlib.import_module(module)
#         importlib.reload(module_imp)
#     return getattr(importlib.import_module(module, package=None), cls)


# def instantiate_from_config(config):
#     """
#     Instantiate object from config
#     """
#     if not "target" in config:
#         raise KeyError("Expected key `target` to instantiate.")
#     return get_obj_from_str(config["target"])(**config.get("params", dict()))


# def resume_config(cfg: OmegaConf):
#     """
#     Resume model and wandb
#     """
    
#     if cfg.TRAIN.RESUME:
#         resume = cfg.TRAIN.RESUME
#         if os.path.exists(resume):
#             # Checkpoints
#             cfg.TRAIN.PRETRAINED = pjoin(resume, "checkpoints", "last.ckpt")
#             # Wandb
#             wandb_files = os.listdir(pjoin(resume, "wandb", "latest-run"))
#             wandb_run = [item for item in wandb_files if "run-" in item][0]
#             cfg.LOGGER.WANDB.params.id = wandb_run.replace("run-","").replace(".wandb", "")
#         else:
#             raise ValueError("Resume path is not right.")

#     return cfg

# def parse_args(phase="train"):
#     """
#     Parse arguments and load config files
#     """

#     parser = ArgumentParser()
#     group = parser.add_argument_group("Training options")

#     # Assets
#     group.add_argument(
#         "--cfg_assets",
#         type=str,
#         required=False,
#         default="./configs/assets.yaml",
#         help="config file for asset paths",
#     )

#     # Default config
#     if phase in ["train", "test", "demo"]:
#         cfg_defualt = "./configs/default.yaml"
#     elif phase == "render":
#         cfg_defualt = "./configs/render.yaml"
#     elif phase == "webui":
#         cfg_defualt = "./configs/webui.yaml"
        
#     group.add_argument(
#         "--cfg",
#         type=str,
#         required=False,
#         default=cfg_defualt,
#         help="config file",
#     )

#     # Parse for each phase
#     if phase in ["train", "test"]:
#         group.add_argument("--batch_size",
#                            type=int,
#                            required=False,
#                            help="training batch size")
#         group.add_argument("--num_nodes",
#                            type=int,
#                            required=False,
#                            help="number of nodes")
#         group.add_argument("--device",
#                            type=int,
#                            nargs="+",
#                            required=False,
#                            help="training device")
#         group.add_argument("--task",
#                            type=str,
#                            required=False,
#                            help="evaluation task type")
#         group.add_argument("--nodebug",
#                            action="store_true",
#                            required=False,
#                            help="debug or not")


#     if phase == "demo":
#         group.add_argument("--task",
#             type=str,
#             required=False,
#             help="evaluation task type")
#         group.add_argument(
#             "--example",
#             type=str,
#             required=False,
#             help="input text and lengths with txt format",
#         )
#         group.add_argument(
#             "--out_dir",
#             type=str,
#             required=False,
#             help="output dir",
#         )

#     if phase == "render":
#         group.add_argument("--npy",
#                            type=str,
#                            required=False,
#                            default=None,
#                            help="npy motion files")
#         group.add_argument("--dir",
#                            type=str,
#                            required=False,
#                            default=None,
#                            help="npy motion folder")
#         group.add_argument("--fps",
#                     type=int,
#                     required=False,
#                     default=30,
#                     help="render fps")
#         group.add_argument(
#             "--mode",
#             type=str,
#             required=False,
#             default="sequence",
#             help="render target: video, sequence, frame",
#         )

#     params = parser.parse_args()
    
#     # Load yaml config files
#     OmegaConf.register_new_resolver("eval", eval)
#     cfg_assets = OmegaConf.load(params.cfg_assets)
#     cfg_base = OmegaConf.load(pjoin(cfg_assets.CONFIG_FOLDER, 'default.yaml'))
#     cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(params.cfg))
#     if not cfg_exp.FULL_CONFIG:
#         cfg_exp = get_module_config(cfg_exp, cfg_assets.CONFIG_FOLDER)
#     print("FESFES: ",cfg_exp.model, cfg_exp.model.target)
#     cfg = OmegaConf.merge(cfg_exp, cfg_assets)

#     # Update config with arguments
#     if phase in ["train", "test"]:
#         cfg.TRAIN.BATCH_SIZE = params.batch_size if params.batch_size else cfg.TRAIN.BATCH_SIZE
#         cfg.DEVICE = params.device if params.device else cfg.DEVICE
#         cfg.NUM_NODES = params.num_nodes if params.num_nodes else cfg.NUM_NODES
#         cfg.model.params.task = params.task if params.task else cfg.model.params.task
#         cfg.DEBUG = not params.nodebug if params.nodebug is not None else cfg.DEBUG

#         # Force no debug in test
#         if phase == "test":
#             cfg.DEBUG = False
#             cfg.DEVICE = [0]
#             print("Force no debugging and one gpu when testing")

#     if phase == "demo":
#         cfg.DEMO.EXAMPLE = params.example
#         cfg.DEMO.TASK = params.task
#         cfg.TEST.FOLDER = params.out_dir if params.out_dir else cfg.TEST.FOLDER
#         os.makedirs(cfg.TEST.FOLDER, exist_ok=True)

#     if phase == "render":
#         if params.npy:
#             cfg.RENDER.NPY = params.npy
#             cfg.RENDER.INPUT_MODE = "npy"
#         if params.dir:
#             cfg.RENDER.DIR = params.dir
#             cfg.RENDER.INPUT_MODE = "dir"
#         if params.fps:
#             cfg.RENDER.FPS = float(params.fps)
#         cfg.RENDER.MODE = params.mode

#     # Debug mode
#     if cfg.DEBUG:
#         cfg.NAME = "debug--" + cfg.NAME
#         cfg.LOGGER.WANDB.params.offline = True
#         cfg.LOGGER.VAL_EVERY_STEPS = 1
        
#     # Resume config
#     cfg = resume_config(cfg)

#     return cfg
import importlib
from argparse import ArgumentParser
from omegaconf import OmegaConf
import os


def get_module_config(cfg_model, path="modules"):
    files = os.listdir(f'./configs/{path}/')
    for file in files:
        if file.endswith('.yaml'):
            with open(f'./configs/{path}/' + file, 'r') as f:
                cfg_model.merge_with(OmegaConf.load(f))
    return cfg_model


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def parse_args(phase="train"):
    parser = ArgumentParser()

    group = parser.add_argument_group("Training options")
    if phase in ["train", "test", "demo"]:
        group.add_argument(
            "--cfg",
            type=str,
            required=False,
            default="./configs/config.yaml",
            help="config file",
        )
        group.add_argument(
            "--cfg_assets",
            type=str,
            required=False,
            default="./configs/assets.yaml",
            help="config file for asset paths",
        )
        group.add_argument("--batch_size",
                           type=int,
                           required=False,
                           help="training batch size")
        group.add_argument("--device",
                           type=int,
                           nargs="+",
                           required=False,
                           help="training device")
        group.add_argument("--nodebug",
                           action="store_true",
                           required=False,
                           help="debug or not")
        group.add_argument("--dir",
                           type=str,
                           required=False,
                           help="evaluate existing npys")

    if phase == "demo":
        # group.add_argument("--motion_transfer", action='store_true', help="Motion Distribution Transfer")
        group.add_argument("--render",
                           action="store_true",
                           help="Render visulizaed figures")
        group.add_argument("--render_mode", type=str, help="video or sequence")
        group.add_argument(
            "--frame_rate",
            type=float,
            default=12.5,
            help="the frame rate for the input/output motion",
        )
        group.add_argument(
            "--replication",
            type=int,
            default=1,
            help="the frame rate for the input/output motion",
        )
        group.add_argument(
            "--example",
            type=str,
            required=False,
            help="input text and lengths with txt format",
        )
        group.add_argument(
            "--task",
            type=str,
            required=False,
            help="random_sampling, reconstrucion or text_motion",
        )
        group.add_argument(
            "--out_dir",
            type=str,
            required=False,
            help="output dir",
        )
        group.add_argument(
            "--allinone",
            action="store_true",
            required=False,
            help="output seperate or combined npy file",
        )

    if phase == "render":
        group.add_argument(
            "--cfg",
            type=str,
            required=False,
            default="./configs/render.yaml",
            help="config file",
        )
        group.add_argument(
            "--cfg_assets",
            type=str,
            required=False,
            default="./configs/assets.yaml",
            help="config file for asset paths",
        )
        # group.add_argument("--motion_transfer", action='store_true', help="Motion Distribution Transfer")
        group.add_argument("--npy",
                           type=str,
                           required=False,
                           default=None,
                           help="npy motion files")
        group.add_argument("--dir",
                           type=str,
                           required=False,
                           default=None,
                           help="npy motion folder")
        group.add_argument(
            "--mode",
            type=str,
            required=False,
            default="sequence",
            help="render target: video, sequence, frame",
        )
        group.add_argument(
            "--joint_type",
            type=str,
            required=False,
            default=None,
            help="mmm or vertices for skeleton",
        )

    # remove None params, and create a dictionnary
    params = parser.parse_args()
    # params = {key: val for key, val in vars(opt).items() if val is not None}

    # update config from files
    cfg_base = OmegaConf.load('./configs/base.yaml')
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(params.cfg))
    print("KAKAKAKAKAK: ")
    print(cfg_exp.model.target)
    cfg_model = get_module_config(cfg_exp.model, cfg_exp.model.target)
    cfg_assets = OmegaConf.load(params.cfg_assets)
    cfg = OmegaConf.merge(cfg_exp, cfg_model, cfg_assets)

    if phase in ["train", "test"]:
        cfg.TRAIN.BATCH_SIZE = (params.batch_size
                                if params.batch_size else cfg.TRAIN.BATCH_SIZE)
        cfg.DEVICE = params.device if params.device else cfg.DEVICE
        cfg.DEBUG = not params.nodebug if params.nodebug is not None else cfg.DEBUG

        # no debug in test
        cfg.DEBUG = False if phase == "test" else cfg.DEBUG
        if phase == "test":
            cfg.DEBUG = False
            cfg.DEVICE = [0]
            print("Force no debugging and one gpu when testing")
        cfg.TEST.TEST_DIR = params.dir if params.dir else cfg.TEST.TEST_DIR

    if phase == "demo":
        # cfg.DEMO.MOTION_TRANSFER = params.motion_transfer
        cfg.DEMO.RENDER = params.render
        cfg.DEMO.FRAME_RATE = params.frame_rate
        cfg.DEMO.EXAMPLE = params.example
        cfg.DEMO.TASK = params.task
        cfg.TEST.FOLDER = params.out_dir if params.dir else cfg.TEST.FOLDER
        cfg.DEMO.REPLICATION = params.replication
        cfg.DEMO.OUTALL = params.allinone

    if phase == "render":
        if params.npy:
            cfg.RENDER.NPY = params.npy
            cfg.RENDER.INPUT_MODE = "npy"
        if params.dir:
            cfg.RENDER.DIR = params.dir
            cfg.RENDER.INPUT_MODE = "dir"
        cfg.RENDER.JOINT_TYPE = params.joint_type
        cfg.RENDER.MODE = params.mode

    # debug mode
    if cfg.DEBUG:
        cfg.NAME = "debug--" + cfg.NAME
        cfg.LOGGER.WANDB.OFFLINE = True
        cfg.LOGGER.VAL_EVERY_STEPS = 1

    return cfg
