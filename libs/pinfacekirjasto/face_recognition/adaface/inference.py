import os
os.environ["NNPACK_ENABLE"] = "0"

from libs.pinfacekirjasto.face_recognition.adaface import net
import torch
from torch.autograd import Variable

torch.backends.mkldnn.enabled = False
torch._C._set_nnpack_enabled(False)
torch.backends.nnpack.enabled = False

torch.backends.nnpack.enabled = False

import numpy as np

# kkk
"""
adaface_models = {
    'ir_50':"pretrained/adaface_ir50_ms1mv2.ckpt",
}
"""
# kkk
load_model_cpu = "cuda:0"
if not torch.cuda.is_available():
    load_model_cpu = "cpu"
# load_model_cpu = 'cpu'

# print('[i] load_model_cpu =',load_model_cpu)

arch = "ir_101"  # Архитектура модели
adaface_models = {
    # 'ir_50': "pretrained/adaface_ir_ms1mv2.ckpt",  # Пример другой модели
    arch: "pretrained/adaface_ir101_webface4m.ckpt",  # Путь к модели
}


# kkk
# def load_pretrained_model(architecture='ir_50'):
def load_pretrained_model(architecture=arch):
    # load model and pretrained statedict
    # kkk
    # assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    # kkk
    # statedict = torch.load(adaface_models[architecture])['state_dict']
    statedict = torch.load(
        adaface_models[architecture], map_location=load_model_cpu, weights_only=False
    )["state_dict"]
    # statedict = torch.load(adaface_models[architecture], map_location=load_model_cpu, weights_only=True)['state_dict']

    model_statedict = {
        key[6:]: val for key, val in statedict.items() if key.startswith("model.")
    }
    model.load_state_dict(model_statedict)
    model.eval()

    # kkk
    # Перемещение модели на GPU, если выбрано
    # if (load_model_cpu == 'cuda:0' or load_model_cpu == 'cuda'):
    #    model = model.to(load_model_cpu)

    return model


def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:, :, ::-1] / 255.0) - 0.5) / 0.5
    # kkk
    # tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
    tensor = torch.tensor(np.array([brg_img.transpose(2, 0, 1)])).float()

    """
    kkk
    # Преобразование в тензор
    if (load_model_cpu == 'cuda:0' or load_model_cpu == 'cuda'):
        tensor = torch.tensor(np.array([brg_img.transpose(2, 0, 1)]), dtype=torch.float, device=load_model_cpu_device)
    else:
        tensor = torch.tensor(np.array([brg_img.transpose(2, 0, 1)])).float()
    """

    return tensor
