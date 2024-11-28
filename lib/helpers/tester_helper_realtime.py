import os
import tqdm

import torch
from torchvision import transforms
import numpy as np

from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections
import lib.eval_tools.eval as eval
import requests
from PIL import Image
from io import BytesIO

class Tester(object):
    def __init__(self, cfg, model, data_loader, logger):
        self.cfg = cfg['tester']
        # for eval
        self.eval_cls = cfg['dataset']['eval_cls']
        self.root_dir = cfg['dataset']['root_dir']
        self.label_dir = os.path.join(self.root_dir, 'label_2_4cls_filter_with_roi_for_eval')
        self.calib_dir = os.path.join(self.root_dir, 'calib')
        self.de_norm_dir = os.path.join(self.root_dir, 'denorm')
        self.model = model
        self.data_loader = data_loader
        self.logger = logger
        self.class_name = data_loader.dataset.class_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.cfg.get('resume_model', None):
            load_checkpoint(model = self.model,
                        optimizer = None,
                        filename = self.cfg['resume_model'],
                        logger = self.logger,
                        map_location=self.device)

        self.model.to(self.device)

    def test(self):
        torch.set_grad_enabled(False)
        self.model.eval()
        index = 0

        results = {}
        while True:
            try:
                inputs, calibs, coord_ranges, _, info, calib_pitch_cos, calib_pitch_sin = self.data_loader.dataset.get_data(index)

                # package to tensor
                inputs = np.expand_dims(inputs, axis=0)
                inputs = torch.tensor(inputs, dtype=torch.float32)
                calibs = np.expand_dims(calibs, axis=0)
                calibs = torch.tensor(calibs, dtype=torch.float32)
                coord_ranges = np.expand_dims(coord_ranges, axis=0)
                coord_ranges = torch.tensor(coord_ranges, dtype=torch.float32)
                calib_pitch_cos = torch.tensor([calib_pitch_cos], dtype=torch.float32)  # [1]
                calib_pitch_sin = torch.tensor([calib_pitch_sin], dtype=torch.float32)  # [1]
                info['img_size'] = np.expand_dims(info['img_size'], axis=0) 
                info['img_size'] =  torch.tensor(info['img_size'], dtype=torch.int8)
                info['bbox_downsample_ratio'] = np.expand_dims(info['bbox_downsample_ratio'], axis=0) 
                info['bbox_downsample_ratio'] =  torch.tensor(info['bbox_downsample_ratio'], dtype=torch.float32)

                # print(f"  inputs: type={type(inputs)}, shape={inputs.shape if isinstance(inputs, torch.Tensor) else 'N/A'}")
                # print(f"  calibs: type={type(calibs)}, shape={calibs.shape if isinstance(calibs, torch.Tensor) else 'N/A'}")
                # print(f"  coord_ranges: type={type(coord_ranges)}, shape={coord_ranges.shape if isinstance(coord_ranges, torch.Tensor) else 'N/A'}")
                # print(f"  info: type={type(info)}, keys={info.keys() if isinstance(info, dict) else 'N/A'}")
                # print(f"  info['img_size']: type={type(info['img_size'])}, shape={info['img_size'].shape if isinstance(info['img_size'], torch.Tensor) else 'N/A'}")
                # print(f"  calib_pitch_cos: type={type(calib_pitch_cos)}, shape={calib_pitch_cos.shape if isinstance(calib_pitch_cos, torch.Tensor) else 'N/A'}")
                # print(f"  calib_pitch_sin: type={type(calib_pitch_sin)}, shape={calib_pitch_sin.shape if isinstance(calib_pitch_sin, torch.Tensor) else 'N/A'}")
                
                # load evaluation data and move data to current device.
                inputs = inputs.to(self.device)
                calibs = calibs.to(self.device)
                coord_ranges = coord_ranges.to(self.device)
                calib_pitch_cos = calib_pitch_cos.to(self.device)
                calib_pitch_sin = calib_pitch_sin.to(self.device)

                # the outputs of centernet
                outputs = self.model(inputs,coord_ranges,calibs,K=50,mode='val', calib_pitch_sin=calib_pitch_sin, calib_pitch_cos=calib_pitch_cos)

                dets = extract_dets_from_outputs(outputs,calibs, K=50)
                dets = dets.detach().cpu().numpy()
                
                # get corresponding calibs & transform tensor to numpy
                calibs = [self.data_loader.dataset.get_calib(0)]
                denorms = [self.data_loader.dataset.get_denorm(0)]
                info['img_id'] = info['img_id']
                info['img_size'] = info['img_size'].detach().cpu().numpy()
                info['bbox_downsample_ratio'] = info['bbox_downsample_ratio'].detach().cpu().numpy()
                cls_mean_size = self.data_loader.dataset.cls_mean_size
                dets = decode_detections(dets = dets,
                                        info = info,
                                        calibs = calibs,
                                        denorms = denorms,
                                        cls_mean_size=cls_mean_size,
                                        threshold = self.cfg['threshold'])   
                                
                results.update(dets)

                # update index
                index += 1
            
            except KeyboardInterrupt:
                print("Real-time evaluation stopped by user.")
                break
            except Exception as e:
                print(f"Error during real-time evaluation: {e}")

        out_dir = os.path.join(self.cfg['out_dir'])
        self.save_results(results,out_dir)
        # progress_bar.close()
        return 0

    def save_results(self, results, output_dir='./outputs'):
        output_dir = os.path.join(output_dir, 'data')
        os.makedirs(output_dir, exist_ok=True)

        # img_id is frame name
        for img_id in results.keys():
            out_path = os.path.join(output_dir, str(img_id)+'.txt')
            f = open(out_path, 'w')
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()    