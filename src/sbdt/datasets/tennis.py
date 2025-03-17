import os
import os.path as osp
import logging
from omegaconf import DictConfig
from tqdm import tqdm
import pandas as pd
import numpy as np

from utils import Center
from utils import load_csv_tennis as load_csv
from utils import refine_gt_clip_tennis as refine_gt_clip

log = logging.getLogger(__name__)

def get_clips(cfg, train_or_test='test', gt=True):
    root_dir      = cfg['dataset']['root_dir']
    matches       = cfg['dataset'][train_or_test]['matches']
    csv_filename  = cfg['dataset']['csv_filename']
    ext           = cfg['dataset']['ext']
    visible_flags = cfg['dataset']['visible_flags']

    clip_dict = {}
    for match in matches:
        match_video_dir = osp.join(root_dir, match)
        clip_names     = os.listdir(match_video_dir)
        clip_names.sort()
        for clip_name in clip_names:
            clip_dir      = osp.join(root_dir, match, clip_name)
            clip_csv_path = osp.join(root_dir, match, clip_name, csv_filename)
            frame_names = []
            for frame_name in os.listdir(clip_dir):
                if frame_name.endswith(ext):
                    frame_names.append(frame_name)
            frame_names.sort()
            ball_xyvs = load_csv(clip_csv_path, visible_flags) if gt else None
            clip_dict[(match, clip_name)] = {'clip_dir_or_path': clip_dir, 'clip_gt_dict': ball_xyvs, 'frame_names': frame_names}

    return clip_dict

class Tennis(object):
    def __init__(self, 
                 cfg: DictConfig,
    ):
        self._root_dir             = cfg['dataset']['root_dir']
        self._ext                  = cfg['dataset']['ext']
        self._csv_filename         = cfg['dataset']['csv_filename']
        self._visible_flags        = cfg['dataset']['visible_flags']
        self._train_matches        = cfg['dataset']['train']['matches']
        self._test_matches         = cfg['dataset']['test']['matches']
        self._train_num_clip_ratio = cfg['dataset']['train']['num_clip_ratio']
        self._test_num_clip_ratio  = cfg['dataset']['test']['num_clip_ratio']

        self._train_refine_npz_path = cfg['dataset']['train']['refine_npz_path']
        self._test_refine_npz_path  = cfg['dataset']['test']['refine_npz_path']

        self._frames_in  = cfg['model']['frames_in']
        self._frames_out = cfg['model']['frames_out']
        self._step       = cfg['detector']['step']

        self._load_train      = cfg['dataloader']['train']
        self._load_test       = cfg['dataloader']['test']
        self._load_train_clip = cfg['dataloader']['train_clip']
        self._load_test_clip  = cfg['dataloader']['test_clip']

        self._train_all        = []
        self._train_clips      = {}
        self._train_clip_gts   = {}
        self._train_clip_disps = {}
        if self._load_train or self._load_train_clip:
            train_outputs = self._gen_seq_list(self._train_matches, self._train_num_clip_ratio, self._train_refine_npz_path)
            self._train_all                = train_outputs['seq_list'] 
            self._train_num_frames         = train_outputs['num_frames']
            self._train_num_frames_with_gt = train_outputs['num_frames_with_gt']
            self._train_num_matches        = train_outputs['num_matches']
            self._train_num_rallies        = train_outputs['num_rallies']
            self._train_disp_mean          = train_outputs['disp_mean']
            self._train_disp_std           = train_outputs['disp_std']
            if self._load_train_clip:
                self._train_clips      = train_outputs['clip_seq_list_dict']
                self._train_clip_gts   = train_outputs['clip_seq_gt_dict_dict']
                self._train_clip_disps = train_outputs['clip_seq_disps']

        self._test_all        = []
        self._test_clips      = {}
        self._test_clip_gts   = {}
        self._test_clip_disps = {}
        if self._load_test or self._load_test_clip:
            test_outputs  = self._gen_seq_list(self._test_matches, self._test_num_clip_ratio, self._test_refine_npz_path)
            self._test_all                 = test_outputs['seq_list']
            self._test_num_frames          = test_outputs['num_frames']
            self._test_num_frames_with_gt  = test_outputs['num_frames_with_gt']
            self._test_num_matches         = test_outputs['num_matches']
            self._test_num_rallies         = test_outputs['num_rallies']
            self._test_disp_mean           = test_outputs['disp_mean']
            self._test_disp_std            = test_outputs['disp_std']
            if self._load_test_clip:
                self._test_clips               = test_outputs['clip_seq_list_dict']
                self._test_clip_gts            = test_outputs['clip_seq_gt_dict_dict']
                self._test_clip_disps          = test_outputs['clip_seq_disps']

        # show stats
        log.info('=> Tennis loaded' )
        log.info("Dataset statistics:")
        log.info("-------------------------------------------------------------------------------------")
        log.info("subset          | # batch | # frame | # frame w/ gt | # rally | # match | disp[pixel]")
        log.info("-------------------------------------------------------------------------------------")
        if self._load_train:
            log.info("train           | {:7d} | {:7d} | {:13d} | {:7d} | {:7d} | {:2.1f}+/-{:2.1f}".format(len(self._train_all), self._train_num_frames, self._train_num_frames_with_gt, self._train_num_rallies, self._train_num_matches, self._train_disp_mean, self._train_disp_std ) )
        if self._load_train_clip:
            num_items_all          = 0
            num_frames_all         = 0
            num_frames_with_gt_all = 0
            num_clips_all          = 0
            disps_all              = []
            for key, clip in self._train_clips.items():
                num_items  = len(clip)
                num_frames = 0
                for tmp in clip:
                    num_frames += len( tmp['frames'] )
                num_frames_with_gt = num_frames
                clip_name = '{}_{}'.format(key[0], key[1])
                disps     = np.array( self._train_clip_disps[key] )
                log.info("{} | {:7d} | {:7d} | {:13d} |         |         | {:2.1f}+/-{:2.1f}".format(clip_name, num_items, num_frames, num_frames_with_gt, np.mean(disps), np.std(disps) ))
            
                num_items_all          += num_items
                num_frames_all         += num_frames
                num_frames_with_gt_all += num_frames_with_gt
                disps_all.extend(disps)
                num_clips_all += 1
            log.info("all         | {:7d} | {:7d} | {:13d} | {:7d} |         | {:2.1f}+/-{:2.1f}".format(num_items_all, num_frames_all, num_frames_with_gt_all, num_clips_all, np.mean(disps_all), np.std(disps_all) ))
        if self._load_test:
            log.info("test            | {:7d} | {:7d} | {:13d} | {:7d} | {:7d} | {:2.1f}+/-{:2.1f}".format(len(self._test_all), self._test_num_frames, self._test_num_frames_with_gt, self._test_num_rallies, self._test_num_matches, self._test_disp_mean, self._test_disp_std) )
        if self._load_test_clip:
            num_items_all          = 0
            num_frames_all         = 0
            num_frames_with_gt_all = 0
            num_clips_all          = 0
            disps_all              = []
            for key, test_clip in self._test_clips.items():
                num_items  = len(test_clip)
                num_frames = 0
                for tmp in test_clip:
                    num_frames += len( tmp['frames'] )
                num_frames_with_gt = num_frames
                clip_name = '{}_{}'.format(key[0], key[1])
                disps     = np.array( self._test_clip_disps[key] )
                log.info("{} | {:7d} | {:7d} | {:13d} |         |         | {:2.1f}+/-{:2.1f}".format(clip_name, num_items, num_frames, num_frames_with_gt, np.mean(disps), np.std(disps) ))
            
                num_items_all          += num_items
                num_frames_all         += num_frames
                num_frames_with_gt_all += num_frames_with_gt
                disps_all.extend(disps)
                num_clips_all += 1
            log.info("all         | {:7d} | {:7d} | {:13d} | {:7d} |         | {:2.1f}+/-{:2.1f}".format(num_items_all, num_frames_all, num_frames_with_gt_all, num_clips_all, np.mean(disps_all), np.std(disps_all) ))
        log.info("-------------------------------------------------------------------------------------")

    def _gen_seq_list(self, 
                      matches, 
                      num_clip_ratio, 
                      refine_npz_path=None,
    ):
        if refine_npz_path is not None:
            log.info('refine gt ball positions with {}'.format(refine_npz_path))

        seq_list              = []
        clip_seq_list_dict    = {}
        clip_seq_gt_dict_dict = {}
        clip_seq_disps        = {}
        num_frames         = 0
        num_matches        = len(matches)
        num_rallies        = 0
        num_frames_with_gt = 0
        disps              = []
        for match in matches:
            match_clip_dir = osp.join(self._root_dir, match)
            clip_names     = os.listdir(match_clip_dir)
            clip_names.sort()
            clip_names = clip_names[:int(len(clip_names)*num_clip_ratio)]
            num_rallies += len(clip_names)
            for clip_name in clip_names:
                clip_seq_list    = []
                clip_seq_gt_dict = {}
                clip_frame_dir   = osp.join(self._root_dir, match, clip_name)
                clip_csv_path    = osp.join(self._root_dir, match, clip_name, self._csv_filename )
                ball_xyvs = load_csv(clip_csv_path, self._visible_flags, frame_dir=clip_frame_dir)
                frame_names = []
                for frame_name in os.listdir(clip_frame_dir):
                    if frame_name.endswith(self._ext):
                        frame_names.append(frame_name)
                frame_names.sort()
                num_frames         += len(frame_names)
                num_frames_with_gt += len(ball_xyvs)
                
                if refine_npz_path is not None:
                    ball_xyvs = refine_gt_clip(ball_xyvs, clip_frame_dir, frame_names, refine_npz_path)

                for i in range(len(ball_xyvs)-self._frames_in+1):
                    names = frame_names[i:i+self._frames_in]
                    paths = [ osp.join(clip_frame_dir, name) for name in names]
                    annos = [ ball_xyvs[j] for j in range(i+self._frames_in-self._frames_out, i+self._frames_in)]
                    seq_list.append( {'frames': paths, 'annos': annos, 'match': match, 'clip': clip_name})
                    if i%self._step==0:
                        clip_seq_list.append( {'frames': paths, 'annos': annos, 'match': match, 'clip': clip_name})
                
                clip_disps = []
                # compute displacement between consecutive frames
                for i in range(len(ball_xyvs)-1):
                    xy1, visi1 = ball_xyvs[i]['center'].xy, ball_xyvs[i]['center'].is_visible
                    xy2, visi2 = ball_xyvs[i+1]['center'].xy, ball_xyvs[i+1]['center'].is_visible
                    if visi1 and visi2:
                        disp = np.linalg.norm(np.array(xy1)-np.array(xy2))
                        disps.append(disp)
                        clip_disps.append(disp)

                for i in range(len(ball_xyvs)):
                    path     = osp.join(clip_frame_dir, frame_names[i])
                    clip_seq_gt_dict[path] = ball_xyvs[i]['center']

                clip_seq_list_dict[(match, clip_name)]    = clip_seq_list
                clip_seq_gt_dict_dict[(match, clip_name)] = clip_seq_gt_dict
                clip_seq_disps[(match, clip_name)]         = clip_disps

        return { 'seq_list': seq_list, 
                 'clip_seq_list_dict': clip_seq_list_dict, 
                 'clip_seq_gt_dict_dict': clip_seq_gt_dict_dict,
                 'clip_seq_disps': clip_seq_disps,
                 'num_frames': num_frames, 
                 'num_frames_with_gt': num_frames_with_gt, 
                 'num_matches': num_matches, 
                 'num_rallies': num_rallies,
                 'disp_mean': np.mean(np.array(disps)),
                 'disp_std': np.std(np.array(disps))}

    @property
    def train(self):
        return self._train_all

    @property
    def test(self):
        return self._test_all

    @property
    def train_clips(self):
        return self._train_clips

    @property
    def train_clip_gts(self):
        return self._train_clip_gts

    @property
    def test_clips(self):
        return self._test_clips

    @property
    def test_clip_gts(self):
        return self._test_clip_gts

