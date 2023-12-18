import os
from fastda.utils.train_api import train
from fastda.utils.arg_parser import fastda_arg_parser
from robustkd.models import *
from robustkd.loaders import *
from robustkd.trainers import *

if __name__ == '__main__':
    project_root = os.getcwd()
    package_name = 'robustkd'
    args = fastda_arg_parser(project_root, package_name)
    # debug
    args.trainer = 'sfda_shot'
    args.validator = 'sfda_shot'
    args.config = './configs/sfda_nrc/sfda_officehome_AC_target_IM_2init2_500_fix_classifier.py'
    # args.config = './configs/hda_srcmix/hda_srcmix_contrsative_officehome_AP_hda_fixmatch_test.py'
    # # args.config = './configs/hda_srcmix/hda_srcmix_contrsative_officehome_AP_hda_fixmatch_mixlrco_test.py'
    #####
    # args.trainer = 'fixmatchgvbsrcmix'
    # args.validator = 'fixmatchgvbsrcmix'
    # args.config = 'configs/gvb_srcmix/gvb_srcmix_contrsative_officehome_AP_fixmatch_test.py'
    # args.config = 'configs/gvb_srcmix/gvb_srcmix_contrsative_officehome_AP_mix_0.2_test.py'
    # args.config = 'configs/gvb_srcmix/gvb_srcmix_contrsative_officehome_CR_fixmatch_test.py'
    # args.config = 'configs/gvb_srcmix/gvb_srcmix_contrsative_officehome_CR_mix_0.2_test.py'
    args.job_id = 'debug'
    train(args)
