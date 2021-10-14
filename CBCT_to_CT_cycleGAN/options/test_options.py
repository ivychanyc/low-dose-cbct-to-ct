from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        #arser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.') #Ivy edited on 23.06.2021
        parser.add_argument('--results_dir', type=str, default='/data/Unpaired_MR_to_CT_Image_Synthesis-master/results/', help='saves results here.') 
        #parser.add_argument('--results_dir', type=str, default='/data/Unpaired_MR_to_CT_Image_Synthesis-master/results/16bit/', help='saves results here.') 
        #parser.add_argument('--results_dir', type=str, default='/data/Unpaired_MR_to_CT_Image_Synthesis-master/results/train/', help='saves results here.')  #for run on training data
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        #parser.add_argument('--which_epoch', type=str, default='180', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=544, help='how many test images to run')
        #parser.add_argument('--how_many', type=int, default=204, help='how many test images to run') #just 3 

        parser.set_defaults(model='test')
        # To avoid cropping, the loadSize should be the same as fineSize
        parser.set_defaults(loadSize=parser.get_default('fineSize'))
        
        self.isTrain = False
        return parser
