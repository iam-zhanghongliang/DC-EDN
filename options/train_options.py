from .base_options import BaseOptions
class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--layer_num', type=int, default=1,
                                 help='number of coupled U-Nets')
        self.parser.add_argument('--order', type=int, default=0,
                                 help='order-K coupling')
        self.parser.add_argument('--class_num', type=int, default=16,
                                 help='number of classes in the prediction')
        self.parser.add_argument('--loss_num', type=int, default=1,
                                 help='number of losses in the CU-Net')
        self.parser.add_argument('--lr', type=float, default=2.5e-3,
                                 help='initial learning rate')
        self.parser.add_argument('--bs', type=int, default=16,
                                 help='mini-batch size')
        self.parser.add_argument('--load_checkpoint', type=bool, default=False,
                                 help='use checkpoint model')
        self.parser.add_argument('--adjust_lr', type=bool, default=False,
                                 help='adjust learning rate')
        self.parser.add_argument('--resume_prefix', type=str, default='',
                                 help='checkpoint name for resuming')
        self.parser.add_argument('--nEpochs', type=int, default=200,
                                 help='number of total training epochs to run')
        self.parser.add_argument('--best_pckh', type=float, default=0.,
                                 help='best result until now')
        self.parser.add_argument('--print_freq', type=int, default=10,
                                 help='print log every n iterations')
        self.parser.add_argument('--display_freq', type=int, default=10,
                                 help='display figures every n iterations')
     
	 
	 