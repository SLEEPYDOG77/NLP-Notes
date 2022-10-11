import warnings
from pprint import pprint

class DefaultConfig(object):
    model = 'CharRNN'

    # Dataset.
    # txt = './dataset/poetry.txt'
    txt = '../0data/chinese_poetry_data/train.txt'
    len = 20
    max_vocab = 8000
    # begin = '天青色等烟雨'  # begin word of text
    begin = '昔人何必重悲傷'
    predict_len = 50  # predict length

    # Store result and save models.
    result_file = './experiment/result.txt'
    save_file = './experiment/checkpoints/'
    # save model every N epochs
    save_freq = 5
    save_best = True

    # Predict mode and generate contexts
    load_model = './experiment/checkpoints/CharRNN_best_model.pth'
    write_file = './experiment/write_context.txt'

    # Visualization parameters.
    vis_dir = './experiment/vis/'
    plot_freq = 100  # plot in tensorboard every N iterations

    # Model parameters.
    embed_dim = 512
    hidden_size = 512
    num_layers = 2
    dropout = 0.5

    # Model hyperparameters.
    # use GPU or not
    use_gpu = True
    # running on which cuda device
    ctx = 0
    # batch size
    batch_size = 128
    # how many workers for loading 0data
    num_workers = 4
    max_epoch = 30
    # initial learning rate
    lr = 1e-3
    weight_decay = 1e-4

    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('=========user config==========')
        pprint(self._state_dict())
        print('============end===============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in DefaultConfig.__dict__.items()
                if not k.startswith('_')}


opt = DefaultConfig()
