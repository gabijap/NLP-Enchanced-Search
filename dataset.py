import torch
from gensim.utils import tokenize
from torch.utils.data.dataset import Dataset


class Corpus(Dataset):
    """ Corpus dataset """

    def __init__(self, read_path, vocab, max_len=360):
        """
        :param read_path: Path to the txt file with sentences.
        :param vocab: Pretrained word vector.
        :param max_len: Maximum sentence length.
        """
        self.vocab = vocab
        self.max_len = max_len
        with open(read_path, 'r') as file:
            self.sentences = list(file)
        file.close()
        print("Completed reading ", len(self.sentences), " lines from ", read_path)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        return torch.LongTensor(
            [self.vocab[x].index for (x, _) in zip(tokenize(self.sentences[i]), range(self.max_len))])
