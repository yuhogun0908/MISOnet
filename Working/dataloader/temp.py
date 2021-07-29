class AudioReader(object):
    '''
        Class that reads Wav format files
        Input:
            scp_path (str): a different scp file address
            sample_rate (int, optional): sample rate (default: 8000)
            chunk_size (int, optional): split audio size (default: 32000(4 s))
            least_size (int, optional): Minimum split size (default: 16000(2 s))
        Output:
            split audio (list)
    '''
    def __init__(self, scp_path, sample_rate=8000, chunk_size=32000, least_size=16000):
        super(AudioReader, self).__init__()
        self.sample_rate = sample_rate
        self.index_dict = handle_scp(scp_path)
        self.keys = list(self.index_dict.keys())
        self.audio = []
        self.chunk_size = chunk_size
        self.least_size = least_size
        self.split()
    def split(self):
        '''
            split audio with chunk_size and least_size
        '''
        print(len(self.keys))
        i=0
        for key in self.keys:
            i+=1
            print(i)
            utt = read_wav(self.index_dict[key])
            if utt.shape[0] < self.least_size:
                continue
            if utt.shape[0] > self.least_size and utt.shape[0] < self.chunk_size:
                gap = self.chunk_size-utt.shape[0]
                self.audio.append(F.pad(utt, (0, gap), mode='constant'))
            if utt.shape[0] >= self.chunk_size:
                start = 0
                while True:
                    if start + self.chunk_size > utt.shape[0]:
                        break
                    self.audio.append(utt[start:start+self.chunk_size])
                    start += self.least_size