from src.dataset.rams import RamsDataset
from src.dataset.cnn import CnnDataset
from src.dataset.anli import AnliDataset
from src.dataset.logiqa import LogiqaDataset
from src.dataset.paws import PawsDataset
from src.dataset.snli import SnliDataset
from src.dataset.agnews import AgnewsDataset
from src.dataset.sst import SstDataset
from src.dataset.pubmed import PubmedDataset
from src.dataset.wic import WicDataset
from src.dataset.spider import SpiderDataset
from src.dataset.wmt import WmtDataset
from src.dataset.conll import ConllDataset
from path import DATA_HOME_PATH as homepath

mix_amr_text_value = True


class load_rams():
    def __init__(self, RamsDataset):
        self.RamsDataset = RamsDataset
    def load(self, prompt_type):
        train_dataset = self.RamsDataset(f'{homepath}/rams/rams_processed_train.jsonl', f'{homepath}/rams/system_message.json', prompt_type, mix_amr_text = mix_amr_text_value)
        val_dataset   = self.RamsDataset(f'{homepath}/rams/rams_processed_dev.jsonl', f'{homepath}/rams/system_message.json', prompt_type, mix_amr_text = mix_amr_text_value)
        test_dataset  = self.RamsDataset(f'{homepath}/rams/rams_processed_test.jsonl', f'{homepath}/rams/system_message.json', prompt_type, mix_amr_text = mix_amr_text_value)
        return train_dataset, val_dataset, test_dataset


class load_anli():
    def __init__(self, AnliDataset):
        self.AnliDataset = AnliDataset
    def load(self, prompt_type):
        train_dataset = self.AnliDataset(f'{homepath}/anli/anli_train.jsonl', f'{homepath}/anli/system_message.json', prompt_type, mix_amr_text = mix_amr_text_value)
        val_dataset   = self.AnliDataset(f'{homepath}/anli/anli_dev.jsonl',   f'{homepath}/anli/system_message.json', prompt_type, mix_amr_text = mix_amr_text_value)
        test_dataset  = self.AnliDataset(f'{homepath}/anli/anli_test.jsonl',  f'{homepath}/anli/system_message.json', prompt_type, mix_amr_text = mix_amr_text_value)
        return train_dataset, val_dataset, test_dataset

class load_logiqa():
    def __init__(self, LogiqaDataset):
        self.LogiqaDataset = LogiqaDataset
    def load(self, prompt_type):
        train_dataset = self.LogiqaDataset(f'{homepath}/logiqa/logiqa_train.jsonl', f'{homepath}/logiqa/system_message.json', prompt_type, mix_amr_text = mix_amr_text_value)
        val_dataset   = self.LogiqaDataset(f'{homepath}/logiqa/logiqa_dev.jsonl',        f'{homepath}/logiqa/system_message.json', prompt_type, mix_amr_text = mix_amr_text_value)
        test_dataset  = self.LogiqaDataset(f'{homepath}/logiqa/logiqa_test.jsonl',       f'{homepath}/logiqa/system_message.json', prompt_type, mix_amr_text = mix_amr_text_value)
        return train_dataset, val_dataset, test_dataset

class load_cnn():
    def __init__(self, CnnDataset):
        self.CnnDataset = CnnDataset
    def load(self, prompt_type):
        train_dataset = self.CnnDataset(f"{homepath}/cnn/cnn_train.jsonl", f"{homepath}/cnn/system_message.json", prompt_type, mix_amr_text = mix_amr_text_value)
        val_dataset   = self.CnnDataset(f"{homepath}/cnn/cnn_dev.jsonl",   f"{homepath}/cnn/system_message.json", prompt_type, mix_amr_text = mix_amr_text_value)
        test_dataset  = self.CnnDataset(f"{homepath}/cnn/cnn_test.jsonl",  f"{homepath}/cnn/system_message.json", prompt_type, mix_amr_text = mix_amr_text_value)
        return train_dataset, val_dataset, test_dataset


class load_snli():
    def __init__(self, SnliDataset):
        self.SnliDataset = SnliDataset
    def load(self, prompt_type):
        train_dataset = self.SnliDataset(f'{homepath}/snli/snli_train_parsed.csv', f'{homepath}/snli/system_message.json', prompt_type, mix_amr_text=mix_amr_text_value)
        val_dataset   = self.SnliDataset(f'{homepath}/snli/snli_dev_parsed.csv',   f'{homepath}/snli/system_message.json', prompt_type, mix_amr_text=mix_amr_text_value)
        test_dataset  = self.SnliDataset(f'{homepath}/snli/snli_test_parsed.csv',  f'{homepath}/snli/system_message.json', prompt_type, mix_amr_text=mix_amr_text_value)
        return train_dataset, val_dataset, test_dataset

class load_paws():
    def __init__(self, PawsDataset):
        self.PawsDataset = PawsDataset
    def load(self, prompt_type):
        train_dataset = self.PawsDataset(f'{homepath}/paws/paws_train_parsed.csv', f'{homepath}/paws/system_message.json', prompt_type, mix_amr_text=mix_amr_text_value)
        val_dataset   = self.PawsDataset(f'{homepath}/paws/paws_dev_parsed.csv',   f'{homepath}/paws/system_message.json', prompt_type, mix_amr_text=mix_amr_text_value)
        test_dataset  = self.PawsDataset(f'{homepath}/paws/paws_test_parsed.csv',  f'{homepath}/paws/system_message.json', prompt_type, mix_amr_text=mix_amr_text_value)
        return train_dataset, val_dataset, test_dataset

class load_paws_nld():
    def __init__(self, PawsDataset):
        self.PawsDataset = PawsDataset
    def load(self, prompt_type):
        train_dataset = self.PawsDataset(f'{homepath}/paws/paws_train_parsed.csv', f'{homepath}/paws/system_message.json', prompt_type,mix_amr_text=mix_amr_text_value)
        val_dataset   = self.PawsDataset(f'{homepath}/paws/paws_dev_parsed.csv',   f'{homepath}/paws/system_message.json', prompt_type, mix_amr_text=mix_amr_text_value)
        test_dataset  = self.PawsDataset(f'{homepath}/paws/paws_test_parsed.csv',  f'{homepath}/paws/system_message.json', prompt_type, nld_path=f'{homepath}/paws/nld.jsonl', mix_amr_text=mix_amr_text_value)
        return train_dataset, val_dataset, test_dataset

class load_agnews():
    def __init__(self, AgnewsDataset):
        self.AgnewsDataset = AgnewsDataset
    def load(self, prompt_type):
        train_dataset = self.AgnewsDataset(f'{homepath}/agnews/agnews_train_parsed.csv', f'{homepath}/agnews/system_message.json', prompt_type, mix_amr_text=mix_amr_text_value)
        val_dataset   = self.AgnewsDataset(f'{homepath}/agnews/agnews_dev_parsed.csv',   f'{homepath}/agnews/system_message.json', prompt_type, mix_amr_text=mix_amr_text_value)
        test_dataset  = self.AgnewsDataset(f'{homepath}/agnews/agnews_test_parsed.csv',  f'{homepath}/agnews/system_message.json', prompt_type, mix_amr_text=mix_amr_text_value)
        return train_dataset, val_dataset, test_dataset

class load_sst():
    def __init__(self, SstDataset):
        self.SstDataset = SstDataset
    def load(self, prompt_type):
        train_dataset = self.SstDataset(f'{homepath}/sst/sst_train_parsed.csv', f'{homepath}/sst/system_message.json', prompt_type, mix_amr_text=mix_amr_text_value)
        val_dataset   = self.SstDataset(f'{homepath}/sst/sst_dev_parsed.csv',   f'{homepath}/sst/system_message.json', prompt_type, mix_amr_text=mix_amr_text_value)
        test_dataset  = self.SstDataset(f'{homepath}/sst/sst_test_parsed.csv',  f'{homepath}/sst/system_message.json', prompt_type, mix_amr_text=mix_amr_text_value)
        return train_dataset, val_dataset, test_dataset

class load_pubmed():
    def __init__(self, PubmedDataset):
        self.PubmedDataset = PubmedDataset
    def load(self, prompt_type):
        train_dataset = self.PubmedDataset(f'{homepath}/pubmed/pubmed_train_parsed.csv', f'{homepath}/pubmed/system_message.json', prompt_type, mix_amr_text=mix_amr_text_value)
        val_dataset   = self.PubmedDataset(f'{homepath}/pubmed/pubmed_dev_parsed.csv',   f'{homepath}/pubmed/system_message.json', prompt_type, mix_amr_text=mix_amr_text_value)
        test_dataset  = self.PubmedDataset(f'{homepath}/pubmed/pubmed_test_parsed.csv',  f'{homepath}/pubmed/system_message.json', prompt_type, mix_amr_text=mix_amr_text_value)
        return train_dataset, val_dataset, test_dataset

class load_wic():
    def __init__(self, WicDataset):
        self.WicDataset = WicDataset
    def load(self, prompt_type):
        train_dataset = self.WicDataset(f'{homepath}/wic/wic_train_parsed.csv', f'{homepath}/wic/system_message.json', prompt_type, mix_amr_text=mix_amr_text_value)
        val_dataset   = self.WicDataset(f'{homepath}/wic/wic_dev_parsed.csv',   f'{homepath}/wic/system_message.json', prompt_type, mix_amr_text=mix_amr_text_value)
        test_dataset  = self.WicDataset(f'{homepath}/wic/wic_test_parsed.csv',  f'{homepath}/wic/system_message.json', prompt_type, mix_amr_text=mix_amr_text_value)
        return train_dataset, val_dataset, test_dataset

class load_spider():
    def __init__(self, SpiderDataset):
        self.SpiderDataset = SpiderDataset
    def load(self, prompt_type):
        train_dataset = self.SpiderDataset(f'{homepath}/spider/spider_train_parsed.csv', f'{homepath}/spider/system_message.json', prompt_type, mix_amr_text=mix_amr_text_value)
        val_dataset   = self.SpiderDataset(f'{homepath}/spider/spider_dev_parsed.csv',   f'{homepath}/spider/system_message.json', prompt_type, mix_amr_text=mix_amr_text_value)
        test_dataset  = self.SpiderDataset(f'{homepath}/spider/spider_test_parsed.csv',  f'{homepath}/spider/system_message.json', prompt_type, mix_amr_text=mix_amr_text_value)
        return train_dataset, val_dataset, test_dataset

class load_conll():
    def __init__(self, ConllDataset):
        self.ConllDataset = ConllDataset
    def load(self, prompt_type):
        train_dataset = self.ConllDataset(f'{homepath}/conll/conll_train_parsed.csv', f'{homepath}/conll/system_message.json', prompt_type, mix_amr_text=mix_amr_text_value)
        val_dataset   = self.ConllDataset(f'{homepath}/conll/conll_dev_parsed.csv',   f'{homepath}/conll/system_message.json', prompt_type, mix_amr_text=mix_amr_text_value)
        test_dataset  = self.ConllDataset(f'{homepath}/conll/conll_test_parsed.csv',  f'{homepath}/conll/system_message.json', prompt_type, mix_amr_text=mix_amr_text_value)
        return train_dataset, val_dataset, test_dataset

class load_wmt():
    def __init__(self, WmtDataset):
        self.WmtDataset = WmtDataset
    def load(self, prompt_type):
        train_dataset = self.WmtDataset(f'{homepath}/wmt/wmt_train_parsed.csv', f'{homepath}/wmt/system_message.json', prompt_type, mix_amr_text=mix_amr_text_value)
        val_dataset   = self.WmtDataset(f'{homepath}/wmt/wmt_dev_parsed.csv',   f'{homepath}/wmt/system_message.json', prompt_type, mix_amr_text=mix_amr_text_value)
        test_dataset  = self.WmtDataset(f'{homepath}/wmt/wmt_test_parsed.csv',  f'{homepath}/wmt/system_message.json', prompt_type, mix_amr_text=mix_amr_text_value)
        return train_dataset, val_dataset, test_dataset


load_dataset = {
    'rams': load_rams(RamsDataset),
    'cnn': load_cnn(CnnDataset),
    'anli': load_anli(AnliDataset),
    'logiqa': load_logiqa(LogiqaDataset),
    'agnews': load_agnews(AgnewsDataset),
    'snli': load_snli(SnliDataset),
    'paws': load_paws(PawsDataset),
    'paws_nld': load_paws_nld(PawsDataset),
    'sst': load_sst(SstDataset),
    'pubmed': load_pubmed(PubmedDataset),
    'wic': load_wic(WicDataset),
    'wmt': load_wmt(WmtDataset),
    'spider': load_spider(SpiderDataset),
    'conll': load_conll(ConllDataset),
}