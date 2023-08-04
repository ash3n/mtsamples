import pandas as pd
import torch
import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased")
def tokenize(txt, max_length=512, return_tensors=None):
    return tokenizer(
        txt, return_tensors=return_tensors,
        truncation=True, padding='max_length', max_length=max_length,
    )

class ClfDataset(torch.utils.data.Dataset):
    def __init__(self, data, classes):
        super().__init__()
        self.data = data
        self.classes = classes
        self.label2id = {v:i for i, v in enumerate(classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, return_tensors=None):
        input_string, category = self.data[idx]
        labels = torch.tensor(self.label2id[category]) if return_tensors=='pt' else [self.label2id[category]]
        input_ids = tokenize(input_string, return_tensors=return_tensors).input_ids
        return {
            'input_ids' : input_ids,
            'labels' : labels,
        }

def get_mtsamples_dataset(path):
    df = pd.read_csv(path)

    not_a_specialty = [
        ' SOAP / Chart / Progress Notes',
        ' Office Notes',
        'Letters',
        ' IME-QME-Work Comp etc.',
        'Emergency Room Reports',
        ' Discharge Summary',
        ' Consult - History and Phy.'
    ]

    df = df[df['medical_specialty'].apply(lambda x: type(x) == str and not (x in not_a_specialty))]
    df = df[df['keywords'].apply(lambda x: type(x) == str)]
    df = df.sample(frac=1).reset_index(drop=True)
    raw_data = []
    for i in range(len(df)):
        try:
            transcription = ', '.join(df['keywords'][i].split(', ')[1:])
            specialty = df['medical_specialty'][i]
            raw_data.append((transcription, specialty))
        except:
            pass

    print('Dataset size:', len(raw_data))
    train_split = int(0.8 * len(raw_data))
    test_split = int(0.1 * len(raw_data))
    raw_train = raw_data[:train_split]
    raw_val = raw_data[train_split:-test_split]
    raw_test = raw_data[-test_split:]
    classes = list(df['medical_specialty'].unique())

    return (
        ClfDataset(raw_train, classes),
        ClfDataset(raw_val, classes),
        ClfDataset(raw_test, classes),
    )