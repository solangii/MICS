import abc
from data.dataloader.data_utils import *
from utils.utils import *
from tqdm import tqdm
import torch.nn.functional as F


class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args
        self.args = set_up_datasets(self.args)
        self.timer = Timer()

        # train statistics
        self.trlog = {}
        self.trlog['train_loss'] = []
        self.trlog['val_loss'] = []
        self.trlog['test_loss'] = []
        self.trlog['proto_loss'] = []
        self.trlog['optimal_loss'] = []
        self.trlog['train_acc'] = []
        self.trlog['val_acc'] = []
        self.trlog['test_acc'] = []
        self.trlog['max_acc_epoch'] = 0
        self.trlog['max_acc'] = [0.0] * args.sessions
        self.trlog['r2m_val_loss'] = np.zeros([args.sessions])
        self.trlog['r2m_proto_loss'] = np.zeros([args.sessions])

    @abc.abstractmethod
    def train(self):
        pass


def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    model = model.train()
    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]

        logits = model(data)
        logits = logits[:, :args.base_class]
        loss = F.cross_entropy(logits, train_label)
        acc = count_acc(logits, train_label)

        total_loss = loss

        lrc = scheduler.get_last_lr()[0]
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    return tl, ta


def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()
    mode = model.module.mode

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_mean = embedding_this.mean(0)
        proto_list.append(embedding_mean)

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc.weight.data[:args.base_class] = proto_list
    model.module.mode = mode
    return model


def test(model, testloader, epoch, args, session):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    pred = []
    label = []

    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            logits = model(data)
            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label)

            vl.add(loss.item())
            pred.extend(logits)
            label.extend(test_label)

        vl = vl.item()
        pred = torch.stack(pred, 0)
        label = torch.stack(label, 0)
        va = count_acc(pred, label)
    print('Epoch {} / Test loss {:.3f} / Test acc {:.2f}%'.format(epoch, vl, va*100))

    return vl, va
