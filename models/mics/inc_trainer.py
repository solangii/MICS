import math
from copy import deepcopy
from models.mics.helper import *
from models.mics.inc_network import MYNET
from data.dataloader.data_utils import *
import torch.nn as nn

class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        if ('cos' in args.model_dir) or ('dot' in args.model_dir):
            self.args.full_fc = True
        self.set_save_path()
        self.args = set_up_datasets(self.args)
        self.set_up_model()
        self.set_acc_table(self.args)
        pass

    def set_save_path(self):
        mode = self.args.base_mode
        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/%s/' % (self.args.project, self.args.phase)
        self.args.save_path = self.args.save_path + '%s/' % mode
        if 'cos' in mode:
            self.args.save_path = self.args.save_path + '-T_%.2f' % (self.args.temperature)
        if 'ft' in self.args.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                self.args.lr_new, self.args.epochs_new)
        self.args.save_path = self.args.save_path + '-' + self.args.memo

        self.args.save_path = os.path.join('results', self.args.save_path)
        ensure_path(self.args.save_path)
        return None

    def set_up_model(self):
        self.model = MYNET(self.args, mode=self.args.base_mode)
        print(MYNET)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        if self.args.model_dir != None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = dict()

            try:
                temp_model_dict = torch.load(self.args.model_dir)['params']
                for key, value in temp_model_dict.items():
                    if 'dummy' not in key:
                        self.best_model_dict[key] = value
            except:
                temp_model_dict = torch.load(self.args.model_dir)['state_dict']
                for key, value in temp_model_dict.items():
                    if 'backbone' in key:
                        temp_key = 'module.encoder' + key.split('backbone')[1]
                        if 'shortcut' in temp_key:
                            temp_key = temp_key.replace('shortcut', 'downsample')
                        self.best_model_dict[temp_key] = value
            self.best_model_dict['module.fc.weight'] = self.model.module.fc.weight

        else:
            raise ValueError('You must initialize a pre-trained model')

    def set_acc_table(self, args):
        self.results = {}
        self.results["acc"] = np.zeros([args.sessions])
        self.results["acc_base"] = np.zeros([args.sessions])
        self.results["acc_novel"] = np.zeros([args.sessions])
        self.results["acc_old"] = np.zeros([args.sessions])
        self.results["acc_new"] = np.zeros([args.sessions])
        self.results["acc_base2"] = np.zeros([args.sessions])
        self.results["acc_novel2"] = np.zeros([args.sessions])
        self.results["acc_old2"] = np.zeros([args.sessions])
        self.results["acc_new2"] = np.zeros([args.sessions])

    def get_logits(self, x, fc):
        if 'cos' in self.args.new_mode:
            logit = self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))
        elif 'dot' in self.args.new_mode:
            logit = F.linear(x, fc)
        elif 'euclidean' in self.args.new_mode:
            w = torch.unsqueeze(fc, 0)
            q = torch.unsqueeze(x, 1)
            logit = -torch.sum(torch.square(q - w), [-1])
        else:
            assert False, 'Not supported metric'
        return logit

    def get_optimizer_new(self):
        optimizer = torch.optim.SGD([{'params': self.model.module.encoder.parameters(), 'lr': self.args.lr_new},
                                     {'params': self.model.module.fc.parameters(), 'lr': self.args.lr_cf}],
                                    momentum=0.9, nesterov=True, weight_decay=self.args.decay)
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_new)

        return optimizer, scheduler

    def get_session_trainable_param_idx(self, model):
        param_dict = dict(model.named_parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_st = math.floor(num_trainable_params * self.args.st_ratio)  # The number of session trainable parameters

        # Remove not trainable parameter from parameter dictionary
        remove_list = [k for k, v in param_dict.items() if not v.requires_grad]
        for k in remove_list:
            param_dict.pop(k)

        # Pre-procession parameter data
        param_flatten_dict = {}
        param_len_dict = {}
        for k, v in param_dict.items():
            param_dict[k] = param_dict[k].data
            param_flatten_dict[k] = torch.reshape(param_dict[k].detach().data, [-1])
            param_len_dict[k] = param_flatten_dict[k].shape[0]

        # Select the unimportant weight (small absolute value)
        all_weight = np.abs(np.array(sum([list(v.cpu().numpy()) for v in param_flatten_dict.values()], [])))
        sel_weight_idx = all_weight.argsort()[:num_st]

        # Get indices of selected parameters
        count = 0
        param_idx_dict = {}
        for k, v in param_len_dict.items():
            param_flattened_idx_list = [w_idx - count for w_idx in sel_weight_idx
                                        if (w_idx >= count) & (w_idx < count + v)]
            param_idx_dict[k] = [np.unravel_index(flattened_idx, param_dict[k].shape)
                                 for flattened_idx in param_flattened_idx_list]
            count += v

        return param_idx_dict

    def update_param(self, model, pretrained_dict):
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        if self.args.st_ratio < 1:
            P_st_idx = self.get_session_trainable_param_idx(model)
            return model, P_st_idx
        else:
            return model, None

    def train(self):
        args = self.args
        self.model.load_state_dict(self.best_model_dict)
        print([{name: param.requires_grad} for name, param in self.model.named_parameters()])
        self.model.module.mode = self.args.new_mode

        optimizer, scheduler = self.get_optimizer_new()

        for session in range(0, args.sessions):

            print("\nSession: [%d]" % (session + 1))
            train_set, trainloader, testloader = get_dataloader(args, session)

            if session != 0:
                train_transform = deepcopy(trainloader.dataset.transform)
                trainloader.dataset.transform = testloader.dataset.transform
                self.model = self.update_novel_proto(self.model, trainloader, np.unique(train_set.targets))

                # Incremental MICS
                trainloader.dataset.transform = train_transform
                for epoch in range(args.epochs_new):
                    self.model, P_st_idx = self.update_param(self.model, self.best_model_dict)
                    tl, ta = self.new_train(self.model, trainloader, optimizer, scheduler, epoch, args, P_st_idx, session)

                trainloader.dataset.transform = testloader.dataset.transform
                self.model = self.update_novel_proto(self.model, trainloader, np.unique(train_set.targets))
            else:
                self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)

            # Evaluation & Save
            tsl, tsa = self.test(self.model, testloader, args, session)
            self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
            save_model_dir = os.path.join(args.save_path, 'session' + str(session + 1) + '_max_acc.pth')
            torch.save(dict(params=self.model.state_dict()), save_model_dir)
            self.best_model_dict = deepcopy(self.model.state_dict())
            print('Test acc = {:.2f}%'.format(self.trlog['max_acc'][session]))

        args.save_path = os.path.join(args.save_path, 'last_session.pth')
        print(f"Save last session model to {args.save_path}")
        torch.save(dict(params=self.model.state_dict()), args.save_path)

        print('\n*************** Final results ***************')
        print(self.trlog['max_acc'], '\n')
        self.print_table(self.results, self.args)

    def update_novel_proto(self, model, dataloader, class_list):
        model = model.eval()
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data = model.module.encode(data).detach()

        new_fc = []
        for class_index in class_list:
            data_index = (label == class_index).nonzero().squeeze(-1)
            embedding = data[data_index]
            proto = embedding.mean(0)
            new_fc.append(proto)
            model.module.fc.weight.data[class_index] = proto

        return model

    def test(self, model, testloader, args, session):
        test_class = args.base_class + session * args.way
        model = model.eval()
        vl = Averager()

        pred = []
        label = []

        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.cuda() for _ in batch]

                logits = model(data)
                logits = logits[:, :test_class]
                loss = F.cross_entropy(logits, test_label)

                vl.add(loss.item())
                pred.extend(logits)
                label.extend(test_label)

            vl = vl.item()

            # Incremental accuracy table
            pred = torch.stack(pred, 0)
            label = torch.stack(label, 0)

            # Ex. CUB: args.base_class + (session - 1) * args.way = 100 + (N-1) * 10
            pred_base = pred[:, :args.base_class + (session - 1) * args.way]
            pred_novel = pred[:, args.base_class + (session - 1) * args.way:]
            label_novel = label - (args.base_class + (session - 1) * args.way)

            # Ex. CUB: args.base_class = 100
            pred_old = pred[:, :args.base_class]
            pred_new = pred[:, args.base_class:]
            label_new = label - args.base_class

            self.results['acc'][session] = count_acc(pred, label)

            if session == 0:
                top1_base = count_acc(pred, label)
                top1_base2 = top1_base

                top1_novel = 0
                top1_novel2 = 0
            else:
                base_idx = label < args.base_class + (session - 1) * args.way
                novel_idx = label >= args.base_class + (session - 1) * args.way

                top1_base = count_acc(pred[base_idx], label[base_idx])
                top1_base2 = count_acc(pred_base[base_idx], label[base_idx])

                top1_novel = count_acc(pred[novel_idx], label[novel_idx])
                top1_novel2 = count_acc(pred_novel[novel_idx], label_novel[novel_idx])

            self.results['acc_base'][session] = top1_base
            self.results['acc_base2'][session] = top1_base2

            self.results['acc_novel'][session] = top1_novel
            self.results['acc_novel2'][session] = top1_novel2

            old_idx = label < args.base_class
            new_idx = label >= args.base_class

            top1_old = count_acc(pred[old_idx], label[old_idx])
            top1_old2 = count_acc(pred_old[old_idx], label[old_idx])

            if session == 0:
                top1_new = 0
                top1_new2 = 0
            else:
                top1_new = count_acc(pred[new_idx], label[new_idx])
                top1_new2 = count_acc(pred_new[new_idx], label_new[new_idx])

            self.results['acc_old'][session] = top1_old
            self.results['acc_old2'][session] = top1_old2

            self.results['acc_new'][session] = top1_new
            self.results['acc_new2'][session] = top1_new2

        return vl, self.results['acc'][session]

    def print_table(self, results, args):
        print("{:<13}{}".format("Pretraining:", args.base_mode.split('_')[-1]))
        print("{:<13}{}".format("Similarity:", args.new_mode.split('_')[-1]))

        str_head = "{:<9}".format('')
        str_acc = "{:<9}".format('Acc:')
        str_acc_base = "{:<9}".format('Base:')
        str_acc_novel = "{:<9}".format('Novel:')
        str_acc_old = "{:<9}".format('Old:')
        str_acc_new = "{:<9}".format('New:')

        for i in range(len(results['acc'])):
            str_head = str_head + "{:<9}".format('sess' + str(int(i + 1)))
            str_acc = str_acc + "{:<9}".format(str(round(results['acc'][i] * 100.0, 2)) + "%")
            str_acc_base = str_acc_base + "{:<9}".format(str(round(results['acc_base'][i] * 100.0, 2)) + "%")
            str_acc_novel = str_acc_novel + "{:<9}".format(str(round(results['acc_novel'][i] * 100.0, 2)) + "%")
            str_acc_old = str_acc_old + "{:<9}".format(str(round(results['acc_old'][i] * 100.0, 2)) + "%")
            str_acc_new = str_acc_new + "{:<9}".format(str(round(results['acc_new'][i] * 100.0, 2)) + "%")

        print(str_head)
        print(str_acc)
        print(str_acc_base)
        print(str_acc_novel)
        print(str_acc_old)
        print(str_acc_new)
        print('\n')

        str_acc_base2 = "{:<9}".format('Base2:')
        str_acc_novel2 = "{:<9}".format('Novel2:')
        str_acc_old2 = "{:<9}".format('Old2:')
        str_acc_new2 = "{:<9}".format('New2:')

        for i in range(len(results['acc'])):
            str_acc_base2 = str_acc_base2 + "{:<9}".format(str(round(results['acc_base2'][i] * 100.0, 2)) + "%")
            str_acc_novel2 = str_acc_novel2 + "{:<9}".format(str(round(results['acc_novel2'][i] * 100.0, 2)) + "%")
            str_acc_old2 = str_acc_old2 + "{:<9}".format(str(round(results['acc_old2'][i] * 100.0, 2)) + "%")
            str_acc_new2 = str_acc_new2 + "{:<9}".format(str(round(results['acc_new2'][i] * 100.0, 2)) + "%")

        print(str_acc_base2)
        print(str_acc_novel2)
        print(str_acc_old2)
        print(str_acc_new2)
        print('\n')

    def new_train(self, model, trainloader, optimizer, scheduler, epoch, args, P_st_idx=None, session=None):
        tl = Averager()
        ta = Averager()
        bce_loss = torch.nn.BCELoss().cuda()
        softmax = torch.nn.Softmax(dim=1).cuda()
        is_mix = "mix" in args.train
        train_class = args.base_class + session * args.way
        base_w = model.module.fc.weight[:train_class - args.way]

        model = model.train()
        tqdm_gen = tqdm(trainloader)

        for i, batch in enumerate(tqdm_gen, 1):
            num_loss, loss, acc = 0, 0., 0.
            data, label = [_.cuda() for _ in batch]

            output, retarget = model.module.forward_mix(args, Variable(data), Variable(label))

            if args.use_midpoint:
                if retarget.shape[1] > softmax(output).shape[1]:
                    retarget = retarget[:, :softmax(output).shape[1]]
                elif retarget.shape[1] < softmax(output).shape[1]:
                    output = output[:, :retarget.shape[1]]
                loss += bce_loss(softmax(output), retarget)
                acc += count_mix_acc(output, label)[0] * 0.01
            else:
                if args.use_mixup or args.use_softlabel:
                    classifier = model.module.fc
                    output = model.module.get_logits(output, classifier.weight)
                    if retarget.shape[1] > softmax(output).shape[1]:
                        retarget = retarget[:, :softmax(output).shape[1]]
                    elif retarget.shape[1] < softmax(output).shape[1]:
                        output = output[:, :retarget.shape[1]]

                    loss += bce_loss(softmax(output), retarget)
                    acc += count_mix_acc(output, label)[0] * 0.01
                else:
                    classifier = model.module.fc
                    output = model.module.get_logits(output, classifier.weight)
                    loss += F.cross_entropy(output, label)

            lrc = scheduler.get_last_lr()[0]
            tqdm_gen.set_description('Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'
                                     .format(epoch, lrc, loss.item(), acc))
            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.args.st_ratio < 1:
                updated_param_dict = deepcopy(model.state_dict())  # parameters after update
                model.load_state_dict(self.best_model_dict)
                for k, v in dict(model.named_parameters()).items():
                    if v.requires_grad:
                        for idx in P_st_idx[k]:
                            v.data[idx] = updated_param_dict[k].data[idx]

        tl = tl.item()
        ta = ta.item()
        return tl, ta
