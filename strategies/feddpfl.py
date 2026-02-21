from typing import List, Tuple
from colorama import Fore, Style
from .fedavg import FedAvg
import numpy as np
import math
import copy
from tqdm import tqdm
import torch
from torch import nn
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F
from functools import reduce
from torch.utils.data import DataLoader, ConcatDataset
from models.model import CGenerator
from utils.loss_fn import DiversityLoss
np.set_printoptions(suppress=True, linewidth=np.inf)

'''
Hyperparameters:
    - dataset: MNIST, CIFAR-10, Office-Caltech
'''

class FedDPFL(FedAvg):
    def __init__(self, fl_strategy, device, args, params):
        super().__init__(device=device, args=args, params=params)
        self.fl_strategy = fl_strategy
        ''' Parameters for data-free knowledge distillation '''
        self.z_dim = 100
        self.cgan = CGenerator(nz=self.z_dim, ngf=16, img_size=32, n_cls=args.num_classes).to(device)
        self.optimizer_cgan = torch.optim.Adam(self.cgan.parameters(), lr=3e-4, weight_decay=1e-2)
        # self.scheduler_cgan = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_cgan, gamma=0.98)

        ''' Hyperparameters for knowledge distillation'''
        self.batch_size = args.batch_size
        self.gen_batch_size = 128                           # int(params['gen_batch_size'])
        self.iterations = 1                                 # int(params['iterations'])
        self.inner_round_g = int(params['inner_round_g'])
        self.inner_round_d = int(params['inner_round_d'])
        self.T = params['con_T']

        ''' Coefficients for individual loss '''
        self.ensemble_gamma = params['kd_gamma']
        self.ensemble_beta = params['kd_beta']
        self.ensemble_eta = params['kd_eta']
        self.age_ld = 1                                     # params['age_ld']
        self.impt_ld = 1                                    # params['impt_ld']
        self.online_rate = params['online_rate']
        self.offline_rate = params['offline_rate']
        
        ''' Parameters for client data statistics '''
        self.num_classes = args.num_classes
        self.label_weights = []
        self.qualified_labels = []
    
        self.criterion_diversity = DiversityLoss(metric='l1').to(device)
        self.cosine_sim = torch.nn.CosineSimilarity(dim=-1)
        self.valLoader = None
        self.params = params

        '''Knowledge Pool 
            - {
                signature (data_classes): {
                    "model": model,
                    "data_summary": {
                        "label": num_data
                    },
                    "num_data": len(data),
                    "online_age": rounds,
                    "offline_age": rounds,
                    "layer": "Active", # HAKP Layer
                    "prototype": tensor, # HAKP Prototype
                }
            }
        '''
        self.knowledge_pool = {}

        ''' HAKP Parameters '''
        self.hakp_alpha = 0.7
        self.hakp_beta = 0.3
        self.hakp_T = 10  # Participation window
        self.hakp_R = 5   # Re-evaluation period
        self.hakp_delta = 0.05 # Hysteresis
        # Decay coefficients (lambda)
        self.hakp_lambda = {"Core": -0.001, "Active": -0.1, "Edge": -0.5}
        # Layer weight factors (gamma)
        self.hakp_gamma = {"Core": 1.5, "Active": 1.0, "Edge": 0.5}
        
        self.participation_history = {} # {cid: [round_participated, ...]}
        self.client_layers = {} # {cid: "Active"} (default)
        self.layer_counters = {} # {cid: counter} (for consecutive observation)
        self.cvs_history = {} # {cid: cvs_score}


    def initialization(self, global_model, client_list, testLoader):
        ''' Create lr_scheduler for global model '''
        self.optimizer_server = torch.optim.SGD(global_model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.scheduler_server = torch.optim.lr_scheduler.StepLR(self.optimizer_server, step_size=1, gamma=0.998)

        self.testLoader = copy.deepcopy(testLoader)

        ''' Get client's validation data '''
        valset = [client_list[i].valLoader.dataset for i in range(len(client_list))]
        self.valLoader = {
            "total_val": DataLoader(ConcatDataset(valset), batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        }
        print(Fore.YELLOW + "\n[Server] Initializing the knowledge pool..." + Fore.RESET)
        print(Fore.YELLOW + "Optimized Parameters for DPFL: {}".format(self.params) + Fore.RESET)
        
        # Initialize participation history and layers
        for client in client_list:
            self.participation_history[client.cid] = []
            self.client_layers[client.cid] = "Active"
            self.layer_counters[client.cid] = 0


    def data_free_knowledge_distillation(self, global_model, payloads, rounds):
        generator = self.cgan
        self.label_weights, self.qualified_labels = self.get_label_weights_from_knowledge_pool(self.knowledge_pool)
        age_weights = self.get_age_weight()

        ''' Integrate the label weight and age weight '''
        self.label_weights = self.combine_weights(self.label_weights, age_weights * self.age_ld)
        total_label_weights = np.sum(self.label_weights, axis=1)                    # get each label's total weight from clients in knowledge pool

        intial_val_acc = self.evaluate(global_model, self.valLoader, payloads)      # inital val_acc
        state = {
            "best_val_acc": intial_val_acc,
            "best_server_model": copy.deepcopy(global_model.state_dict()),
            "best_generator": copy.deepcopy(generator.state_dict()),
        }
        intial_test_acc = self.evaluate(global_model, self.testLoader, payloads)      # inital test_acc
        print("[Server | E. Distillation] Start ensemble distillation, inital test_acc: {:.4f}".format(intial_test_acc))
        
        pbar = tqdm(range(self.iterations), desc="[Server | E. Knowledge Distillation]", leave=False)
        for _ in pbar:
            ''' Train Generator '''
            generator.train()
            global_model.eval()

            loss_G_total = []
            loss_KD_total = []
            loss_con_total = []
            loss_cls_total = []
            loss_div_total = []

            # y = np.random.choice(self.qualified_labels, self.batch_size)
            y = self.generate_labels(self.batch_size, total_label_weights)
            y_input = F.one_hot(torch.tensor(y), num_classes=self.args.num_classes).type(torch.float32).to(self.device)

            for _ in range(self.inner_round_g):
                ''' feed to generator '''
                z = torch.randn((self.batch_size, self.z_dim), device=self.device)
                gen_output = generator(z, y_input)

                ''' get the student logit '''
                _, student_feature = global_model(gen_output)

                ''' compute diversity loss '''
                loss_div = self.criterion_diversity(z, gen_output)

                loss_con = 0
                loss_cls = 0
                loss_proto = 0
                
                # Check if we have any Core/Active clients with models to compute loss_con/loss_cls
                # If all clients are Edge (model=None), we skip loss_con and loss_cls calculation
                has_model_teachers = any(val["model"] is not None for val in self.knowledge_pool.values())
                
                if has_model_teachers:
                    ''' Train the generator using contrastive learning to separate the features (class separation) '''
                    # create the class feature
                    class_features = [[] for _ in range(self.num_classes)]
                    for i in range(self.batch_size):
                        class_features[y[i]].append(student_feature[i])
                    class_features = [torch.stack(class_feature) if len(class_feature) > 0 else torch.zeros((1, student_feature.shape[1]), device=self.device) for class_feature in class_features]


                    ''' create positive pairs and negative pairs of each class '''
                    features_pos, features_neg = [[] for _ in range(self.num_classes)], [[] for _ in range(self.num_classes)]
                    for i in range(self.num_classes):
                        features_pos[i] = torch.mean(class_features[i], dim=0).view(1, -1)
                        features_neg[i] = torch.stack([torch.mean(class_features[j], dim=0) for j in range(self.num_classes) if j != i])


                    # create the positive and negative pairs for each data
                    pos_pairs, neg_pairs = [], []
                    for k in range(self.batch_size):
                        pos_pairs.append(features_pos[y[k]])
                        neg_pairs.append(features_neg[y[k]])

                    pos_pairs = torch.stack(pos_pairs)
                    neg_pairs = torch.stack(neg_pairs)

                    pos_sim = self.cosine_sim(student_feature.unsqueeze(1), pos_pairs)
                    neg_sim = self.cosine_sim(student_feature.unsqueeze(1), neg_pairs)

                    logits = torch.cat([pos_sim, neg_sim], dim=1).to(self.device)
                    logits /= self.T

                    target = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)
                    loss_con = F.cross_entropy(logits, target)
 

                for idx, (sig, value) in enumerate(self.knowledge_pool.items()):
                    # HAKP: Handle Edge Clients (Prototype Loss)
                    if value["model"] is None:
                         # Calculate Prototype Loss if we have prototypes for this client
                         prototypes = value.get("prototype", {})
                         if not prototypes:
                             continue
                             
                         # Check if we have prototypes for the generated classes
                         # y is the batch of labels
                         # student_feature is the batch of student features for generated images
                         
                         # Vectorized check? No, prototypes is a dict.
                         # We need to gather prototypes for y
                         
                         batch_prototypes = []
                         valid_indices = []
                         
                         for k in range(self.batch_size):
                             label = y[k].item()
                             if label in prototypes:
                                 batch_prototypes.append(prototypes[label])
                                 valid_indices.append(k)
                         
                         if valid_indices:
                             batch_prototypes = torch.stack(batch_prototypes).to(self.device)
                             # student_feature[valid_indices] vs batch_prototypes
                             # Use MSE or Cosine distance
                             # Here using MSE for simplicity and "centroid" concept
                             loss_proto += F.mse_loss(student_feature[valid_indices], batch_prototypes)
                         
                         continue

                    _teacher = value["model"].to(self.device)
                    _teacher.eval()

                    weight = self.label_weights[y][:, idx].reshape(-1, 1)
                    weight = torch.tensor(weight.squeeze(), dtype=torch.float32, device=self.device)
                    
                    # Skip if weight is 0
                    if torch.sum(weight) == 0:
                        _teacher.to("cpu")
                        continue

                    teacher_logit, _ = _teacher(gen_output)
                    loss_cls += torch.mean(F.cross_entropy(teacher_logit.detach(), y_input) * weight)
                    _teacher.to("cpu")                                                  # move model back to cpu to save memory

                loss = self.ensemble_gamma * loss_con + self.ensemble_beta * loss_cls + self.ensemble_eta * loss_div + 0.1 * loss_proto
                if isinstance(loss_con, torch.Tensor):
                    loss_con_total.append(loss_con.item() * self.ensemble_gamma)
                else:
                    loss_con_total.append(loss_con)
                
                if isinstance(loss_cls, torch.Tensor):
                    loss_cls_total.append(loss_cls.item() * self.ensemble_beta)
                else:
                    loss_cls_total.append(loss_cls)
                    
                loss_div_total.append(loss_div.item() * self.ensemble_eta)
                loss_G_total.append(loss.item())

                self.optimizer_cgan.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 10)               # clip the gradient to prevent exploding
                self.optimizer_cgan.step()

            # ''' save the images from the generator '''
            gen_output = F.interpolate(gen_output, scale_factor=2, mode='bilinear', align_corners=False)
            save_image(make_grid(gen_output[:8], nrow=8, normalize=True), f"./figures/gen_output.png")

            ''' Train student (global model) '''
            generator.eval()
            global_model.train()

            ''' Sample new data '''
            # y = np.random.choice(self.qualified_labels, self.gen_batch_size)
            y = self.generate_labels(self.gen_batch_size, total_label_weights)
            y_input = F.one_hot(torch.tensor(y), num_classes=self.args.num_classes).type(torch.float32).to(self.device)

            for _ in range(self.inner_round_d):
                z = torch.randn((self.gen_batch_size, self.z_dim), device=self.device)
                gen_output = generator(z, y_input)
                student_logit, _ = global_model(gen_output)

                t_logit_merge = 0
                valid_teacher_count = 0
                with torch.no_grad():
                    for idx, (sig, value) in enumerate(self.knowledge_pool.items()):
                        if value["model"] is None:
                            continue
                            
                        _teacher = value["model"].to(self.device)
                        _teacher.eval()

                        weight = self.label_weights[y][:, idx].reshape(-1, 1)
                        expand_weight = np.tile(weight, (1, self.num_classes))
                        
                        if np.sum(expand_weight) == 0:
                            _teacher.to("cpu")
                            continue

                        teacher_logit, _ = _teacher(gen_output)

                        ''' knowledge distillation loss '''
                        t_logit_merge += teacher_logit.detach() * torch.tensor(expand_weight, dtype=torch.float32, device=self.device)
                        _teacher.to("cpu")                                                  # move model back to cpu to save memory
                        valid_teacher_count += 1
                
                if valid_teacher_count > 0:
                    loss_KD = F.kl_div(F.log_softmax(student_logit, dim=1)/self.T, F.softmax(t_logit_merge, dim=1)/self.T, reduction='batchmean')
                    loss_KD_total.append(loss_KD.item())

                    self.optimizer_server.zero_grad()
                    loss_KD.backward()
                    torch.nn.utils.clip_grad_norm_(global_model.parameters(), 10)               # clip the gradient to prevent exploding
                    self.optimizer_server.step()
                else:
                    loss_KD_total.append(0)

            val_acc = self.evaluate(global_model, self.valLoader, payloads)
            if val_acc > state["best_val_acc"]:
                state["best_val_acc"] = val_acc
                state["best_server_model"] = copy.deepcopy(global_model.state_dict())
                state["best_generator"] = copy.deepcopy(generator.state_dict())

            pbar.set_postfix({
                "loss_KD": np.mean(loss_KD_total),
                "loss_G": np.mean(loss_G_total),
                "loss_con": np.mean(loss_con_total),
                "loss_cls": np.mean(loss_cls_total),
                "loss_div": np.mean(loss_div_total),
                "val_acc": val_acc
            })

        # self.scheduler_cgan.step()
        # self.scheduler_server.step()

        # restore the best model
        generator.load_state_dict(state["best_generator"])
        global_model.load_state_dict(state["best_server_model"])
        global_model.eval()
        print("[Server | E. Distillation] Best val_acc: {:.4f}".format(state["best_val_acc"]))

        test_acc = self.evaluate(global_model, self.testLoader, payloads)
        print("[Server | Generative Knowledge Distilaltion]  After test_acc: {:.4f}".format(test_acc))
        return np.mean(loss_G_total), np.mean(loss_con_total), np.mean(loss_cls_total), np.mean(loss_div_total), np.mean(loss_KD_total), state["best_val_acc"], global_model.state_dict().values()


    def generate_labels(self, number, cls_num):
        labels = np.arange(number)
        proportions = cls_num / cls_num.sum()
        proportions = (np.cumsum(proportions) * number).astype(int)[:-1]
        labels_split = np.split(labels, proportions)
        for i in range(len(labels_split)):
            labels_split[i].fill(i)
        labels = np.concatenate(labels_split)
        np.random.shuffle(labels)
        return labels.astype(int)


    def get_label_weights_from_knowledge_pool(self, knowledge_pool):
        MIN_SAMPLES_PER_LABEL = 1
        label_weights = np.zeros((self.args.num_classes, len(knowledge_pool)))

        for i, (sig, value) in enumerate(knowledge_pool.items()):
            for label, num_data in value["data_summary"].items():
                label_weights[label, i] += num_data  

        qualified_labels = np.where(label_weights.sum(axis=1) >= MIN_SAMPLES_PER_LABEL)[0]
        for i in range(self.args.num_classes):
            if np.sum(label_weights[i], axis=0) > 0:                    # avoid division by zero (lack of data at current round)
                label_weights[i] /= np.sum(label_weights[i], axis=0)
            else:
                label_weights[i] = 0

        return label_weights, qualified_labels


    def combine_weights(self, weights_1, weights_2):
        ''' Combine two weight tensors '''
        weight = weights_1 + weights_2

        if weight.ndim > 1:
            for i in range(len(weight)):
                if np.sum(weight[i], axis=0) > 0:                    # avoid division by zero (lack of data at current round)
                    weight[i] /= np.sum(weight[i], axis=0)
                else:
                    weight[i] = 0
        else:
            weight /= np.sum(weight, axis=0)

        return weight


    def get_age_weight(self,):
        ''' Calculate the age weight for each model in the knowledge pool using HAKP decay '''
        age_weights = np.zeros(len(self.knowledge_pool))
        for idx, (sig, value) in enumerate(self.knowledge_pool.items()):
            layer = value.get("layer", "Active")
            decay_rate = self.hakp_lambda.get(layer, -0.1)
            
            # Use decay rate with age
            # If rate is negative, exp(age * rate) decays as age increases
            if value["online_age"] > 0:
                age_weights[idx] = math.exp(value["online_age"] * decay_rate)
            else:
                age_weights[idx] = math.exp(value["offline_age"] * decay_rate)

        ''' Normalize the age weight '''
        if np.sum(age_weights) > 0:
            age_weights /= np.sum(age_weights, axis=0)
        return age_weights


    def get_importance_weight(self,):
        ''' Importance weight for each model can be approximated by the sum of the label weights '''
        label_weights, _ = self.get_label_weights_from_knowledge_pool(self.knowledge_pool)
        importance_weights = np.sum(label_weights, axis=0)

        ''' Normalize the importance weight '''
        if np.sum(importance_weights) > 0:
            importance_weights /= np.sum(importance_weights, axis=0)
        return importance_weights


    def update_knowledge_pool(self, client_list, rounds, global_payload=None):
        # 1. Update Participation History
        for client in client_list:
            if client.cid not in self.participation_history:
                self.participation_history[client.cid] = []
            
            if client.status == "online":
                self.participation_history[client.cid].append(rounds)
                
        # 2. Re-evaluate Layers (Every R rounds)
        if rounds % self.hakp_R == 0 and rounds > 0:
            self.evaluate_layers(client_list, rounds)

        # 3. Update Knowledge Pool
        for client in client_list:
            status = client.status
            
            # HAKP: Get Layer
            layer = self.client_layers.get(client.cid, "Active")
            
            # If online, we might update the model/prototype
            if status == "online":
                acc = client.test(payload=global_payload)["test_acc"]
                signature = "{cid}-{data}".format(cid=client.cid, data=str(list(client.data_distribution.keys())))
                
                # Compute Prototype
                prototype = self.compute_prototype(client)

                if signature in self.knowledge_pool:
                    item = self.knowledge_pool[signature]
                    item["layer"] = layer
                    item["prototype"] = prototype
                    item["payload"] = copy.deepcopy(client.local_payload)
                    item["data_summary"] = client.data_distribution
                    item["num_data"] = len(client.trainLoader.dataset)
                    item["performance"] = acc
                    item["online_age"] += 1
                    item["offline_age"] = 0
                    
                    if layer == "Edge":
                        item["model"] = None # Don't store model for Edge
                    else:
                        item["model"] = copy.deepcopy(client.model)
                else:
                    self.knowledge_pool[signature] = {
                        "model": copy.deepcopy(client.model) if layer != "Edge" else None,
                        "payload": copy.deepcopy(client.local_payload),
                        "data_summary": client.data_distribution,
                        "num_data": len(client.trainLoader.dataset),
                        "performance": acc,
                        "online_age": 1,
                        "offline_age": 0,
                        "layer": layer,
                        "prototype": prototype,
                    }
            else:
                # Update age for offline clients
                signature = "{cid}-{data}".format(cid=client.cid, data=str(list(client.data_distribution.keys())))
                if signature in self.knowledge_pool:
                    self.knowledge_pool[signature]["online_age"] = 0
                    self.knowledge_pool[signature]["offline_age"] += 1


    def compute_prototype(self, client):
        ''' Compute knowledge prototype for the client '''
        client.model.to(self.device)
        client.model.eval()
        
        prototypes = {} # {label: [features]}
        counts = {}
        
        with torch.no_grad():
            for x, labels in client.trainLoader:
                x = x.to(self.device)
                _, features = client.model(x)
                features = features.cpu()
                
                for i in range(len(labels)):
                    label = labels[i].item()
                    feat = features[i]
                    if label not in prototypes:
                        prototypes[label] = feat
                        counts[label] = 1
                    else:
                        prototypes[label] += feat
                        counts[label] += 1
                        
        client.model.to("cpu")
        
        # Average
        final_prototypes = {}
        for label, feat_sum in prototypes.items():
            final_prototypes[label] = feat_sum / counts[label]
            
        return final_prototypes

    def evaluate_layers(self, client_list, rounds):
        ''' Calculate CVS and assign layers '''
        # Calculate Global Class Counts (N_j)
        N_total = 0
        N_j = np.zeros(self.num_classes)
        
        # Aggregate data from ALL clients (assuming we know their distribution)
        for client in client_list:
            for label, count in client.data_distribution.items():
                N_j[label] += count
                N_total += count
        
        cvs_scores = {}
        for client in client_list:
            # DSS
            dss = 0
            N_i = sum(client.data_distribution.values())
            if N_i > 0:
                for label, n_ij in client.data_distribution.items():
                    scarcity = 1 - (N_j[label] / N_total) if N_total > 0 else 0
                    focus = n_ij / N_i
                    dss += scarcity * focus
            
            # PSS
            history = [r for r in self.participation_history[client.cid] if r > rounds - self.hakp_T]
            participation_count = len(history)
            
            if participation_count > 1:
                intervals = np.diff(history)
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                cv = std_interval / mean_interval if mean_interval > 0 else 0
            else:
                cv = 1.0 # High variation if only 0 or 1 participation
            
            pss = (participation_count / self.hakp_T) * (1 - cv)
            
            cvs = self.hakp_alpha * dss + self.hakp_beta * pss
            cvs_scores[client.cid] = cvs
            self.cvs_history[client.cid] = cvs

        # Determine Thresholds
        scores = list(cvs_scores.values())
        if not scores:
            return
            
        theta_high = np.percentile(scores, 80)
        theta_low = np.percentile(scores, 20)
        
        # Assign Layers with Hysteresis and Consecutive Observation
        CONSECUTIVE_ROUNDS = 2
        
        for client in client_list:
            cid = client.cid
            current_layer = self.client_layers.get(cid, "Active")
            score = cvs_scores[cid]
            
            # Determine candidate layer based on score only
            candidate_layer = current_layer
            if score > theta_high + self.hakp_delta:
                candidate_layer = "Core"
            elif score < theta_low - self.hakp_delta:
                candidate_layer = "Edge"
            elif current_layer == "Core" and score < theta_high:
                candidate_layer = "Active" # Drop from Core
            elif current_layer == "Edge" and score > theta_low:
                candidate_layer = "Active" # Rise from Edge
            
            # Check consecutive observation
            if candidate_layer != current_layer:
                self.layer_counters[cid] = self.layer_counters.get(cid, 0) + 1
                if self.layer_counters[cid] >= CONSECUTIVE_ROUNDS:
                     self.client_layers[cid] = candidate_layer
                     self.layer_counters[cid] = 0 # Reset counter after switch
            else:
                self.layer_counters[cid] = 0 # Reset counter if condition not met

    def show_knowledge_pool(self,):
        print("\n-----> Knowledge Pool Status, Size: {}".format(len(self.knowledge_pool)))
        for key, value in self.knowledge_pool.items():
            layer = value.get("layer", "N/A")
            line_head = Fore.YELLOW + "[Tag: {} | {}]".format(key, layer)
            line_mid1 = "Acc.: {:.4f}".format(value["performance"]) + Fore.RESET
            line_mid2 = "{}On. Age: {}".format(Fore.GREEN if value["online_age"] > 0 else Fore.LIGHTBLACK_EX, value["online_age"]) + Fore.RESET
            line_tail = "{}Off. Age: {}".format(Fore.RED if value["offline_age"] > 0 else Fore.LIGHTBLACK_EX, value["offline_age"]) + Fore.RESET
            print("{:<55} {:<25} {:<26} {:<27}".format(line_head, line_mid1, line_mid2, line_tail))
        print("-" * 101 + "\n")


    def aggregate_from_knowledge_pool(self, rounds) -> np.ndarray:
        ''' Aggregate with all models in the knowledge pool (no selection) '''
        # HAKP: Filter out Edge clients (model is None)
        valid_items = [(sig, val) for sig, val in self.knowledge_pool.items() if val["model"] is not None]
        
        if not valid_items:
            return None, None

        client_numData_model_pair = [
            (value["model"].to(self.device), value["num_data"]) for sig, value in valid_items
        ]

        # Create a list of weights, each multiplied by the related number of examples
        model_weights = [
            [model.state_dict()[params] for params in model.state_dict()] for model, _ in client_numData_model_pair
        ]

        # Calculate the totol number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples in client_numData_model_pair])

        # Calculate the weights of each model
        dataset_weights = [
            num_examples / num_examples_total for _, num_examples in client_numData_model_pair
        ]

        # Get the age weight of each model
        # Recalculate age weights for ONLY the valid models to match dimensions
        age_weights = []
        for sig, value in valid_items:
            layer = value.get("layer", "Active")
            decay_rate = self.hakp_lambda.get(layer, -0.1)
            if value["online_age"] > 0:
                w = math.exp(value["online_age"] * decay_rate)
            else:
                w = math.exp(value["offline_age"] * decay_rate)
            age_weights.append(w)
        
        age_weights = np.array(age_weights)
        if np.sum(age_weights) > 0:
            age_weights /= np.sum(age_weights)

        # Get the importance weight of each model
        # Need to recalculate label weights for valid items only? 
        # Or just use the global ones and slice?
        # Let's recalculate for simplicity and correctness
        temp_pool = {sig: val for sig, val in valid_items}
        label_weights, _ = self.get_label_weights_from_knowledge_pool(temp_pool)
        importance_weights = np.sum(label_weights, axis=0)
        if np.sum(importance_weights) > 0:
            importance_weights /= np.sum(importance_weights)

        # HAKP: Layer Weight Factor (gamma)
        layer_weights = np.array([self.hakp_gamma.get(val.get("layer", "Active"), 1.0) for sig, val in valid_items])

        # Calculate the final weights
        # Formula: gamma * (age_weight * dataset_weight + importance_weight)
        # Note: dataset_weights is effectively "awi" (aggregation weight based on data/performance? No, just data size here).
        # The prompt says awi * epsilon + dwi.
        # Here we have age_weights * dataset_weights + importance_weights.
        
        final_weights = self.age_ld * (torch.tensor(age_weights, device=self.device) * torch.tensor(dataset_weights, device=self.device)) + \
                         + self.impt_ld * torch.tensor(importance_weights, device=self.device)
        
        # Apply Gamma
        final_weights = final_weights * torch.tensor(layer_weights, device=self.device)
        
        if torch.sum(final_weights) > 0:
            final_weights /= torch.sum(final_weights)
        else:
             # Fallback if weights are zero
             final_weights = torch.ones_like(final_weights) / len(final_weights)

        # Compute average weight of each layer using the findal weights
        weights_prime = [
            reduce(torch.add, [w * weight for w, weight in zip(layer_updates, final_weights)])
            for layer_updates in zip(*model_weights)
        ]
        
        # Clean up GPU
        for model, _ in client_numData_model_pair:
            model.to("cpu")

        # Aggregate the payloads (class-wise by final weights, logits, protos, etc.)
        '''
        client_1_payload = {
            "data": {
                "class_0": [0.1, 0.2, 0.3],
                "class_1": [0.4, 0.5, 0}
        
        '''
        payloads_prime = {}
        # Only check payload type of first valid item
        if next(iter(temp_pool.items()))[1]["payload"]["type"] != "None":
            # Map index in final_weights back to correct client index? 
            # final_weights corresponds to valid_items list order.
            # payload_aggregation takes a dict of payloads.
            
            local_payload_list = {i: value["payload"]["data"] for i, (sig, value) in enumerate(valid_items)}
            client_weight_list = {i: final_weights[i] for i in range(len(valid_items))}
            
            payloads_prime = self.fl_strategy.payload_aggregation(
                local_payload_list=local_payload_list,
                client_weight_list = client_weight_list
            )

        return weights_prime, payloads_prime
    

    def evaluate(self, server_model, testLoader, payloads=None) -> Tuple[float, float]:
        avg_acc = []
        for name, loader in testLoader.items():
            acc_indi = self.fl_strategy._test(server_model, loader, payloads)["test_acc"]
            avg_acc.append(acc_indi)
        return np.mean(avg_acc)
