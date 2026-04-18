import torch
import torch.nn as nn
import torch.optim as optim
import random
from engines.base_engine import BaseTrainingEngine
from models_bert import BertForPreTraining

class BertEngine(BaseTrainingEngine):
    def initialize_model(self, input_size, output_size, lr):
        # We only use output_size (English vocab) to build the BERT vocabulary since
        # we are doing unsupervised monolingual pretraining on the English target text.
        self.model = BertForPreTraining(
            vocab_size=output_size, 
            d_model=256, 
            nhead=8, 
            num_layers=4, 
            dim_feedforward=1024, 
            dropout=0.1
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        # CrossEntropyLoss: ignore index 2 (PAD) for MLM, 
        # but for MLM we will actually pass labels and use ignore_index=-100
        self.criterion_mlm = nn.CrossEntropyLoss(ignore_index=-100)
        self.criterion_nsp = nn.CrossEntropyLoss()
        
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type == 'cuda'))

        # Special Tokens indices from our updated utils.py
        self.sos_id = 0 # CLS
        self.eos_id = 1 # SEP
        self.pad_id = 2 # PAD
        self.mask_id = 3 # MASK
        self.vocab_size = output_size

    def _prepare_batch(self, tgt_batch):
        # We need to create:
        # 1. input_ids: [CLS] Sentence A [SEP] Sentence B [SEP]
        # 2. token_type_ids: 0 for A, 1 for B
        # 3. next_sentence_label: 1 if IsNext, 0 if NotNext
        # 4. mlm_labels: original token if masked, -100 otherwise
        
        # To make it simple for the user's dataset (which are short sentences ~ max_length 25),
        # we will randomly split the `src` sequence into A and B for IsNext.
        # For NotNext, we will take A from current and B from a randomly shifted item.
        
        bsz, seq_len = tgt_batch.shape
        
        new_batch = []
        nsp_labels = []
        token_type_batch = []
        
        for i in range(bsz):
            # Extract valid tokens (remove SOS 0, EOS 1, PAD 2)
            # Actually, tgt in DataLoader already includes 0 and 1, and padded with 2.
            # So let's strip them!
            valid_tokens = []
            for t in tgt_batch[i].tolist():
                if t not in [self.sos_id, self.eos_id, self.pad_id]:
                    valid_tokens.append(t)
            
            # If sequence is extremely short, just don't split it
            if len(valid_tokens) < 4:
                # Force dummy behavior
                segment_a = valid_tokens
                segment_b = valid_tokens
                is_next = 1
            else:
                is_next = random.random() > 0.5
                if is_next:
                    # Random split point
                    split_idx = random.randint(1, len(valid_tokens)-2)
                    segment_a = valid_tokens[:split_idx]
                    segment_b = valid_tokens[split_idx:]
                else:
                    # Random split point, but take B from next batch item
                    split_idx = random.randint(1, len(valid_tokens)-2)
                    segment_a = valid_tokens[:split_idx]
                    
                    rand_i = (i + random.randint(1, bsz-1)) % bsz
                    other_valid_tokens = [t for t in tgt_batch[rand_i].tolist() if t not in [self.sos_id, self.eos_id, self.pad_id]]
                    if len(other_valid_tokens) < 2:
                        segment_b = segment_a
                        is_next = 1
                    else:
                        split_idx_other = random.randint(1, len(other_valid_tokens)-1)
                        segment_b = other_valid_tokens[split_idx_other:]
            
            # Construct Sequence
            input_ids = [self.sos_id] + segment_a + [self.eos_id] + segment_b + [self.eos_id]
            token_ids = [0] * (len(segment_a) + 2) + [1] * (len(segment_b) + 1)
            
            new_batch.append(torch.tensor(input_ids, dtype=torch.long))
            token_type_batch.append(torch.tensor(token_ids, dtype=torch.long))
            nsp_labels.append(1 if is_next else 0)
            
        # Pad sequence manually
        input_ids = torch.nn.utils.rnn.pad_sequence(new_batch, batch_first=True, padding_value=self.pad_id).to(self.device)
        token_type_ids = torch.nn.utils.rnn.pad_sequence(token_type_batch, batch_first=True, padding_value=self.pad_id).to(self.device)
        nsp_labels = torch.tensor(nsp_labels, dtype=torch.long).to(self.device)
        
        # Create MLM Mask
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, 0.15)
        
        # Don't mask special tokens
        special_tokens_mask = (input_ids == self.sos_id) | (input_ids == self.eos_id) | (input_ids == self.pad_id)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # 80% of the time, we replace masked input tokens with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(4, self.vocab_size, labels.shape, dtype=torch.long).to(self.device)
        input_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10%) we keep the masked input tokens unchanged
        # labels only contain useful labels for masked indices! otherwise -100
        labels[~masked_indices] = -100
        
        attention_mask = (input_ids != self.pad_id).float()
        
        return input_ids, token_type_ids, attention_mask, labels, nsp_labels

    def train_one_epoch(self, train_loader, epoch_idx):
        self.model.train()
        running_loss = 0.0
        train_count = 0

        for src, tgt in train_loader:
            input_ids, token_type_ids, attention_mask, mlm_labels, nsp_labels = self._prepare_batch(tgt)

            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                prediction_scores, seq_relationship_score = self.model(
                    input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
                )
                
                # Reshape for MLM cross entropy
                vocab_size = prediction_scores.shape[-1]
                loss_mlm = self.criterion_mlm(prediction_scores.view(-1, vocab_size), mlm_labels.view(-1))
                loss_nsp = self.criterion_nsp(seq_relationship_score, nsp_labels)
                
                loss = loss_mlm + loss_nsp

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            train_count += 1
            yield running_loss / train_count

    def validate(self, val_loader):
        self.model.eval()
        total_val_loss = 0.0
        val_count = 0

        with torch.no_grad():
            for src, tgt in val_loader:
                input_ids, token_type_ids, attention_mask, mlm_labels, nsp_labels = self._prepare_batch(tgt)

                with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                    prediction_scores, seq_relationship_score = self.model(
                        input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
                    )
                    
                    vocab_size = prediction_scores.shape[-1]
                    loss_mlm = self.criterion_mlm(prediction_scores.view(-1, vocab_size), mlm_labels.view(-1))
                    loss_nsp = self.criterion_nsp(seq_relationship_score, nsp_labels)
                    loss = loss_mlm + loss_nsp

                total_val_loss += loss.item()
                val_count += 1

        return total_val_loss / val_count if val_count > 0 else 0.0

    def save_checkpoint(self, path, epoch_idx, val_loss):
        torch.save({
            'epoch': epoch_idx,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }, path)
