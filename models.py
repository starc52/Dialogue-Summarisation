from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from torch import nn
from transformers import BartForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from torch.nn import CrossEntropyLoss
from transformers.models.bart.modeling_bart import BartConfig
import torch
from typing import *
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.bart.modeling_bart import shift_tokens_right
import random
from tqdm import tqdm
import gc
import itertools

__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""


class T5WithSCL(T5ForConditionalGeneration):
    def __init__(self, config: BartConfig):
        super().__init__(config)

    def set_losses_list(self, SCLossesList=['token']):
        self.SCLossesList = SCLossesList

    def set_scl_coeff(self, scl_coeff=1e-1):
        self.scl_coeff = scl_coeff

    def token_scl(self,
                  last_hidden_state: torch.FloatTensor,
                  spk_utt_pos: torch.LongTensor,
                  ) -> torch.FloatTensor:
        r"""
        last_hidden_state (torch.LongTensor) of shape (batch_size, sequence_length, n_dims):
            Output of the last layer of the encoder.
        spk_utt_pos (torch.LongTensor) of shape (batch_size, sequence_length,):
            metadata about the speaker tokens and utterance tokens
        Returns:
        Token Level Supervised Constrastive Loss (torch.LongTensor)
        """
        batch_scl = 0
        for i in range(len(spk_utt_pos)):
            batch_element = spk_utt_pos[i]
            spk_utt_list = []
            spk_dict = {'start': 0, 'end': 0, 'spk_id': 0, 'bool': False}
            utt_dict = {'start': 0, 'end': 0, 'spk_id': 0, 'bool': False}
            for j in range(len(batch_element)):
                if batch_element[j] == 0 and j > 0:
                    utt_dict['end'] = j
                    utt_dict['bool'] = False
                    spk_utt_list.append({'spk': [spk_dict['start'], spk_dict['end'], spk_dict['spk_id']],
                                         'utt': [utt_dict['start'], utt_dict['end'], utt_dict['spk_id']]})
                    break
                if batch_element[j] > 0 and spk_dict['bool'] == False:
                    utt_dict['end'] = j
                    utt_dict['bool'] = False
                    if j > 1:
                        spk_utt_list.append({'spk': [spk_dict['start'], spk_dict['end'], spk_dict['spk_id']],
                                             'utt': [utt_dict['start'], utt_dict['end'], utt_dict['spk_id']]})
                    spk_dict['start'] = j
                    spk_dict['bool'] = True
                    spk_dict['spk_id'] = batch_element[j]

                if batch_element[j] < 0 and spk_dict['bool'] == True:
                    spk_dict['end'] = j
                    spk_dict['bool'] = False
                    utt_dict['spk_id'] = spk_dict['spk_id']
                    utt_dict['start'] = j
                    utt_dict['bool'] = True
            # uniq spks
            if spk_utt_list[0]['spk'][2] == 0:
                continue
            uniq_spks = list(set([int(dic['spk'][2].cpu()) for dic in spk_utt_list]))
            if len(uniq_spks) == 1:
                continue
            # spk_utt_states
            spk_utt_states = {spk: [] for spk in uniq_spks}

            for spk in uniq_spks:
                for dic in spk_utt_list:
                    if spk == dic['utt'][2]:
                        spk_utt_states[spk].append(last_hidden_state[i, dic['utt'][0]:dic['utt'][1]])

            # ---------- hitesh------------------------------
            # positive samples
            # L_pos = 0
            # L_neg = 0

            # sampled_spk_utt_states = []

            # for spk in uniq_spks:
            #     utts = len(spk_utt_states[spk])
            #     spk_utt = []
            #     if utts > 1:
            #         # ids = random.sample(list(range(len(spk_utt_states[spk]))), random.randint(1, utts))
            #         ids = random.sample(list(range(len(spk_utt_states[spk]))), 2)
            #         for i in ids:
            #           spk_utt.append(spk_utt_states[spk][i])
            #     sampled_spk_utt_states.append(spk_utt)

            # for instance in sampled_spk_utt_states:
            #   for i in range(len(instance)):
            #     for j in range(len(instance)):
            #       mat_mul = torch.einsum('ij, kj->ik', instance[i], instance[j])
            #       sigm = torch.sigmoid(mat_mul)
            #       log = torch.log(sigm)
            #       L_pos += torch.sum(-1 * log)
            # # print("L_pos", L_pos)

            # #negative loss
            # for i in range(0,len(sampled_spk_utt_states)):
            #   instance = sampled_spk_utt_states[i]

            #   neg_instances = sampled_spk_utt_states[:i]+sampled_spk_utt_states[i+1:]
            #   neg_instances = list(itertools.chain(*neg_instances))
            #   # neg_instances = random.choices(neg_instances,k = random.randint(1, len(neg_instances)))
            #   if len(neg_instances)>0:
            #     # print(len(neg_instances))
            #     # print("-------------------------")
            #     # print(sampled_spk_utt_states)
            #     neg_instances = random.choices(neg_instances,k = 2)
            #     for i in range(len(instance)):
            #       for j in range(len(neg_instances)):
            #         mat_mul = torch.einsum('ij, kj->ik', instance[i], neg_instances[j])
            #         sigm = torch.sigmoid(mat_mul)
            #         log = torch.log(1 - sigm+1e-5)
            #         L_neg += torch.sum(-1 * log)
            # ---------- hitesh------------------------------

            # positive samples
            L_pos = 0

            for spk in uniq_spks:
                if len(spk_utt_states[spk]) > 1:
                    ids = random.sample(list(range(len(spk_utt_states[spk]))), 2)
                    id1 = ids[0]
                    id2 = ids[1]
                    mat_mul = torch.einsum('ij, kj->ik', spk_utt_states[spk][id1], spk_utt_states[spk][id1])
                    sigm = torch.sigmoid(mat_mul)
                    log = torch.log(sigm)
                    L_pos += torch.sum(-1 * log)
                    L_pos = torch.nan_to_num(L_pos, posinf=1e10, neginf=-1e10)
            # print("L_pos", L_pos)
            # negative samples

            L_neg = 0
            for spk in uniq_spks:
                new_uniq_spks = uniq_spks.copy()
                new_uniq_spks.remove(spk)

                spk2 = random.choice(new_uniq_spks)

                id1 = random.randint(0, len(spk_utt_states[spk]) - 1)
                id2 = random.randint(0, len(spk_utt_states[spk2]) - 1)

                mat_mul = torch.einsum('ij, kj->ik', spk_utt_states[spk][id1], spk_utt_states[spk2][id2])
                sigm = torch.sigmoid(mat_mul)
                # print(1 - sigm)
                # print(1 - sigm+1e-5)
                log = torch.log(1 - sigm + 1e-5)
                L_neg += torch.sum(-1 * log)

                L_neg = torch.nan_to_num(L_neg, posinf=1e10, neginf=-1e10)

            # print("L_neg", L_neg)

            batch_scl += L_pos
            batch_scl += L_neg
            # wandb.log({"batch_scl-token": batch_scl})
        batch_scl /= last_hidden_state.size(0)
        gc.collect()
        return batch_scl

    def turn_scl(self,
                 last_hidden_state: torch.FloatTensor,
                 spk_utt_pos: torch.LongTensor,
                 ) -> torch.FloatTensor:
        r"""
        last_hidden_state (torch.LongTensor) of shape (batch_size, sequence_length, n_dims):
            Output of the last layer of the encoder.
        spk_utt_pos (torch.LongTensor) of shape (batch_size, sequence_length,):
            metadata about the speaker tokens and utterance tokens
        Returns:
        Turn Level Supervised Constrastive Loss (torch.LongTensor)
        """
        batch_scl = 0
        for i in range(len(spk_utt_pos)):
            batch_element = spk_utt_pos[i]
            spk_utt_list = []
            spk_dict = {'start': 0, 'end': 0, 'spk_id': 0, 'bool': False}
            utt_dict = {'start': 0, 'end': 0, 'spk_id': 0, 'bool': False}
            for j in range(len(batch_element)):
                if batch_element[j] == 0 and j > 0:
                    utt_dict['end'] = j
                    utt_dict['bool'] = False
                    spk_utt_list.append({'spk': [spk_dict['start'], spk_dict['end'], spk_dict['spk_id']],
                                         'utt': [utt_dict['start'], utt_dict['end'], utt_dict['spk_id']]})
                    break
                if batch_element[j] > 0 and spk_dict['bool'] == False:
                    utt_dict['end'] = j
                    utt_dict['bool'] = False
                    if j > 1:
                        spk_utt_list.append({'spk': [spk_dict['start'], spk_dict['end'], spk_dict['spk_id']],
                                             'utt': [utt_dict['start'], utt_dict['end'], utt_dict['spk_id']]})
                    spk_dict['start'] = j
                    spk_dict['bool'] = True
                    spk_dict['spk_id'] = batch_element[j]

                if batch_element[j] < 0 and spk_dict['bool'] == True:
                    spk_dict['end'] = j
                    spk_dict['bool'] = False
                    utt_dict['spk_id'] = spk_dict['spk_id']
                    utt_dict['start'] = j
                    utt_dict['bool'] = True
            # uniq spks
            if spk_utt_list[0]['spk'][2] == 0:
                continue
            uniq_spks = list(set([int(dic['spk'][2].cpu()) for dic in spk_utt_list]))
            if len(uniq_spks) == 1:
                continue
            # spk_utt_states
            spk_utt_states = {spk: [] for spk in uniq_spks}

            for spk in uniq_spks:
                for dic in spk_utt_list:
                    if spk == dic['utt'][2]:
                        mean_pool = torch.mean(last_hidden_state[i, dic['utt'][0]:dic['utt'][1]], 0)
                        spk_utt_states[spk].append(mean_pool)

            # positive samples
            L_pos = 0
            for spk in uniq_spks:
                if len(spk_utt_states[spk]) > 1:
                    ids = random.sample(list(range(len(spk_utt_states[spk]))), 2)
                    id1 = ids[0]
                    id2 = ids[1]
                    mat_mul = torch.einsum('i, j->', spk_utt_states[spk][id1], spk_utt_states[spk][id1])
                    sigm = torch.sigmoid(mat_mul)
                    log = torch.log(sigm)
                    L_pos += torch.sum(-1 * log)
                    # L_pos = torch.nan_to_num(L_pos, posinf = 1e10, neginf = -1e10)
            # print("L_pos", L_pos)
            # negative samples
            L_neg = 0
            for spk in uniq_spks:
                new_uniq_spks = uniq_spks.copy()
                new_uniq_spks.remove(spk)

                spk2 = random.choice(new_uniq_spks)

                id1 = random.randint(0, len(spk_utt_states[spk]) - 1)
                id2 = random.randint(0, len(spk_utt_states[spk2]) - 1)

                mat_mul = torch.einsum('i, j->', spk_utt_states[spk][id1], spk_utt_states[spk2][id2])
                sigm = torch.sigmoid(mat_mul)
                # print(1 - sigm)
                # print(1 - sigm+1e-5)
                log = torch.log(1 - sigm + 1e-5)
                L_neg += torch.sum(-1 * log)

                # L_neg = torch.nan_to_num(L_neg, posinf = 1e10, neginf = -1e10)

            # print("L_neg", L_neg)

            batch_scl += L_pos
            batch_scl += L_neg
            # wandb.log({"batch_scl-turn": batch_scl})

        batch_scl /= last_hidden_state.size(0)
        gc.collect()
        return batch_scl

    def global_scl(self,
                   last_hidden_state: torch.FloatTensor,
                   spk_utt_pos: torch.LongTensor,
                   ) -> torch.FloatTensor:
        r"""
        last_hidden_state (torch.LongTensor) of shape (batch_size, sequence_length, n_dims):
            Output of the last layer of the encoder.
        spk_utt_pos (torch.LongTensor) of shape (batch_size, sequence_length,):
            metadata about the speaker tokens and utterance tokens
        Returns:
        Turn Level Supervised Constrastive Loss (torch.LongTensor)
        """
        batch_scl = 0
        for i in range(len(spk_utt_pos)):
            batch_element = spk_utt_pos[i]
            spk_utt_list = []
            spk_dict = {'start': 0, 'end': 0, 'spk_id': 0, 'bool': False}
            utt_dict = {'start': 0, 'end': 0, 'spk_id': 0, 'bool': False}
            for j in range(len(batch_element)):
                if batch_element[j] == 0 and j > 0:
                    utt_dict['end'] = j
                    utt_dict['bool'] = False
                    spk_utt_list.append({'spk': [spk_dict['start'], spk_dict['end'], spk_dict['spk_id']],
                                         'utt': [utt_dict['start'], utt_dict['end'], utt_dict['spk_id']]})
                    break
                if batch_element[j] > 0 and spk_dict['bool'] == False:
                    utt_dict['end'] = j
                    utt_dict['bool'] = False
                    if j > 1:
                        spk_utt_list.append({'spk': [spk_dict['start'], spk_dict['end'], spk_dict['spk_id']],
                                             'utt': [utt_dict['start'], utt_dict['end'], utt_dict['spk_id']]})
                    spk_dict['start'] = j
                    spk_dict['bool'] = True
                    spk_dict['spk_id'] = batch_element[j]

                if batch_element[j] < 0 and spk_dict['bool'] == True:
                    spk_dict['end'] = j
                    spk_dict['bool'] = False
                    utt_dict['spk_id'] = spk_dict['spk_id']
                    utt_dict['start'] = j
                    utt_dict['bool'] = True
            # uniq spks
            if spk_utt_list[0]['spk'][2] == 0:
                continue
            uniq_spks = list(set([int(dic['spk'][2].cpu()) for dic in spk_utt_list]))
            if len(uniq_spks) == 1:
                continue
            # spk_utt_states
            spk_utt_states = {spk: [] for spk in uniq_spks}

            for spk in uniq_spks:
                for dic in spk_utt_list:
                    if spk == dic['utt'][2]:
                        mean_pool = torch.mean(last_hidden_state[i, dic['utt'][0]:dic['utt'][1]], 0)
                        spk_utt_states[spk].append(mean_pool)

            # positive samples
            L_pos = 0
            L_neg = 0
            for spk in uniq_spks:
                if len(spk_utt_states[spk]) > 1:
                    ids = random.choice(list(range(len(spk_utt_states[spk]))))

                    spk_mean_exc = torch.mean(torch.vstack(
                        [spk_utt_states[spk][temp] for temp in range(len(spk_utt_states[spk])) if temp != ids]), 0)

                    pos_mat_mul = torch.einsum('i, j->', spk_utt_states[spk][ids], spk_mean_exc)
                    pos_sigm = torch.sigmoid(pos_mat_mul)
                    pos_log = torch.log(pos_sigm)
                    L_pos += torch.sum(-1 * pos_log)

                    # negative sample

                    new_uniq_spks = uniq_spks.copy()
                    new_uniq_spks.remove(spk)

                    spk2 = random.choice(new_uniq_spks)
                    id_neg = random.choice(list(range(len(spk_utt_states[spk2]))))
                    neg_mat_mul = torch.einsum('i, j->', spk_utt_states[spk2][id_neg], spk_mean_exc)
                    neg_sigm = torch.sigmoid(neg_mat_mul)
                    neg_log = torch.log(1 - neg_sigm + 1e-5)
                    L_neg += torch.sum(-1 * neg_log)

            # print("L_neg", L_neg)

            batch_scl += L_pos
            batch_scl += L_neg
            # wandb.log({"batch_scl-global": batch_scl})

        batch_scl /= last_hidden_state.size(0)
        gc.collect()
        return batch_scl

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            spk_utt_pos: Optional[torch.Tensor] = None,  ##changed here
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`
        Returns:
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            turn_attention_mask = None
            token_encoder_outputs = None
            tog_encoder_outputs = None

            if 'token' in self.SCLossesList:
                token_encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

            if 'turn' in self.SCLossesList or 'global' in self.SCLossesList:
                tog_attention_mask = torch.where(spk_utt_pos > 0, 0, attention_mask)
                tog_encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=tog_attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
            turn_attention_mask = None
            token_encoder_outputs = None
            tog_encoder_outputs = None

            if 'token' in self.SCLossesList:
                token_encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

            if 'turn' in self.SCLossesList or 'global' in self.SCLossesList:
                tog_attention_mask = torch.where(spk_utt_pos > 0, 0, attention_mask)
                tog_encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=tog_attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        # added here
        sc_loss = 0
        if 'token' in self.SCLossesList and labels is not None:
            sc_loss += self.token_scl(last_hidden_state=token_encoder_outputs['last_hidden_state'],
                                      spk_utt_pos=spk_utt_pos)
            # print(sc_loss)
        if 'turn' in self.SCLossesList and labels is not None:
            sc_loss += self.turn_scl(last_hidden_state=tog_encoder_outputs['last_hidden_state'],
                                     spk_utt_pos=spk_utt_pos)

        if 'global' in self.SCLossesList and labels is not None:
            sc_loss += self.global_scl(last_hidden_state=tog_encoder_outputs['last_hidden_state'],
                                       spk_utt_pos=spk_utt_pos)

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((masked_lm_loss + (self.scl_coeff * sc_loss),) + output) if loss is not None else output

        loss = None
        if masked_lm_loss is None:
            loss = None
        else:
            loss = masked_lm_loss + (self.scl_coeff * sc_loss)

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class BartWithSCL(BartForConditionalGeneration):
    def __init__(self, config: BartConfig):
        super().__init__(config)

    def set_losses_list(self, SCLossesList=['token']):
        self.SCLossesList = SCLossesList
    def set_scl_coeff(self, scl_coeff=1e-1):
        self.scl_coeff=scl_coeff
    def token_scl(self,
                  last_hidden_state: torch.FloatTensor,
                  spk_utt_pos: torch.LongTensor,
    ) -> torch.FloatTensor:
        r"""
        last_hidden_state (torch.LongTensor) of shape (batch_size, sequence_length, n_dims):
            Output of the last layer of the encoder.
        spk_utt_pos (torch.LongTensor) of shape (batch_size, sequence_length,):
            metadata about the speaker tokens and utterance tokens
        Returns:
        Token Level Supervised Constrastive Loss (torch.LongTensor)
        """
        batch_scl = 0
        for i in range(len(spk_utt_pos)):
            batch_element = spk_utt_pos[i]
            spk_utt_list = []
            spk_dict = {'start': 0, 'end': 0, 'spk_id': 0, 'bool': False}
            utt_dict = {'start': 0, 'end': 0, 'spk_id': 0, 'bool': False}
            for j in range(len(batch_element)):
                if batch_element[j] == 0 and j > 0:
                    utt_dict['end'] = j
                    utt_dict['bool'] = False
                    spk_utt_list.append({'spk': [spk_dict['start'], spk_dict['end'], spk_dict['spk_id']],
                                         'utt': [utt_dict['start'], utt_dict['end'], utt_dict['spk_id']]})
                    break
                if batch_element[j] > 0 and spk_dict['bool'] == False:
                    utt_dict['end'] = j
                    utt_dict['bool'] = False
                    if j > 1:
                        spk_utt_list.append({'spk': [spk_dict['start'], spk_dict['end'], spk_dict['spk_id']],
                                             'utt': [utt_dict['start'], utt_dict['end'], utt_dict['spk_id']]})
                    spk_dict['start'] = j
                    spk_dict['bool'] = True
                    spk_dict['spk_id'] = batch_element[j]
                    

                if batch_element[j] < 0 and spk_dict['bool'] == True:
                    spk_dict['end'] = j
                    spk_dict['bool'] = False
                    utt_dict['spk_id'] = spk_dict['spk_id']
                    utt_dict['start'] = j
                    utt_dict['bool'] = True
            # uniq spks
            if spk_utt_list[0]['spk'][2]==0:
                continue
            uniq_spks = list(set([int(dic['spk'][2].cpu()) for dic in spk_utt_list]))
            if len(uniq_spks)==1:
                continue
            # spk_utt_states
            spk_utt_states = {spk: [] for spk in uniq_spks}

            for spk in uniq_spks:
                for dic in spk_utt_list:
                    if spk == dic['utt'][2]:
                        spk_utt_states[spk].append(last_hidden_state[i, dic['utt'][0]:dic['utt'][1]])

            # positive samples
            L_pos = 0
            for spk in uniq_spks:
                if len(spk_utt_states[spk]) > 1:
                    ids = random.sample(list(range(len(spk_utt_states[spk]))), 2)
                    id1 = ids[0]
                    id2 = ids[1]
                    mat_mul = torch.einsum('ij, kj->ik', spk_utt_states[spk][id1], spk_utt_states[spk][id1])
                    sigm = torch.sigmoid(mat_mul)
                    log = torch.log(sigm)
                    L_pos += torch.sum(-1 * log)
                    # L_pos = torch.nan_to_num(L_pos, posinf = 1e10, neginf = -1e10)
            # print("L_pos", L_pos)
            # negative samples
            L_neg = 0
            for spk in uniq_spks:
                new_uniq_spks = uniq_spks.copy()
                new_uniq_spks.remove(spk)

                spk2 = random.choice(new_uniq_spks)

                id1 = random.randint(0, len(spk_utt_states[spk])-1)
                id2 = random.randint(0, len(spk_utt_states[spk2])-1)

                mat_mul = torch.einsum('ij, kj->ik', spk_utt_states[spk][id1], spk_utt_states[spk2][id2])
                sigm = torch.sigmoid(mat_mul)
                # print(1 - sigm)
                # print(1 - sigm+1e-5)
                log = torch.log(1 - sigm+1e-5)
                L_neg += torch.sum(-1 * log)
                
                # L_neg = torch.nan_to_num(L_neg, posinf = 1e10, neginf = -1e10)

            # print("L_neg", L_neg)
            
            batch_scl += L_pos
            batch_scl += L_neg
        batch_scl /= last_hidden_state.size(0)
        gc.collect()
        return batch_scl
    
    def turn_scl(self,
                  last_hidden_state: torch.FloatTensor,
                  spk_utt_pos: torch.LongTensor,
    ) -> torch.FloatTensor:
        r"""
        last_hidden_state (torch.LongTensor) of shape (batch_size, sequence_length, n_dims):
            Output of the last layer of the encoder.
        spk_utt_pos (torch.LongTensor) of shape (batch_size, sequence_length,):
            metadata about the speaker tokens and utterance tokens
        Returns:
        Turn Level Supervised Constrastive Loss (torch.LongTensor)
        """
        batch_scl = 0
        for i in range(len(spk_utt_pos)):
            batch_element = spk_utt_pos[i]
            spk_utt_list = []
            spk_dict = {'start': 0, 'end': 0, 'spk_id': 0, 'bool': False}
            utt_dict = {'start': 0, 'end': 0, 'spk_id': 0, 'bool': False}
            for j in range(len(batch_element)):
                if batch_element[j] == 0 and j > 0:
                    utt_dict['end'] = j
                    utt_dict['bool'] = False
                    spk_utt_list.append({'spk': [spk_dict['start'], spk_dict['end'], spk_dict['spk_id']],
                                         'utt': [utt_dict['start'], utt_dict['end'], utt_dict['spk_id']]})
                    break
                if batch_element[j] > 0 and spk_dict['bool'] == False:
                    utt_dict['end'] = j
                    utt_dict['bool'] = False
                    if j > 1:
                        spk_utt_list.append({'spk': [spk_dict['start'], spk_dict['end'], spk_dict['spk_id']],
                                             'utt': [utt_dict['start'], utt_dict['end'], utt_dict['spk_id']]})
                    spk_dict['start'] = j
                    spk_dict['bool'] = True
                    spk_dict['spk_id'] = batch_element[j]
                    

                if batch_element[j] < 0 and spk_dict['bool'] == True:
                    spk_dict['end'] = j
                    spk_dict['bool'] = False
                    utt_dict['spk_id'] = spk_dict['spk_id']
                    utt_dict['start'] = j
                    utt_dict['bool'] = True
            # uniq spks
            if spk_utt_list[0]['spk'][2]==0:
                continue
            uniq_spks = list(set([int(dic['spk'][2].cpu()) for dic in spk_utt_list]))
            if len(uniq_spks)==1:
                continue
            # spk_utt_states
            spk_utt_states = {spk: [] for spk in uniq_spks}

            for spk in uniq_spks:
                for dic in spk_utt_list:
                    if spk == dic['utt'][2]:
                        mean_pool = torch.mean(last_hidden_state[i, dic['utt'][0]:dic['utt'][1]], 0)
                        spk_utt_states[spk].append(mean_pool)

            # positive samples
            L_pos = 0
            for spk in uniq_spks:
                if len(spk_utt_states[spk]) > 1:
                    ids = random.sample(list(range(len(spk_utt_states[spk]))), 2)
                    id1 = ids[0]
                    id2 = ids[1]
                    mat_mul = torch.einsum('i, j->', spk_utt_states[spk][id1], spk_utt_states[spk][id1])
                    sigm = torch.sigmoid(mat_mul)
                    log = torch.log(sigm)
                    L_pos += torch.sum(-1 * log)
                    # L_pos = torch.nan_to_num(L_pos, posinf = 1e10, neginf = -1e10)
            # print("L_pos", L_pos)
            # negative samples
            L_neg = 0
            for spk in uniq_spks:
                new_uniq_spks = uniq_spks.copy()
                new_uniq_spks.remove(spk)

                spk2 = random.choice(new_uniq_spks)

                id1 = random.randint(0, len(spk_utt_states[spk])-1)
                id2 = random.randint(0, len(spk_utt_states[spk2])-1)

                mat_mul = torch.einsum('i, j->', spk_utt_states[spk][id1], spk_utt_states[spk2][id2])
                sigm = torch.sigmoid(mat_mul)
                # print(1 - sigm)
                # print(1 - sigm+1e-5)
                log = torch.log(1 - sigm+1e-5)
                L_neg += torch.sum(-1 * log)
                
                # L_neg = torch.nan_to_num(L_neg, posinf = 1e10, neginf = -1e10)

            # print("L_neg", L_neg)
            
            batch_scl += L_pos
            batch_scl += L_neg
        batch_scl /= last_hidden_state.size(0)
        gc.collect()
        return batch_scl
    
    def global_scl(self,
                  last_hidden_state: torch.FloatTensor,
                  spk_utt_pos: torch.LongTensor,
    ) -> torch.FloatTensor:
        r"""
        last_hidden_state (torch.LongTensor) of shape (batch_size, sequence_length, n_dims):
            Output of the last layer of the encoder.
        spk_utt_pos (torch.LongTensor) of shape (batch_size, sequence_length,):
            metadata about the speaker tokens and utterance tokens
        Returns:
        Turn Level Supervised Constrastive Loss (torch.LongTensor)
        """
        batch_scl = 0
        for i in range(len(spk_utt_pos)):
            batch_element = spk_utt_pos[i]
            spk_utt_list = []
            spk_dict = {'start': 0, 'end': 0, 'spk_id': 0, 'bool': False}
            utt_dict = {'start': 0, 'end': 0, 'spk_id': 0, 'bool': False}
            for j in range(len(batch_element)):
                if batch_element[j] == 0 and j > 0:
                    utt_dict['end'] = j
                    utt_dict['bool'] = False
                    spk_utt_list.append({'spk': [spk_dict['start'], spk_dict['end'], spk_dict['spk_id']],
                                         'utt': [utt_dict['start'], utt_dict['end'], utt_dict['spk_id']]})
                    break
                if batch_element[j] > 0 and spk_dict['bool'] == False:
                    utt_dict['end'] = j
                    utt_dict['bool'] = False
                    if j > 1:
                        spk_utt_list.append({'spk': [spk_dict['start'], spk_dict['end'], spk_dict['spk_id']],
                                             'utt': [utt_dict['start'], utt_dict['end'], utt_dict['spk_id']]})
                    spk_dict['start'] = j
                    spk_dict['bool'] = True
                    spk_dict['spk_id'] = batch_element[j]
                    

                if batch_element[j] < 0 and spk_dict['bool'] == True:
                    spk_dict['end'] = j
                    spk_dict['bool'] = False
                    utt_dict['spk_id'] = spk_dict['spk_id']
                    utt_dict['start'] = j
                    utt_dict['bool'] = True
            # uniq spks
            if spk_utt_list[0]['spk'][2]==0:
                continue
            uniq_spks = list(set([int(dic['spk'][2].cpu()) for dic in spk_utt_list]))
            if len(uniq_spks)==1:
                continue
            # spk_utt_states
            spk_utt_states = {spk: [] for spk in uniq_spks}

            for spk in uniq_spks:
                for dic in spk_utt_list:
                    if spk == dic['utt'][2]:
                        mean_pool = torch.mean(last_hidden_state[i, dic['utt'][0]:dic['utt'][1]], 0)
                        spk_utt_states[spk].append(mean_pool)

            # positive samples
            L_pos = 0
            L_neg = 0
            for spk in uniq_spks:
                if len(spk_utt_states[spk]) > 1:
                    ids = random.choice(list(range(len(spk_utt_states[spk]))))
                    
                    spk_mean_exc = torch.mean(torch.vstack([spk_utt_states[spk][temp] for temp in range(len(spk_utt_states[spk])) if temp != ids]), 0)
                    
                    pos_mat_mul = torch.einsum('i, j->', spk_utt_states[spk][ids], spk_mean_exc)
                    pos_sigm = torch.sigmoid(pos_mat_mul)
                    pos_log = torch.log(pos_sigm)
                    L_pos += torch.sum(-1 * pos_log)

                    # negative sample

                    new_uniq_spks = uniq_spks.copy()
                    new_uniq_spks.remove(spk)
                    
                    spk2 = random.choice(new_uniq_spks)
                    id_neg = random.choice(list(range(len(spk_utt_states[spk2]))))
                    neg_mat_mul = torch.einsum('i, j->', spk_utt_states[spk2][id_neg], spk_mean_exc)
                    neg_sigm = torch.sigmoid(neg_mat_mul)
                    neg_log = torch.log(1 - neg_sigm+1e-5)
                    L_neg += torch.sum(-1 * neg_log)
                

            # print("L_neg", L_neg)
            
            batch_scl += L_pos
            batch_scl += L_neg
        batch_scl /= last_hidden_state.size(0)
        gc.collect()
        return batch_scl

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            spk_utt_pos: Optional[torch.Tensor] = None, ##changed here
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[List[torch.FloatTensor]] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if encoder_outputs is None:
            encoder = self.get_encoder()
            # TODO: mask the speaker names from the input IDs using the speaker pos info
            turn_attention_mask=None
            token_encoder_outputs=None
            tog_encoder_outputs=None
            
            if 'token' in self.SCLossesList:
                token_encoder_outputs = encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

            if 'turn' in self.SCLossesList or 'global' in self.SCLossesList:
                tog_attention_mask = torch.where(spk_utt_pos>0, 0, attention_mask)
                tog_encoder_outputs = encoder(
                    input_ids=input_ids,
                    attention_mask=tog_attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
        # if 'hidden_states' in encoder_outputs:
        #     print("encoder_outputs['last_hidden_state'].size(), encoder_outputs['hidden_states'].size()",
        #     encoder_outputs['last_hidden_state'].size(), encoder_outputs['hidden_states'].size())
        # else:
        #     print("encoder_outputs['last_hidden_state'].size()", encoder_outputs['last_hidden_state'].size())

        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        # added here
        sc_loss = 0
        if 'token' in self.SCLossesList and labels is not None:
            sc_loss += self.token_scl(last_hidden_state=token_encoder_outputs['last_hidden_state'], spk_utt_pos=spk_utt_pos)
            # print(sc_loss)
        if 'turn' in self.SCLossesList and labels is not None:
            sc_loss += self.turn_scl(last_hidden_state=tog_encoder_outputs['last_hidden_state'], spk_utt_pos=spk_utt_pos)
        
        if 'global' in self.SCLossesList and labels is not None:
            sc_loss += self.global_scl(last_hidden_state=tog_encoder_outputs['last_hidden_state'], spk_utt_pos=spk_utt_pos)
        
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss+(self.scl_coeff*sc_loss),) + output) if masked_lm_loss is not None else output
        loss = None
        if masked_lm_loss is None:
            loss = None
        else:
            loss = masked_lm_loss+(self.scl_coeff*sc_loss)
        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
