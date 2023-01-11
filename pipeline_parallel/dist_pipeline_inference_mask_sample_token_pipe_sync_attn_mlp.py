import time
import json
import torch.nn.functional
from comm.comm_utils import *
from modules.generation_utils import get_logits_processor, get_logits_warper


from .dist_pipeline_inference_mask_greedy_token_pipe_sync import DistGreedyInferenceMaskTokenPipeSync


class DistSampleInferenceMaskTokenPipeSyncAttnMLP(DistGreedyInferenceMaskTokenPipeSync):
    
    def __init__(self, args, device, rank=None, be_coordinated=False):
        super().__init__(args, device, rank=rank, be_coordinated=be_coordinated)
        self.update_processors(args)
        
    def update_processors(self, args):
        self.logits_processor = get_logits_processor()
        self.logits_warper = get_logits_warper(
            top_k = (None if args.top_k is None or args.top_k == 0 else args.top_k),
            top_p = (None if args.top_p is None or args.top_p <= 0 else args.top_p),
            temperature = args.temperature,
            num_beams = 1,
        )
        
    def _get_embedding_size(self):
        if self.model_type == 'gpt2':
            from modules.hf_gpt2_module import GPTConfig
            config = GPTConfig.from_pretrained(self.model_name)
            return config.n_embd
        elif self.model_type == 'gptj':
            from modules.hf_gptj_module import GPTConfig
            config = GPTConfig.from_pretrained(self.model_name)
            return config.n_embd
        elif self.model_type == 'gptneox':
            from modules.hf_gptneox_module import GPTConfig
            config = GPTConfig.from_pretrained(self.model_name)
            return config.hidden_size
        elif self.model_type in ['opt', 'opt-save', 'opt-flash', 'opt-flash-sparse', 'opt-baseline', 'opt-sparse', 'opt-classifier-sparse','opt-classifier-sparse-topk','opt-classifier-sparse-bylayer', 'opt-attn-mlp']:
            from modules.hf_opt_module import GPTConfig
            config = GPTConfig.from_pretrained(self.model_name)
            return config.hidden_size
        elif self.model_type in ['bloom', 'bloom-sparse', 'bloom-relu']:
            from modules.hf_bloom_module import GPTConfig
            config = GPTConfig.from_pretrained(self.model_name)
            return config.hidden_size
        elif self.model_type == 'yalm':
            from modules.yalm_module import GPTConfig
            config = GPTConfig.from_pretrained(self.model_name)
            return config.hidden_size
        elif self.model_type == 'glm':
            from modules.glm_module import GPTConfig
            config = GPTConfig.from_pretrained(self.model_name)
            return config.hidden_size
        else:
            raise Exception(f'unknown model type {self.model_type}')

    def _create_layers(self):
        if self.model_type == 'gpt2':
            from modules.hf_gpt2_module import GPTEmbeddings, GPTBlock, GPTLMHead
        elif self.model_type == 'gptj':
            from modules.hf_gptj_module import GPTEmbeddings, GPTBlock, GPTLMHead
        elif self.model_type == 'gptneox':
            from modules.hf_gptneox_module import GPTEmbeddings, GPTBlock, GPTLMHead
        elif self.model_type == 'opt':
            from modules.hf_opt_module import GPTEmbeddings, GPTBlock, GPTLMHead
        elif self.model_type == 'opt-save':
            from modules.hf_opt_module_save import GPTEmbeddings, GPTBlock, GPTLMHead
        elif self.model_type == 'opt-flash':
            from modules.hf_opt_flash_module import GPTEmbeddings, GPTBlock, GPTLMHead
        elif self.model_type == 'opt-flash-sparse':
            from modules.hf_opt_flash_sparse_module import GPTEmbeddings, GPTBlock, GPTLMHead
        elif self.model_type == 'opt-baseline':
            from modules.hf_opt_baseline_module import GPTEmbeddings, GPTBlock, GPTLMHead
        elif self.model_type == 'opt-attn-mlp':
            from modules.hf_opt_baseline_module_attn_mlp import GPTEmbeddings, GPTBlock, GPTLMHead
        elif self.model_type == 'opt-sparse':
            from modules.hf_opt_sparse_module import GPTEmbeddings, GPTBlock, GPTLMHead
        elif self.model_type == 'opt-classifier-sparse':
            from modules.hf_opt_sparse_classifier import GPTEmbeddings, GPTBlock, GPTLMHead
        elif self.model_type == 'opt-classifier-sparse-topk':
            from modules.hf_opt_sparse_classifier_topk import GPTEmbeddings, GPTBlock, GPTLMHead
        elif self.model_type == 'opt-classifier-sparse-bylayer':
            from modules.hf_opt_sparse_classifier_topk_bylayer import GPTEmbeddings, GPTBlock, GPTLMHead
        elif self.model_type == 'bloom':
            from modules.hf_bloom_module import GPTEmbeddings, GPTBlock, GPTLMHead
        elif self.model_type == 'bloom-sparse':
            from modules.hf_bloom_sparse_module import GPTEmbeddings, GPTBlock, GPTLMHead
        elif self.model_type == 'bloom-relu':
            from modules.hf_bloom_relu_module import GPTEmbeddings, GPTBlock, GPTLMHead
        elif self.model_type == 'yalm':
            from modules.yalm_module import GPTEmbeddings, GPTBlock, GPTLMHead
        elif self.model_type == 'glm':
            from modules.glm_module import GPTEmbeddings, GPTBlock, GPTLMHead
            GPTBlock.echo_prompt = self.echo_prompt
        else:
            raise Exception(f'unknown model type {self.model_type}')
        
        if self.pp_rank == 0:
            self.layers['emb'] = GPTEmbeddings.from_pretrained(
                self.model_name
            ).to(self.dtype).eval().to(self.device)
        for layer_index in range(self.num_layers):
            # global layer indexing could be an argument
            global_layer_index = self.num_layers * self.pp_rank + layer_index
            if self.max_layers is not None and global_layer_index >= self.max_layers:
                # TODO: this is a hack
                self.num_layers = layer_index
                break
            print(f'loading layer {global_layer_index}')
            self.layers['block'+str(layer_index)] = GPTBlock.from_pretrained(
                self.model_name, layer_index=global_layer_index
            ).to(self.dtype).eval().to(self.device)
            if self.coord_client:
                self.coord_client.update_status('running', returned_payload={
                    'rank': self.pp_rank, 'loaded_layer': layer_index, 'total_layer': self.num_layers})
        if self.pp_rank == self.pipeline_group_size - 1:
            self.layers['lm'] = GPTLMHead.from_pretrained(
                self.model_name
            ).to(self.dtype).eval().to(self.device)

    def _generate_new_token(self, index):
        assert self.pp_rank == self.pipeline_group_size - 1
        z = self.layers['lm'](self.output_token_emb[index])
        if torch.isnan(z).any():
            print('containing nan, setting them to zero!')
            print(z)
        z = z.float().nan_to_num() # test if fp32 whether cause inf
        z = torch.nn.functional.log_softmax(z, -1)
        
        if self.top_k_per_token > 0:
            logprobs, indices = z.topk(k=self.top_k_per_token, dim=-1)
            self.ret_topk_tokens[
                index*self.token_micro_batch_size*self.num_completions:(index+1)*self.token_micro_batch_size*self.num_completions,
                self.i_current_token
            ] = indices.squeeze(1)
            self.ret_topk_token_logprobs[
                index*self.token_micro_batch_size*self.num_completions:(index+1)*self.token_micro_batch_size*self.num_completions,
                self.i_current_token
            ] = logprobs.squeeze(1)
            
        # [:, -1] because multinomial only accept 1/2d tensors
        z_to_sample = z[:, -1] # bs, vocab
        z_to_sample = self.logits_warper(None, z_to_sample)
        p_to_sample = z_to_sample.softmax(-1).clamp(0, 1).nan_to_num() 
        indices = torch.multinomial(p_to_sample, num_samples=1) # bs, 1
        logprobs = torch.gather(z[:, -1], -1, indices) # bs, 1
        self.send_new_tokens[index] = indices
        
        self.ret_tokens[
            index*self.token_micro_batch_size*self.num_completions:(index+1)*self.token_micro_batch_size*self.num_completions,
            self.i_current_token
        ] = indices.squeeze(-1)
        self.ret_token_logprobs[
            index*self.token_micro_batch_size*self.num_completions:(index+1)*self.token_micro_batch_size*self.num_completions,
            self.i_current_token
        ] = logprobs.squeeze(-1)
        
        if index == self.token_micro_batch_num - 1:
            self.i_current_token += 1
            
    def _forward_compute_prompt_seq(self, index, seq, mask):
        print("Compute prompt seq<", index, ">.")
        if self.pp_rank == 0:
            self.input_seq_emb[index] = self.layers['emb'](seq, mask=mask)
        current_emb = self.input_seq_emb[index]
        caches = [None] * self.num_layers
        
        # when disabled, this will do nothing
        mask, current_emb, caches = self.share_prefix.process_inputs(
            seq, mask, current_emb, caches)
        
        assert self.num_layers % 2 == 0

        for count in range(self.num_layers // 2):
            
            layer_index = count * 2
            layer_index_next = count * 2 + 1
            
            if self.layers['block' + str(layer_index)].layer_index > 0:
                
                print('parallel attn and mlp')
                
                input_emb = current_emb.clone()
            
                # attn runs
                output_emb, caches[layer_index] = \
                    self.layers['block' + str(layer_index)](input_emb, caches[layer_index], mask=mask, subnet='attn')
                self.cached_attention[layer_index][index] = caches[layer_index]
                
                # add
                current_emb += output_emb - input_emb

                output_emb, caches[layer_index_next] = \
                    self.layers['block' + str(layer_index_next)](input_emb, caches[layer_index_next], mask=mask, subnet='attn')
                self.cached_attention[layer_index_next][index] = caches[layer_index_next]

                # add
                current_emb += output_emb - input_emb

                # mlp runs
                output_emb, _ = \
                    self.layers['block' + str(layer_index)](input_emb, caches[layer_index], mask=mask, subnet='mlp')
                
                current_emb += output_emb - input_emb

                output_emb, _ = \
                    self.layers['block' + str(layer_index_next)](input_emb, caches[layer_index_next], mask=mask, subnet='mlp')

                # add
                current_emb += output_emb - input_emb


            else:
                
                input_emb = current_emb.clone()
                
                # attn runs
                output_emb, caches[layer_index] = \
                    self.layers['block' + str(layer_index)](input_emb, caches[layer_index], mask=mask, subnet='attn')
                self.cached_attention[layer_index][index] = caches[layer_index]

                output_emb, _ = \
                    self.layers['block' + str(layer_index)](output_emb, caches[layer_index], mask=mask, subnet='mlp')

                current_emb += output_emb - input_emb

                # mlp runs

                input_emb = current_emb.clone()

                output_emb, caches[layer_index_next] = \
                    self.layers['block' + str(layer_index_next)](input_emb, caches[layer_index_next], mask=mask, subnet='attn')
                self.cached_attention[layer_index_next][index] = caches[layer_index_next]

                output_emb, _ = \
                    self.layers['block' + str(layer_index_next)](output_emb, caches[layer_index_next], mask=mask, subnet='mlp')

                current_emb += output_emb - input_emb
            
#             if self.layers['block' + str(layer_index)].layer_index > 0:
                
#                 print('parallel attn and mlp')
                
#                 input_emb = current_emb.clone()
            
#                 # attn runs
#                 output_emb, caches[layer_index] = \
#                     self.layers['block' + str(layer_index)](input_emb, caches[layer_index], mask=mask, subnet='attn')
#                 self.cached_attention[layer_index][index] = caches[layer_index]

#                 output_emb, caches[layer_index_next] = \
#                     self.layers['block' + str(layer_index_next)](output_emb, caches[layer_index_next], mask=mask, subnet='attn')
#                 self.cached_attention[layer_index_next][index] = caches[layer_index_next]

#                 current_emb += output_emb - input_emb
                
#                 # input_emb = current_emb.clone()

#                 # mlp runs
#                 output_emb, _ = \
#                     self.layers['block' + str(layer_index)](input_emb, caches[layer_index], mask=mask, subnet='mlp')

#                 output_emb, _ = \
#                     self.layers['block' + str(layer_index_next)](output_emb, caches[layer_index_next], mask=mask, subnet='mlp')

#                 current_emb += output_emb - input_emb


#             else:
                
#                 input_emb = current_emb.clone()
                
#                 # attn runs
#                 output_emb, caches[layer_index] = \
#                     self.layers['block' + str(layer_index)](input_emb, caches[layer_index], mask=mask, subnet='attn')
#                 self.cached_attention[layer_index][index] = caches[layer_index]

#                 output_emb, _ = \
#                     self.layers['block' + str(layer_index)](output_emb, caches[layer_index], mask=mask, subnet='mlp')

#                 current_emb += output_emb - input_emb

#                 # mlp runs

#                 input_emb = current_emb.clone()

#                 output_emb, caches[layer_index_next] = \
#                     self.layers['block' + str(layer_index_next)](input_emb, caches[layer_index_next], mask=mask, subnet='attn')
#                 self.cached_attention[layer_index_next][index] = caches[layer_index_next]

#                 output_emb, _ = \
#                     self.layers['block' + str(layer_index_next)](output_emb, caches[layer_index_next], mask=mask, subnet='mlp')

#                 current_emb += output_emb - input_emb
            

        # when disabled, this will do nothing
        current_emb = self.share_prefix.process_outputs(
            seq, mask, current_emb, caches)
        
        self.output_seq_emb[index] = current_emb
        
        if self.pp_rank == self.pipeline_group_size - 1:
            self._copy_initial_token_emb(index)
            
            if self.echo_prompt:
                self._generate_echo_token_logprobs(index, indices=seq)