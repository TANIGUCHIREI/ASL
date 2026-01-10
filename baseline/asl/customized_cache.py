from transformers.cache_utils import Cache,DynamicCache
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from baseline.asl.adaptive_select_layer_utils import get_top_p_indices,get_rank_descending,relative_normalized_variance,repeat_kv
class DynamicCache(DynamicCache):
    def __init__(self, num_hidden_layers: Optional[int] = None) -> None:
        super().__init__()
        if num_hidden_layers is None:
            self.key_cache: List[torch.Tensor] = []
            self.value_cache: List[torch.Tensor] = []
            self.prefilling_score_cache =[]
            self.value_norm_cache =[]

            
        else:
            self.key_cache: List[torch.Tensor] = [[] for _ in range(num_hidden_layers)]
            self.value_cache: List[torch.Tensor] = [[] for _ in range(num_hidden_layers)]
            self.prefilling_score_cache = [None for _ in range(num_hidden_layers)]
            self.value_norm_cache = [None for _ in range(num_hidden_layers)]
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen


        #cache
        self.initial_variance = None
        self.select_idx = None
        self.args_initialized = False

        #args
        self.use_layer_num=None
        self.L_min = None
        self.select_layer = None
        self.layer_th=None
        self.select_k=None
        self.window_size=None
        self.kernel_size=None
        self.pooling = None
        self.cache_k = None
        self.cache_p = None
        self.done_autoset = False
        self.mlp_chunk_size = 32768 
        self.enable_Full_Head_top = False
        # MMP calcuation requires huge amount of VRAM and causes Out of memory despite KV Cache size is relatively small.
        # For avoiding OoM, we split mlp input tensors into fixed chunk size and finally concat after calcuation.
        

        #for time counting
        self.total_seqlen = 0
        self.init_time= time.time()
        self.analysis_data = {"prefill_time":None,"output_time":None,"input_length":None,"output_length":None,"allocated_memory":None,"select_layer":None}
        
        #for top-p varlen update
        self.varlen_key_cache: List[List[torch.Tensor]] = []
        self.varlen_value_cache: List[List[torch.Tensor]] = []



    def set_cache_kwargs(self,cache_kwargs):
        if self.args_initialized==False:
            self.select_layer = cache_kwargs.default_select_layer
            self.use_layer_num = cache_kwargs.use_layer_num
            self.layer_th=cache_kwargs.layer_th
            self.select_k=cache_kwargs.select_k
            self.window_size=cache_kwargs.window_size
            self.kernel_size=cache_kwargs.kernel_size
            self.L_min = cache_kwargs.default_select_layer
            self.pooling = cache_kwargs.pooling
            self.cache_k = cache_kwargs.cache_k
            self.cache_p = cache_kwargs.cache_p
            
            self.args_initialized = True

    def position_update(self,position_ids, position_embeddings,layer_idx,_update=False):
        if layer_idx==0 or _update:
            self.position_ids=position_ids
            self.position_embeddings = position_embeddings
        return self.position_ids,self.position_embeddings

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            if self._seen_tokens!=0: 
                self.total_seqlen += key_states.shape[-2]
            if self.analysis_data["prefill_time"]==None and key_states.shape[-2]==1:
                self.analysis_data["prefill_time"] = time.time() -  self.init_time 
            self._seen_tokens += key_states.shape[-2]


        # Update the cache
        if cache_kwargs!=None and "selected_cache_indices" in cache_kwargs and cache_kwargs["selected_cache_indices"]!=None or self.varlen_key_cache!=[]:
            if self.cache_p or self.enable_Full_Head_top:
                #top-p(varlen update)
                selected_cache_indices = None if "selected_cache_indices" not in cache_kwargs else cache_kwargs["selected_cache_indices"]
                return self.varlen_update(key_states,value_states,layer_idx,selected_cache_indices) 
            elif self.cache_k:
                #top-k
                selected_cache_indices=cache_kwargs["selected_cache_indices"].unsqueeze(-1).expand(-1, -1, -1, key_states.size(-1))
                key_states = torch.gather(key_states, dim=2,index=selected_cache_indices)
                value_states = torch.gather(value_states, dim=2,index=selected_cache_indices)

        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        # content on layer cache can be a tensor and checking not tensor causes errors
        # so we explicitly check for the empty list
        elif self.key_cache[layer_idx] == []:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def varlen_update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        selected_cache_indices,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the cache

        if len(self.varlen_key_cache) <= layer_idx:
            #Prefill Phase
            self.varlen_key_cache.append([ key_states[:,h,indices] for h,indices in enumerate(selected_cache_indices)])
            self.varlen_value_cache.append([ value_states[:,h,indices] for h,indices in enumerate(selected_cache_indices)])
        else:
            #Generate Phase
            bsz,num_key_value_heads,q_len,head_dim = key_states.shape
            for h in range(num_key_value_heads):
                self.varlen_key_cache[layer_idx][h] = torch.cat([self.varlen_key_cache[layer_idx][h], key_states[:,h]], dim=-2)
                self.varlen_value_cache[layer_idx][h] = torch.cat([self.varlen_value_cache[layer_idx][h], value_states[:,h]], dim=-2)
        return self.varlen_key_cache[layer_idx], self.varlen_value_cache[layer_idx]


    def sum_attn_cache(self, key_states, query_states, window_size):
        num_key_value_groups = query_states.shape[1]//key_states.shape[1]
        bsz, num_heads, q_len, head_dim = query_states.shape 
        key_states_temp = repeat_kv(key_states, num_key_value_groups)
        attn_weights = torch.matmul(query_states[..., -window_size:, :], key_states_temp.transpose(2, 3)) / math.sqrt(head_dim)
        mask = torch.full((window_size, window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -window_size:, -window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32) #[bs,head_num,window_size,seqlen]
        attn_weights_sum = attn_weights[..., : -window_size].sum(dim = -2) #[bs,head_num,window_size,seqlen-window_size] sum in GQA groups
        if self.pooling == 'avgpool':
            pooled_attn = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        elif self.pooling == 'maxpool':
            pooled_attn = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
        else:
            raise ValueError('Pooling method not supported')

        kv_compress_indices = self.compress_kv(attn_weights,pooled_attn,key_states,query_states)
        pooled_attn = pooled_attn.view(bsz, -1, num_key_value_groups, q_len-window_size).sum(dim=-2) 
        pooled_attn= pooled_attn.sum(dim=-2) #[1,seqlen] sum in num_key_value_groups
        ranks =get_rank_descending(pooled_attn).to(torch.float) #[layer_num, seqlen] 
        return ranks,kv_compress_indices

    def compress_kv(self, attn_weights=None,pooled_attn=None,key_states=None,query_states=None):
        #calc faseter if atten_weights or pooled attn have been already calcuated and inputted as argument.
        num_key_value_groups = query_states.shape[1]//key_states.shape[1]
        bsz, query_head_num, q_len, head_dim = query_states.shape 
        kv_head_num = query_head_num//num_key_value_groups
        window_indices = torch.arange(q_len - self.window_size, q_len)
        if attn_weights==None:
            key_states_temp = repeat_kv(key_states, num_key_value_groups)
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states_temp.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)  #[bs,head_num,window_size,seqlen]
        if self.cache_p:
            if self.select_k and q_len <= self.select_k:
                #stop using top-p if select_k > seqlen
                selected_cache_indices=torch.arange(0,q_len).view(1,-1).repeat(kv_head_num,1).tolist()
                k_per_batch= torch.tensor([q_len]).repeat(kv_head_num)
            else:
                attn_cache = attn_weights.view(-1, self.window_size,num_key_value_groups,q_len).mean(dim=-2,keepdim=False) #[kv_head_num,window_size,seqlen]
                attn_cache = attn_cache.mean(dim=-2,keepdim=False) #[kv_head_num,seqlen] ...for top-p GQA
                sel_sorted, k_per_batch = get_top_p_indices(attn_cache,top_p=self.cache_p,_is_softmaxed=True)
                attention_sink = torch.zeros(1, device=sel_sorted.device,dtype=window_indices.dtype)
                selected_cache_indices =[]
                window_indices = window_indices.to(sel_sorted)
                for h,indicies in enumerate(sel_sorted):
                    valid_indices = torch.cat([attention_sink,indicies[indicies != -1],window_indices]) #add window tokens
                    valid_indices = valid_indices.unique() 
                    selected_cache_indices.append(valid_indices)
                
            # print(f"{k_per_batch=}")
        elif self.cache_k:
            # similar to SnapKV/FastKV
            if pooled_attn==None:
                attn_weights_sum = attn_weights[..., : -self.window_size].sum(dim = -2) #[bs,head_num,window_size,seqlen-window_size]
                if self.pooling == 'avgpool':
                    pooled_attn = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                elif self.pooling == 'maxpool':
                    pooled_attn = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
                else:
                    raise ValueError('Pooling method not supported')
            kv_head_num = pooled_attn.shape[-2]//num_key_value_groups
            pooled_attn = pooled_attn.view(1, kv_head_num, num_key_value_groups, q_len-self.window_size).sum(dim=-2) #[bs,kv_head_num,q_len-self.window_size]
            k= self.cache_k - self.window_size if q_len > self.cache_k else q_len-self.window_size #for avoiding RuntimeError: selected index k out of range
            selected_cache_indices = pooled_attn.topk(int(k) , dim=-1).indices #[bs,kv_head_num,k-window_size]
            selected_cache_indices = torch.cat([selected_cache_indices,window_indices.view(1,1,-1).expand(-1,kv_head_num,-1).to(pooled_attn.device)],dim=-1) #[bs,kv_head_num,k]
        else:
            selected_cache_indices=None
        return selected_cache_indices
    
    
    def prefilling_score_update(
        self,
        query_states:torch.Tensor,
        key_states:torch.Tensor,
        value_states:torch.Tensor, #not used
        layer_idx: int,
        just_for_analysis=False #use this for other method(pyramidinfer, gemfilter ... etc) if true, this function will nothing to save input length.
    ):
        #for time counting
        if just_for_analysis: 
            if layer_idx == 0 and self.analysis_data["input_length"]==None:
                self.analysis_data["input_length"]=query_states.shape[-2]
            return 0
        
        if layer_idx == 0:
            if self.analysis_data["input_length"]==None:
                self.analysis_data["input_length"]=query_states.shape[-2]
            self.total_seqlen = query_states.shape[-2]
        if len(self.prefilling_score_cache) <= layer_idx:
            self.prefilling_score_cache.append(None)
            self.value_norm_cache.append(None)


        if (self.cache_k or self.cache_p) and self.layer_th==None and self.select_layer==None:
            return self.compress_kv(None,None,key_states,query_states)

        if layer_idx < len(self.prefilling_score_cache) and self.prefilling_score_cache[layer_idx] !=None or self.select_layer==None:
            #work as vanilla
            return None 
        elif self.done_autoset:
            return self.compress_kv(None,None,key_states,query_states)
        elif self.layer_th==None:
            if layer_idx==self.select_layer:
                #token selection without ASL
                ranks,kv_compress_indices=self.sum_attn_cache(key_states,query_states,self.window_size)
                seqlen = ranks.shape[-1]
                k=self.select_k - self.window_size if seqlen > self.select_k - self.window_size else seqlen #for avoid RuntimeError: selected index k out of range
                select_idx = torch.topk(ranks,dim=-1,k=k,largest=False)[1][0]
                window_indices = torch.arange(query_states.shape[-2] - self.window_size, query_states.shape[-2], device=query_states.device)
                select_idx = torch.cat([select_idx,window_indices]).unique().view(1,-1) #add window indices
                self.select_idx = select_idx
                return kv_compress_indices
            else:
                return self.compress_kv(None,None,key_states,query_states)

        sum_attn_cacche,kv_compress_indices = self.sum_attn_cache(key_states, query_states, self.window_size)
        self.prefilling_score_cache[layer_idx] = sum_attn_cacche
        
        select_idx = self.selecting_select_layer(layer_idx)
        self.select_idx = select_idx
        
        return kv_compress_indices

    def selecting_select_layer(self,layer_idx,L_min=None,top_k=1024):
        L_min = self.L_min
        L_obs = self.use_layer_num
        top_k = self.select_k
        if layer_idx < L_min:
            return None
            # raise ValueError(f"Calcuating layer_th needs at least {L_min} layer's output data, but only {layer_idx} data are inputted.")
        mean_socre = self.prefilling_score_cache[layer_idx] #[output_len,total_len]
        ranks = torch.stack([self.prefilling_score_cache[l][0].to(mean_socre) for l in range(layer_idx-L_obs+1,layer_idx+1)]) #[layer_num,seqlen]
        seqlen= mean_socre.shape[-1]
        k=top_k - self.window_size if seqlen > top_k - self.window_size else seqlen #for avoid RuntimeError: selected index k out of range
        topk_idx_per_layers = torch.topk(ranks, k=int(k),dim=-1,largest=False)[1]
        latest_layer_sorted_indices = torch.unique(topk_idx_per_layers.view(-1))
        # normalized_ratio = normalized_variance(ranks[:,latest_layer_sorted_indices],N=max_N)
        normalized_ratio,self.initial_variance = relative_normalized_variance(ranks[-L_obs:,latest_layer_sorted_indices],self.initial_variance)
        if normalized_ratio < self.layer_th:
            self.select_layer = layer_idx
            self.done_autoset = True
            print(F"select layer is set as: {layer_idx}")
            window_indices = torch.arange(mean_socre.shape[-1], mean_socre.shape[-1] + self.window_size, device=topk_idx_per_layers.device) #add window_indcies
            select_idx = torch.cat([topk_idx_per_layers[-1],window_indices]).unique().view(1,-1)
            self.prefilling_score_cache = [] #release Cache
            return select_idx
        else:
            return None

    def get_analysis_data(self):
        self.analysis_data["output_time"] = time.time() - self.init_time
        self.analysis_data["output_length"] = self.total_seqlen - self.analysis_data["input_length"]
        if (self.layer_th and self.done_autoset)or (not self.layer_th and self.select_layer):
            self.analysis_data["select_layer"] = self.select_layer
        allocated=0
        for i in range(torch.cuda.device_count()):
            allocated += torch.cuda.memory_allocated(i) #add allocated memory size
        
        self.analysis_data["allocated_memory"] = allocated
        return self.analysis_data