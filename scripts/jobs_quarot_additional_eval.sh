source /home/ylsung/codes/env.sh

nsamples_list=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096)
train_seqlen_list=(1048576 524288 262144 131072 65536 32768 16384 8192 4096 2048 1024 512 256)

# 3 to 12
# 3 6 9 12
nsamples_list=(128 256 512 1024 2048 8192 32768)
train_seqlen_list=(8192 4096 2048 1024 512 128 32)
val_size_list=(16 32 64 128 256 1024 4096)

declare -a jobs

source scripts/additional_eval_func.sh


save_names=(
    llama3-8b-instruct_quarot_3bit_w_n1024_l1024_input_magnitude                                                                                                         
    llama3-8b-instruct_quarot_3bit_w_n1024_l1024_input_magnitude_reverse                                                                                                 
    llama3-8b-instruct_quarot_3bit_w_n1024_l1024_masking_first                                                                                                           
    llama3-8b-instruct_quarot_3bit_w_n1024_l1024_masking_second                                                                                                          
    llama3-8b-instruct_quarot_3bit_w_n1024_l1024_org_attn_none3                                                                                                          
    llama3-8b-instruct_quarot_3bit_w_n1024_l1024_org_attn_none3_bin                                                                                                      
    llama3-8b-instruct_quarot_3bit_w_n1024_l1024_org_attn_none5                                                                                                          
    llama3-8b-instruct_quarot_3bit_w_n1024_l1024_org_attn_none5_bin                                                                                                      
    llama3-8b-instruct_quarot_3bit_w_n1024_l1024_org_attn_none_masking0.25
    llama3-8b-instruct_quarot_3bit_w_n1024_l1024_org_attn_none_masking0.5
    llama3-8b-instruct_quarot_3bit_w_n1024_l1024_org_attn_none_masking0.75
    llama3-8b-instruct_quarot_3bit_w_n1024_l1024_output_magnitude
    llama3-8b-instruct_quarot_3bit_w_n1024_l1024_output_magnitude_reverse
)

for _save_name in "${save_names[@]}"
do
    for seed in 0 1 2
    do
        save_name=${_save_name}@${seed}
        # no rotate
        
        mapfile new_jobs < <(add_additional_jobs "meta-llama/Meta-Llama-3-8B-Instruct" "llama3-8b-instruct" ${save_name})

        for item in "${new_jobs[@]}"; do
            echo $item
        done

        jobs+=("${new_jobs[@]}")

    done
done

# for i in 0 1 2 3 4 5
# do
#     nsamples=${nsamples_list[i]}
#     train_seqlen=${train_seqlen_list[i]}
#     val_size=${val_size_list[i]}
#     for w_bits in 2 3 4
#     do
#         for seed in 0 1 2
#         do
#             q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}@${seed}
#             save_name=llama3-8b-instruct_${q_type}
#             # no rotate
            
#             mapfile new_jobs < <(add_additional_jobs "meta-llama/Meta-Llama-3-8B-Instruct" "llama3-8b-instruct" ${save_name})

#             for item in "${new_jobs[@]}"; do
#                 echo $item
#             done

#             jobs+=("${new_jobs[@]}")

#         done
#     done
# done


for i in 0
do
    nsamples=${nsamples_list[i]}
    train_seqlen=${train_seqlen_list[i]}
    val_size=${val_size_list[i]}
    for w_bits in 3
    do
        for seed in 0
        do
            q_type=pajama_quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}@${seed}
            save_name=llama3-8b-instruct_${q_type}
            # no rotate
            
            mapfile new_jobs < <(add_additional_jobs "meta-llama/Meta-Llama-3-8B-Instruct" "llama3-8b-instruct" ${save_name})

            for item in "${new_jobs[@]}"; do
                echo $item
            done

            jobs+=("${new_jobs[@]}")

        done
    done
done


save_name=mistral-nemo-instruct_16bit
# no rotate

mapfile new_jobs < <(add_additional_jobs "mistralai/Mistral-Nemo-Instruct-2407" "mistral-nemo-instruct-2407" ${save_name})

for item in "${new_jobs[@]}"; do
    echo $item
done

jobs+=("${new_jobs[@]}")

for i in 0 1 2 3 4 5
do
    nsamples=${nsamples_list[i]}
    train_seqlen=${train_seqlen_list[i]}
    val_size=${val_size_list[i]}
    for w_bits in 3 4
    do
        for seed in 0 1 2
        do
            q_type=quarot_${w_bits}bit_w_n${nsamples}_l${train_seqlen}@${seed}
            save_name=mistral-nemo-instruct-2407_${q_type}
            # no rotate
            
            mapfile new_jobs < <(add_additional_jobs "mistralai/Mistral-Nemo-Instruct-2407" "mistral-nemo-instruct-2407" ${save_name})

            for item in "${new_jobs[@]}"; do
                echo $item
            done

            jobs+=("${new_jobs[@]}")

        done
    done
done
