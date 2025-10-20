#!/bin/bash

# 设置基础路径
BASE_LOG_DIR="results/myidea/2WikiMultiHopQA/fakenum1/idealsetting"
BASE_DATA_DIR="/home/jiangjp/trace-idea/data/2wikimultihopqa"

echo "开始在单GPU上顺序执行实验..."
echo "开始时间: $(date)"

# 循环处理weight从0到0.9，间隔0.1
for i in {0..10}; do
     if [ $i -lt 10 ]; then
        continue
    fi
    # 计算当前weight值
    weight=$(echo "scale=1; $i * 0.1" | bc)
    
    # 格式化weight为两位数字符串
    weight_str=$(printf "w%02d" $((i)))
    
    # 创建对应的文件夹
    log_dir="${BASE_LOG_DIR}/${weight_str}"
    mkdir -p "$log_dir"
    
    # 生成时间戳
    timestamp=$(date +%Y%m%d_%H%M%S)
    
    # 生成输出和日志文件名
    output_file="${BASE_DATA_DIR}/wiki_dev100ideal_with_reasoning_chains_modify_${weight_str}.json"
    log_file="${log_dir}/wiki_construct_reasoning_chains_modify_log100_${timestamp}.log"
    
    echo "=================================="
    echo "正在执行第 $((i+1))/11 个任务"
    echo "Weight: ${weight} (${weight_str})"
    echo "开始时间: $(date)"
    echo "日志文件: ${log_file}"
    echo "输出文件: ${output_file}"
    echo "=================================="
    
    # 执行命令并等待完成
    python construct_reasoning_chains_modify.py \
      --dataset 2wikimultihopqa \
      --input_data_file "${BASE_DATA_DIR}/dev100_add_truthful_scores_with_kgs_final.json" \
      --save_data_file "${output_file}" \
      --calculate_ranked_prompt_indices \
      --fake_num 1 \
      --weight ${weight} \
      --max_chain_length 4 \
      > "${log_file}" 2>&1
    
    # 检查任务执行结果
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ Weight ${weight} 执行成功"
    else
        echo "❌ Weight ${weight} 执行失败，错误码: ${exit_code}"
    fi
    
    echo "完成时间: $(date)"
    echo ""
done

echo "🎉 所有任务已完成！"
echo "结束时间: $(date)"
