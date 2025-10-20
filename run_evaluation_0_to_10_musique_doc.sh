#!/bin/bash

# 设置基础路径
BASE_LOG_DIR="results/myidea/MuSiQue/fakenum1/idealsetting"
BASE_DATA_DIR="/home/jiangjp/trace-idea/data/musique"

echo "开始在单GPU上顺序执行评估实验..."
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
    
    # 对应的输入文件（之前脚本生成的json文件）
    test_file="${BASE_DATA_DIR}/musique_dev100ideal_with_reasoning_chains_modify_${weight_str}.json"
    
    # 创建对应的日志文件夹
    log_dir="${BASE_LOG_DIR}/${weight_str}"
    
    # 生成时间戳
    timestamp=$(date +%Y%m%d_%H%M%S)
    
    # 生成日志文件名
    log_file="${log_dir}/evaluation_trace_docdev100ideal_log_${timestamp}.log"
    
    echo "=================================="
    echo "正在评估第 $((i+1))/11 个文件"
    echo "Weight: ${weight} (${weight_str})"
    echo "测试文件: ${test_file}"
    echo "日志文件: ${log_file}"
    echo "开始时间: $(date)"
    echo "=================================="
    
    # 检查输入文件是否存在
    if [ ! -f "$test_file" ]; then
        echo "❌ 输入文件不存在: $test_file"
        echo "跳过此评估任务"
        echo ""
        continue
    fi
    
    # 执行评估命令并等待完成（移除了nohup和&，改为顺序执行）
    python evaluation.py \
      --test_file "${test_file}" \
      --reader llama3 \
      --context_type documents \
      --n_context 5 \
      > "${log_file}" 2>&1
    
    # 检查任务执行结果
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ Weight ${weight} 评估成功"
    else
        echo "❌ Weight ${weight} 评估失败，错误码: ${exit_code}"
    fi
    
    echo "完成时间: $(date)"
    echo ""
done

echo "🎉 所有评估任务已完成！"
echo "结束时间: $(date)"
