#!/bin/bash

# 设置基础路径
BASE_DIR="/home/jiangjp/trace-idea"
DATA_BASE_DIR="${BASE_DIR}/data"
RESULTS_BASE_DIR="${BASE_DIR}/results/myidea"

echo "开始执行所有评估实验..."
echo "总计: 33个数据集 × 2种context_type = 66个实验"
echo "开始时间: $(date)"

# 定义数据集信息
declare -A datasets=(
    ["hotpotqa"]="HotPotQA"
    ["2wikimultihopqa"]="2WikiMultiHopQA"
    ["musique"]="MuSiQue"
)

declare -A file_patterns=(
    ["hotpotqa"]="hotpotqa_dev100ideal_with_reasoning_chains_modify"
    ["2wikimultihopqa"]="wiki_dev100ideal_with_reasoning_chains_modify"
    ["musique"]="musique_dev100ideal_with_reasoning_chains_modify"
)

# 定义context_type信息
declare -A context_types=(
    ["triples"]="tripledev100idea"
    ["documents"]="docdev100idea"
)

# 实验计数器
experiment_count=0
total_experiments=66

# 循环处理每个数据集
for dataset in hotpotqa 2wikimultihopqa musique; do
    dataset_display_name="${datasets[$dataset]}"
    file_pattern="${file_patterns[$dataset]}"

    echo ""
    echo "========================================================"
    echo "开始处理数据集: $dataset ($dataset_display_name)"
    echo "========================================================"

    # 循环处理每个权重文件 (w00 到 w10)
    for i in {0..10}; do
        # 格式化weight为两位数字符串
        weight_str=$(printf "w%02d" $i)

        # 构造输入文件路径
        test_file="${DATA_BASE_DIR}/${dataset}/${file_pattern}_${weight_str}.json"

        # 循环处理两种context_type
        for context_type in triples documents; do
            experiment_count=$((experiment_count + 1))
            log_suffix="${context_types[$context_type]}"

            # 构造输出目录和保存目录
            save_ie_dir="${RESULTS_BASE_DIR}/${dataset_display_name}/fakenum1/idealsetting/${weight_str}"
            log_dir="${save_ie_dir}"

            # 生成时间戳
            timestamp=$(date +%Y%m%d_%H%M%S)

            # 生成日志文件名
            log_file="${log_dir}/evaluation_getIE_wrong_${log_suffix}_log_${timestamp}.log"

            echo ""
            echo "=================================="
            echo "实验进度: ${experiment_count}/${total_experiments}"
            echo "数据集: $dataset"
            echo "权重: ${weight_str}"
            echo "Context Type: ${context_type}"
            echo "测试文件: ${test_file}"
            echo "保存目录: ${save_ie_dir}"
            echo "日志文件: ${log_file}"
            echo "开始时间: $(date)"
            echo "=================================="

            # 检查输入文件是否存在
            if [ ! -f "$test_file" ]; then
                echo "❌ 输入文件不存在: $test_file"
                echo "跳过此实验"
                continue
            fi

            # 创建输出目录
            mkdir -p "$log_dir"

            # 执行评估命令并等待完成
            python evaluation_getIE_wrong.py \
              --test_file "$test_file" \
              --reader llama3 \
              --context_type "$context_type" \
              --save_IE_file "$save_ie_dir" \
              --n_context 5 \
              > "$log_file" 2>&1

            # 检查任务执行结果
            exit_code=$?
            if [ $exit_code -eq 0 ]; then
                echo "✅ 实验 ${experiment_count} 执行成功"
                echo "   数据集: $dataset, 权重: ${weight_str}, Context: ${context_type}"
            else
                echo "❌ 实验 ${experiment_count} 执行失败，错误码: ${exit_code}"
                echo "   数据集: $dataset, 权重: ${weight_str}, Context: ${context_type}"
            fi

            echo "完成时间: $(date)"
        done
    done

    echo "========================================================"
    echo "数据集 $dataset 处理完成"
    echo "========================================================"
done

echo ""
echo "🎉 所有评估实验已完成！"
echo "总执行实验数: ${experiment_count}"
echo "结束时间: $(date)"

