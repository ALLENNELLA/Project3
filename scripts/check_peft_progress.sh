#!/bin/bash
# 快速查看 PEFT 实验进度

cd /root/25S151115/project3/scripts

echo "📊 PEFT 实验实时进度"
echo "=========================================="

# 检查主脚本
if ps aux | grep -q "[r]un_peft_from_saved_indices"; then
    echo "✅ 主脚本正在运行"
else
    echo "❌ 主脚本未运行"
fi

# 统计任务
total=$(find outputs/automated_experiments/logs -name "*_peft.log" | wc -l)
success=$(grep -l "✅\|completed\|Finished" outputs/automated_experiments/logs/*_peft.log 2>/dev/null | wc -l)
failed=$(grep -l "Traceback\|FAIL" outputs/automated_experiments/logs/*_peft.log 2>/dev/null | grep -v "out of memory" | wc -l)
running=$(ps aux | grep "[r]un_finetune_from_config" | wc -l)

echo "总任务数: $total"
echo "✅ 成功: $success"
echo "🔄 运行中: $running"
echo "❌ 失败: $failed"
echo "完成率: $(( success * 100 / total ))%" 
echo "=========================================="

# GPU 状态
echo ""
echo "🖥️  GPU 2-6 使用情况:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
awk -F', ' 'NR>=3 && NR<=7 {gpu=$1; used=$2; total=$3; util=$4; pct=int(used*100/total); printf "  GPU %s: %d/%d MB (%d%%), Util: %s%%\n", gpu, used, total, pct, util}'
