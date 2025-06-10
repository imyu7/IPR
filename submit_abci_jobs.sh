#!/bin/bash
# ABCI 3.0用のWebShop評価バッチジョブスクリプト

# デフォルト設定
TOTAL_TASKS=200          # 総タスク数
TASKS_PER_JOB=5        # ジョブあたりのタスク数
CONFIG_FILE="webshop_evaluator/configs/webshop_llama_3_2_3B.yaml"  # 設定ファイル
RESULTS_DIR="webshop_evaluator/results"  # 結果保存ディレクトリ

# ABCI設定
GROUP="gcb50389"               # ABCIグループID（必須）
RESOURCE_TYPE=""       # 資源タイプ（設定ファイルのagent typeに基づいて自動設定）
SELECT_COUNT=1         # 資源数
WALLTIME="2:00:00"     # 最大実行時間

# コマンドライン引数の処理
while [[ $# -gt 0 ]]; do
    case $1 in
        --total-tasks)
            TOTAL_TASKS="$2"
            shift 2
            ;;
        --tasks-per-job)
            TASKS_PER_JOB="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --group)
            GROUP="$2"
            shift 2
            ;;
        --resource-type)
            RESOURCE_TYPE="$2"
            shift 2
            ;;
        --walltime)
            WALLTIME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --total-tasks NUM        総タスク数 (default: $TOTAL_TASKS)"
            echo "  --tasks-per-job NUM      ジョブあたりのタスク数 (default: $TASKS_PER_JOB)"
            echo "  --config FILE            設定ファイル (default: $CONFIG_FILE)"
            echo "  --group GROUP            ABCIグループID (default: $GROUP)"
            echo "  --resource-type TYPE     資源タイプ (default: $RESOURCE_TYPE)"
            echo "  --walltime TIME          最大実行時間 (default: $WALLTIME)"
            echo "  --help                   このヘルプを表示"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# 必須パラメータのチェック
if [ -z "$GROUP" ]; then
    echo "❌ エラー: ABCIグループIDが指定されていません。--group オプションで指定してください。"
    echo "例: --group gaa12345"
    exit 1
fi

# 設定ファイルの存在確認
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ エラー: 設定ファイルが見つかりません: $CONFIG_FILE"
    echo "利用可能な設定ファイル:"
    ls -la webshop_evaluator/configs/*.yaml 2>/dev/null || echo "  webshop_evaluator/configs/ディレクトリに設定ファイルがありません"
    exit 1
fi

# 設定ファイルからtask_end_idxを読み取り（参考情報として）
CONFIG_TASK_END_IDX=$(grep -E "^\s*task_end_idx:\s*" "$CONFIG_FILE" | head -1 | sed 's/.*task_end_idx:\s*\([0-9]*\).*/\1/')
if [ -n "$CONFIG_TASK_END_IDX" ] && [ "$CONFIG_TASK_END_IDX" -gt 0 ]; then
    if [ "$CONFIG_TASK_END_IDX" -lt "$TOTAL_TASKS" ]; then
        echo "⚠️  警告: 設定ファイルのtask_end_idx ($CONFIG_TASK_END_IDX) がTOTAL_TASKS ($TOTAL_TASKS) より小さいです"
        echo "   必要に応じて --total-tasks オプションで調整してください"
    fi
fi

# 設定ファイルからmodel名を読み取り
MODEL_NAME=$(grep -A 5 "^model:" "$CONFIG_FILE" | grep -E "^\s*name:\s*" | head -1 | sed 's/.*name:\s*"\?\([^"]*\)"\?.*/\1/')
if [ -z "$MODEL_NAME" ]; then
    echo "⚠️  設定ファイルからmodel名が読み取れませんでした。"
    MODEL_NAME="unknown"
fi

# MODEL_NAMEの"/"を"-"に置き換え（ファイル名として使用するため）
MODEL_NAME_SAFE=$(echo "$MODEL_NAME" | sed 's/\//-/g')

# 設定ファイルからリソースタイプを読み取り
if [ -z "$RESOURCE_TYPE" ]; then
    # YAMLの構造を考慮して job.queue: を正しく読み取り
    RESOURCE_TYPE=$(grep -A 10 "^job:" "$CONFIG_FILE" | grep -E "^\s*queue:\s*" | head -1 | sed 's/.*queue:\s*"\?\([^"]*\)"\?.*/\1/')
    if [ -z "$RESOURCE_TYPE" ]; then
        RESOURCE_TYPE="rt_HG"  # デフォルト
        echo "⚠️  設定ファイルからjob.queueが読み取れませんでした。デフォルトで rt_HG (GPU共有) を使用"
    fi
fi

# ディレクトリの作成
mkdir -p "$RESULTS_DIR"
mkdir -p "logs/abci"
mkdir -p "job_scripts"

# ジョブ数の計算（調整後のTOTAL_TASKSを使用）
NUM_JOBS=$(( (TOTAL_TASKS + TASKS_PER_JOB - 1) / TASKS_PER_JOB ))

echo "=== ABCI バッチジョブ投入 ==="
echo "設定ファイル: $CONFIG_FILE"
echo "モデル名: $MODEL_NAME"
if [ -n "$CONFIG_TASK_END_IDX" ] && [ "$CONFIG_TASK_END_IDX" -gt 0 ]; then
    echo "設定ファイルのtask_end_idx: $CONFIG_TASK_END_IDX (参考)"
fi
echo "総タスク数: $TOTAL_TASKS"
echo "ジョブあたりのタスク数: $TASKS_PER_JOB"
echo "総ジョブ数: $NUM_JOBS"
echo "結果保存ディレクトリ: $RESULTS_DIR"
echo "ABCIグループ: $GROUP"
echo "資源タイプ: $RESOURCE_TYPE"
echo "最大実行時間: $WALLTIME"
echo "==============================="

# タイムスタンプを生成
TIMESTAMP=$(date +%m%d_%H%M)

# 個別ジョブの投入
echo "ABCIジョブを投入しています..."
JOB_IDS=()

for ((i=0; i<NUM_JOBS; i++)); do
    # 開始・終了タスクの計算
    START_IDX=$((i * TASKS_PER_JOB))
    END_IDX=$(((i + 1) * TASKS_PER_JOB))
    
    # 最後のジョブの場合、総タスク数を超えないように調整
    if [ $END_IDX -gt $TOTAL_TASKS ]; then
        END_IDX=$TOTAL_TASKS
    fi
    
    JOB_ID="${TIMESTAMP}_${i}"
    
    # 各ジョブ用のスクリプトを作成（abci_job_template.shをベース）
    cat > "job_scripts/job_${TIMESTAMP}_${MODEL_NAME_SAFE}_${i}.sh" <<EOF
#!/bin/bash
#PBS -l select=$SELECT_COUNT
#PBS -l walltime=$WALLTIME
#PBS -P $GROUP
#PBS -j oe
#PBS -m be
#PBS -N ${MODEL_NAME_SAFE}_${i}
#PBS -q $RESOURCE_TYPE
#PBS -o logs/abci/job_${TIMESTAMP}_${i}.out

# 作業ディレクトリに移動
cd \${PBS_O_WORKDIR}

# Environment Modulesの初期化
source /etc/profile.d/modules.sh

# 必要なモジュールのロード（ABCIで利用可能なモジュール）
module load cuda/12.6/12.6.1

# 仮想環境のアクティベート
source venv/bin/activate

# 環境変数の設定
export CONFIG_FILE="$CONFIG_FILE"
export START_IDX="$START_IDX"
export END_IDX="$END_IDX"
export JOB_ID="$JOB_ID"

echo "=== WebShop評価ジョブ ${i} の実行 ==="
echo "設定ファイル: \$CONFIG_FILE"
echo "タスク範囲: \$START_IDX-\$END_IDX"
echo "Job ID: \$JOB_ID"
echo "PBS Job ID: \${PBS_JOBID}"
echo "Node: \$(hostname)"
echo "実行開始時刻: \$(date)"

# 設定ファイルの存在確認
if [ ! -f "\$CONFIG_FILE" ]; then
    echo "❌ エラー: 設定ファイルが見つかりません: \$CONFIG_FILE"
    exit 1
fi

# 実行コマンドの構築
CMD_ARGS="--config \$CONFIG_FILE"
CMD_ARGS="\$CMD_ARGS --task-start-idx \$START_IDX"
CMD_ARGS="\$CMD_ARGS --task-end-idx \$END_IDX"
CMD_ARGS="\$CMD_ARGS --job-id \$JOB_ID"

# 統一されたWebShop評価スクリプトを使用
PYTHON_SCRIPT="eval_webshop.py"

echo "実行するスクリプト: \$PYTHON_SCRIPT"
echo "実行コマンド: python \$PYTHON_SCRIPT \$CMD_ARGS"

# 評価の実行
python \$PYTHON_SCRIPT \$CMD_ARGS

EXIT_CODE=\$?

echo "ジョブ ${i} が完了しました (終了コード: \$EXIT_CODE)"
echo "実行終了時刻: \$(date)"

exit \$EXIT_CODE
EOF
    
    # ジョブの投入
    JOB_SUBMIT_ID=$(qsub "job_scripts/job_${TIMESTAMP}_${MODEL_NAME_SAFE}_${i}.sh")
    if [ $? -eq 0 ]; then
        echo "ジョブ $i を投入しました: $JOB_SUBMIT_ID (タスク ${START_IDX}-${END_IDX})"
        JOB_IDS+=("$JOB_SUBMIT_ID")
    else
        echo "❌ エラー: ジョブ $i の投入に失敗しました"
    fi
    
    # 短い間隔を置く（システムへの負荷軽減）
    sleep 1
done

echo ""
echo "✅ ${#JOB_IDS[@]} 個のジョブが投入されました。"
echo ""
echo "ジョブの状態確認:"
echo "  qstat           # 自分のジョブを確認"
echo "  qgstat          # グループのジョブを確認"
echo "  qstat -f <job_id> # 詳細情報"
echo ""
echo "ジョブの削除:"   
echo "  qdel <job_id>"

echo ""
echo "全ジョブ完了後、以下のコマンドで結果を統合できます:"
echo "python merge_results.py ${RESULTS_DIR}/${TIMESTAMP}_${MODEL_NAME_SAFE}"
echo ""
echo "投入されたジョブID:"
for job_id in "${JOB_IDS[@]}"; do
    echo "  $job_id"
done

echo ""
echo "ジョブスクリプトファイル:"
for ((i=0; i<NUM_JOBS; i++)); do
    echo "  job_scripts/job_${TIMESTAMP}_${MODEL_NAME_SAFE}_${i}.sh"
done 