for i in {1..10}; do
  python code/train.py \
    --backbone="mm_dfn" \
    --name="demo6_cl_run${i}" \
    --hidden_dim=256 \
    --learning_rate=0.0001 \
    --dataset="iemocap" \
    --modalities=atv \
    --batch_size=32 \
    --epochs=60 \
    --seed=50 \
    --drop_rate=0.4 \
    --early_stopping=20 \
    --encoder_modules=transformer \
    --comet \
    --comet_api Fd1aGmcly8SdDO5Ez4DMyCIt5 \
    --comet_workspace mattam301 \
    --project_name backbone_coid
done
