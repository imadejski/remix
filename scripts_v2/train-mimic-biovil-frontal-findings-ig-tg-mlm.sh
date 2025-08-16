LOSS_COMBO=ig_tg
SECTION=findings
NAME=mimic-biovil-frontal
MLM_P=0.15

is_zero=$(echo "$MLM_P == 0.0" | bc)

if [[ "$is_zero" -eq 1 ]]; then
    MLM="no-mlm"
else
    MLM="mlm"
fi

VERSION=$SECTION-$LOSS_COMBO-$MLM

python train.py fit \
--config configs/$NAME.yaml \
--model.checkpoint_path /opt/gpudata/remix/$NAME-$VERSION \
--model.loss_combo $LOSS_COMBO \
--model.use_v2 True \
--data.section $SECTION \
--data.mlm_probability $MLM_P \
--trainer.logger.name $NAME \
--trainer.logger.version $VERSION
